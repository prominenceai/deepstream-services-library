/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Note: Up-to-date working examples for avformt were hard to come by.
//   - the below link was used to create the AvInputFile utilty.
// https://kibichomurage.medium.com/ffmpeg-minimum-working-example-17f68c985f0d

#include "Dsl.h"
#include "DslAvFile.h"

namespace DSL
{
    AvInputFile::AvInputFile(const char* filepath)
        : m_pFormatCtx(NULL)
        , fpsN(0)
        , fpsD(0)
        , videoWidth(0)
        , videoHeight(0)
    {
        LOG_FUNC();
        
        av_register_all();
        avformat_network_init();
        
        m_pFormatCtx = avformat_alloc_context();
        
        if (avformat_open_input(&m_pFormatCtx, filepath, NULL, NULL) < 0)
        {
            LOG_ERROR("Unable to open video file: " << filepath);
            throw std::invalid_argument("Invalid media file - failed to open.");
        }
        // Retrieve stream information
        if (avformat_find_stream_info(m_pFormatCtx, NULL) < 0)
        {
            LOG_ERROR("Unable to find stream info from file: " << filepath);
            throw std::invalid_argument("Invalid Media File - no stream info.");
        }

        bool videoCodecFound(false);
        
        for (int i = 0 ; i < m_pFormatCtx->nb_streams; i++)
        {
            AVCodecParameters* pCodecParameters = NULL;
            pCodecParameters = m_pFormatCtx->streams[i]->codecpar;

            if (pCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                // We only want the first video codec, on the chance 
                // that there are multiple? 
                if(!videoCodecFound)
                {
                    videoCodecFound = true;
                    videoWidth = pCodecParameters->width;
                    videoHeight = pCodecParameters->height;
                    fpsN = m_pFormatCtx->streams[i]->r_frame_rate.num;
                    fpsD = m_pFormatCtx->streams[i]->r_frame_rate.den;

                    LOG_INFO("Video codec data found in media file: " << filepath);
                    LOG_INFO("  dimensions : " << videoWidth << "x" << videoHeight);
                    LOG_INFO("  frame-rate : " << fpsN << "/" << fpsD);
                }
            }
        }
        if(!videoCodecFound)
        {
            LOG_ERROR("Unsupported codec found in media file: " << filepath);
            throw std::invalid_argument(
                "Invalid media file - NO video codec found.");
        }
    }
        
    AvInputFile::~AvInputFile()
    {
        LOG_FUNC();
        
        if (m_pFormatCtx)
        {
            avformat_close_input(&m_pFormatCtx);        
        }
    }

    AvJpgOutputFile::AvJpgOutputFile(
        std::shared_ptr<DslBufferSurface> pBufferSurface, 
        const char* filepath)
        : m_rgbaImage(NULL)
        , m_outfile(NULL)
        , m_pPkt(NULL)
        , m_pMjpegCodecContext(NULL)
        , m_pScaleContext(NULL)
    {
        LOG_FUNC();

        // Get the dimensions of the buffer-surface
        uint width = (&(*pBufferSurface))->surfaceList[0].width;
        uint height = (&(*pBufferSurface))->surfaceList[0].height;

        // Allocate and initialize a new array to create a compressed buffer 
        // of raw RGBA image data - i.e. with the memory alignment padding removed.
        m_rgbaImage = (uint8_t*)g_malloc0(
            (&(*pBufferSurface))->surfaceList[0].dataSize);
        
        // Initialize the source pointer to the start of the mapped surface
        uint8_t* pSrcIndex = 
            (uint8_t*)(&(*pBufferSurface))->surfaceList[0].mappedAddr.addr[0];
            
        // Initialize the destination pointer to the start of the RGBA Image buffer
        uint8_t* pDstIndex = &m_rgbaImage[0];
        
        // Get the offset of the plane within the pitch
        uint offset = 
            (&(*pBufferSurface))->surfaceList[0].planeParams.offset[0];

        // Size of each line to copy will be the plane width * the bytes/pixel
        uint copySize = (&(*pBufferSurface))->surfaceList[0].planeParams.width[0] *
            (&(*pBufferSurface))->surfaceList[0].planeParams.bytesPerPix[0];

        // For each line in the plane/buffer copy over the image data
        // and advance the source and destination buffer pointers.
        for (auto line=0; line < height; line++)
        {
            memcpy(pDstIndex, pSrcIndex+offset, copySize);
            pSrcIndex += 
                (&(*pBufferSurface))->surfaceList[0].planeParams.pitch[0];
            pDstIndex += copySize;
        }

        av_register_all();
        
        // Find the correct codec and 
        AVCodec* pMjpecCodec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
        if(!pMjpecCodec)
        {
            LOG_ERROR("Unable to find codec: AV_CODEC_ID_MJPEG");
            throw std::system_error();
        }
        
        // Allocate context from the 
        m_pMjpegCodecContext = avcodec_alloc_context3(pMjpecCodec);
        if(!m_pMjpegCodecContext)
        {
            LOG_ERROR("Failed to get context for codec: AV_CODEC_ID_MJPEG");
            throw std::system_error();
        }
        
        m_pMjpegCodecContext->bit_rate = 400000;
        m_pMjpegCodecContext->width = width;
        m_pMjpegCodecContext->height = height;
        m_pMjpegCodecContext->time_base = (AVRational){1,25};
        m_pMjpegCodecContext->pix_fmt = AV_PIX_FMT_YUVJ420P;

        if (avcodec_open2(m_pMjpegCodecContext, pMjpecCodec, NULL) < 0)
        {
            LOG_ERROR("Failed to open codec: AV_CODEC_ID_MJPEG");
            throw std::system_error();
        }

        // Allocat new source and destination frames for the conversion
        AVFrame* pSrcFrame = av_frame_alloc();
        AVFrame* pDstFrame = av_frame_alloc();
        if (!pSrcFrame or !pDstFrame)
        {
            LOG_ERROR("Failed to allocate frame-buffers");
            throw std::system_error();
        }
        
        // Setup the params for the source frame with the input RGBA buffer
        pSrcFrame->format = AV_PIX_FMT_RGBA;
        pSrcFrame->width = width;
        pSrcFrame->height = height;
        pSrcFrame->pts = 1;
        pSrcFrame->linesize[0] = width*4;
        pSrcFrame->data[0] = m_rgbaImage;
        
        pDstFrame->format = m_pMjpegCodecContext->pix_fmt;
        pDstFrame->width  = m_pMjpegCodecContext->width;
        pDstFrame->height = m_pMjpegCodecContext->height;
        pDstFrame->pts = 1;

        // allocate data for the new destination frame
        if (av_image_alloc(pDstFrame->data, pDstFrame->linesize, 
            pDstFrame->width, pDstFrame->height, 
            AV_PIX_FMT_YUV420P, 32) < 0)
        {
            LOG_ERROR("Failed to allocate new dst-image");
            throw std::system_error();
        }

        // Get context to convert the image
        m_pScaleContext = sws_getContext(width, height, AV_PIX_FMT_RGBA, 
            width, height, AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL); 
        if (!m_pScaleContext)
        {
            LOG_ERROR("Unable to get context for SwScale");
            throw std::system_error();
        }
        
        // Convert the image from RGBA to YUV420P using the scale funtion
        int retHeight = sws_scale(m_pScaleContext, pSrcFrame->data, pSrcFrame->linesize, 0,
            height, pDstFrame->data, pDstFrame->linesize);

        // --------- Start JPEG Encodeing

        // Send the converted frame to the MJPEG codec for encoding
        int retval = avcodec_send_frame(m_pMjpegCodecContext, pDstFrame);
        if ( retval < 0)
        {
            LOG_ERROR("Failed to send frame to codec: AV_CODEC_ID_MJPEG");
            throw std::system_error();
        }

        // Allocate a Packet to receive the converted data
        m_pPkt = av_packet_alloc();
        if (!m_pPkt)
        {
            LOG_ERROR("Failed to allocate Packet");
            throw std::system_error();
        }
        
        // Open the output file using the provided filepath
        m_outfile = fopen(filepath, "wb");
        while (retval >= 0)
        {
            retval = avcodec_receive_packet(m_pMjpegCodecContext, m_pPkt);
            if (retval == AVERROR(EAGAIN) || retval == AVERROR_EOF)
            {
                break;
            }
            else if (retval < 0) 
            {
                LOG_ERROR("Failed to send frame to codec: AV_CODEC_ID_MJPEG");
                throw std::system_error();
            }
            fwrite(m_pPkt->data, 1, m_pPkt->size, m_outfile);
        }
        
        av_freep(&pDstFrame->data[0]);
        av_frame_free(&pSrcFrame);
        av_frame_free(&pDstFrame);
    }
    
    AvJpgOutputFile::~AvJpgOutputFile()
    {
        LOG_FUNC();

        // close the output file and free all allocated data.   
        if (m_outfile)
        {
            fclose(m_outfile);
        }
        if (m_rgbaImage)
        {
            g_free(m_rgbaImage);
        }        
        if (m_pPkt)
        {
            av_packet_free(&m_pPkt);
        }
        if (m_pScaleContext)
        {
            sws_freeContext(m_pScaleContext);            
        }
        if(m_pMjpegCodecContext)
        {
            // We can use the codec-context to close the codec.
            avcodec_close(m_pMjpegCodecContext);
            
            // Then free the context
            avcodec_free_context(&m_pMjpegCodecContext);
        }
    }
}
