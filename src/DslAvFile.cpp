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

#include "Dsl.h"
#include "DslAvFile.h"

namespace DSL
{
    AvFile::AvFile(const char* filepath)
    : pFormatCtx(NULL)
    {
        LOG_FUNC();
        
        av_register_all();
        avformat_network_init();
        
        pFormatCtx = avformat_alloc_context();
        
        if (avformat_open_input(&pFormatCtx, filepath, NULL, NULL) < 0)
        {
            LOG_ERROR("Unable to open video file: " << filepath);
            throw std::invalid_argument("Invalid media file - failed to open.");
        }
        // Retrieve stream information
        if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
        {
            LOG_ERROR("Unable to find stream info from file: " << filepath);
            throw std::invalid_argument("Invalid Media File - no stream info.");
        }

        bool videoCodecFound(false);
        
        for (int i = 0 ; i < pFormatCtx->nb_streams; i++)
        {
            AVCodecParameters* localCodecParameters = NULL;
            localCodecParameters = pFormatCtx->streams[i]->codecpar;

            fpsN = pFormatCtx->streams[i]->r_frame_rate.num;
            fpsD = pFormatCtx->streams[i]->r_frame_rate.den;

            AVCodec *localCodec = NULL;
            localCodec = avcodec_find_decoder(localCodecParameters->codec_id);
            if (localCodec == NULL)
            {
                LOG_ERROR("Unsupported codec found in media file: " << filepath);
                throw std::invalid_argument(
                    "Invalid media file - unsupported codec.");
            }
            if (localCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                if(!videoCodecFound)
                {
                    videoCodecFound = true;
                    videoWidth = localCodecParameters->width;
                    videoHeight = localCodecParameters->height;
                    LOG_INFO("Video codec found in media file: " << filepath);
                    LOG_INFO("  dimensions      : " 
                        << videoWidth << "x" << videoHeight);
                    LOG_INFO("  Video frame-rate: " 
                        << fpsN << "/" << fpsD);
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
        
    AvFile::~AvFile()
    {
        LOG_FUNC();
        
        if (pFormatCtx)
        {
            avformat_close_input(&pFormatCtx);        
        }
    }
    
}
