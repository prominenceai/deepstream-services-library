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
//   - the below link was used to create the AvFile utilty.
// https://kibichomurage.medium.com/ffmpeg-minimum-working-example-17f68c985f0d

#include "Dsl.h"
#include "DslAvFile.h"

namespace DSL
{
    AvFile::AvFile(const char* filepath)
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
        
    AvFile::~AvFile()
    {
        LOG_FUNC();
        
        if (m_pFormatCtx)
        {
            avformat_close_input(&m_pFormatCtx);        
        }
    }
    
}
