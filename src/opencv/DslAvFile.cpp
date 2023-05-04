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
        : fpsN(0)
        , fpsD(0)
        , videoWidth(0)
        , videoHeight(0)
    {
        LOG_FUNC();

        m_vidCap.open(filepath, cv::CAP_ANY);

        if (!m_vidCap.isOpened())
        {
            LOG_ERROR("Unable to open video file: " << filepath);
            throw std::invalid_argument("Invalid media file - failed to open.");
        }
        videoWidth = m_vidCap.get(cv::CAP_PROP_FRAME_WIDTH);
        videoHeight = m_vidCap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        fpsN =  m_vidCap.get(cv::CAP_PROP_FPS);
        fpsD = 1;
    }
        
    AvInputFile::~AvInputFile()
    {
        LOG_FUNC();

        m_vidCap.release();
    }

    AvJpgOutputFile::AvJpgOutputFile(
        std::shared_ptr<DslBufferSurface> pBufferSurface, 
        const char* filepath)
        : m_pBgrFrame(NULL)
    {
        LOG_FUNC();

        // Get the dimensions of the buffer-surface
        uint width = (&(*pBufferSurface))->surfaceList[0].width;
        uint height = (&(*pBufferSurface))->surfaceList[0].height;

        // New background Mat for our image
        m_pBgrFrame = new cv::Mat(cv::Size(width, height), CV_8UC3);

        // Use openCV to remove padding
        cv::Mat in_mat = cv::Mat(height, width, CV_8UC4, 
            (&(*pBufferSurface))->surfaceList[0].mappedAddr.addr[0],
            (&(*pBufferSurface))->surfaceList[0].pitch);

        // Convert the RGBA buffer to BGR
#if (CV_MAJOR_VERSION >= 4)
        cv::cvtColor (in_mat, *m_pBgrFrame, cv::COLOR_RGBA2BGR);
#else
        cv::cvtColor(in_mat, *m_pBgrFrame, CV_RGBA2BGR);
#endif

        cv::imwrite(filepath, *m_pBgrFrame);        
    }
    
    AvJpgOutputFile::~AvJpgOutputFile()
    {
        LOG_FUNC();
        
        if (m_pBgrFrame)
        {
            delete m_pBgrFrame;
        }
    }
}
