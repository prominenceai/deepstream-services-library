/*
The MIT License

Copyright (c) 2023, Prominence AI, Inc.

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

#ifndef _DSL_AV_FILE_H
#define _DSL_AV_FILE_H

extern "C" { 
#include <libavformat/avformat.h>
}

namespace DSL
{
    /**
     * @struct AvFile 
     * @brief Utility to read the video dimensions and frame-rate for a given
     * media file. Has been tested with ".mp4", ".mov", .jpg", and ".mjpeg"
     */
    struct AvFile
    {
        /**
         * @brief ctor for the AvFile class
         * @param uri relative or absolute path to the media file to query.
         */
        AvFile(const char* filepath);
        
        /**
         * @brief dtor for the AvFile class
         */
        ~AvFile();

        /**
         * @brief frames-per second fractional numerator for the AV File
         */
        uint fpsN;
        
        /**
         * @brief frames-per second fractional numerator for the AV File
         */
        uint fpsD;
            
        /**
         * @brief width of the Video stream in pixels
         */
        uint videoWidth;
        
        /**
         * @brief height of the Video stream in pixels
         */
        uint videoHeight;
        
    private:
    
        /**
         * @brief pointer to a AV Format Context populated with avformat_open_input.
         */
        AVFormatContext* m_pFormatCtx;
        
    };
    
}

#endif // _DSL_AV_FILE_H