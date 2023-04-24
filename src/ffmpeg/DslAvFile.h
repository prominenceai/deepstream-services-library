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

#include "DslSurfaceTransform.h"

extern "C" { 
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
}

namespace DSL
{
    /**
     * @class AvInputFile 
     * @brief Utility to read the video dimensions and frame-rate for a given
     * media file. Has been tested with ".mp4", ".mov", .jpg", and ".mjpeg"
     */
    class AvInputFile
    {
    public:
        /**
         * @brief ctor for the AvInputFile class
         * @param uri relative or absolute path to the media file to query.
         */
        AvInputFile(const char* filepath);
        
        /**
         * @brief dtor for the AvInputFile class
         */
        ~AvInputFile();

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

    /**
     * @class AvJpgOutputFile
     * @brief Implements a utility class used to convert an RGBA image-buffer
     * into a JPEG Image file. 
     */
    class AvJpgOutputFile
    {
    public:
    
        /**
         * @brief ctor for the AvJpgOutputFile utility class.
         * @param[in] pBufferSurface machine aligned surface buffer.
         * @param[in] filepath for the JPEG output file to save.
         */
        AvJpgOutputFile(std::shared_ptr<DslBufferSurface> pBufferSurface, 
            const char* filepath);
        
        /**
         * @brief ctor for the AvJpgOutputFile utility class.
         */
        ~AvJpgOutputFile();
        
    private:
    
        /**
         * @brief buffer for the packed RGBA Image.
         */
        uint8_t* m_rgbaImage;
        
        /**
         * @brief Handle to opened output file.
         */
        FILE* m_outfile;
        
        /**
         * @brief Packet to receive the converted MJPEG data.
         */
        AVPacket* m_pPkt;
        
        /**
         * @brief MJPEG codec context pointer to provide context for all Codec calls.
         */
        AVCodecContext* m_pMjpegCodecContext;
        
        /**
         * @brief SW Scale utility context to provide context all Scale/format calls.
         */
        SwsContext* m_pScaleContext;
    };
}

#endif // _DSL_AV_FILE_H