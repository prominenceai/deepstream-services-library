/*
The MIT License

Copyright (c)   2023, Prominence AI, Inc.

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

#ifndef _DSL_CAPS_H
#define _DSL_CAPS_H

#include "Dsl.h"

namespace DSL
{
    /**
     * @class DslCaps
     * @brief Helper class to create GST Caps from String
     */
    class DslCaps
    {
    public:
    
        /**
         * @brief ctor for DslCaps class
         */
        DslCaps(const char* media, const char* format, 
            uint width, uint height, uint fpsN, uint fpsD, bool addMemFeature)
        {
            LOG_FUNC();
            
            // All caps strings start with media
            m_ssCaps << media;
            
            // Optionally add the Nvidia memory feature
            if (addMemFeature)
            {
                m_ssCaps << "(memory:NVMM)";
            }
            
            // Optionally add the format dimensions and frame-rate
            if (format)
            {
                m_ssCaps << "," << "format=(string)" << format;
            }
            if (width and height)
            {
                m_ssCaps << "," << "width=(int)" << width 
                    << "," << "height=(int)" << height;
            }
            if (fpsN and fpsD)
            {
                m_ssCaps << "," << "framerate=(fraction)" << fpsN << "/" << fpsD;
            }
            LOG_INFO("Creating new cap from string = ' " 
                << m_ssCaps.str().c_str() << "'");
            m_pGstCaps = gst_caps_from_string(m_ssCaps.str().c_str());
        }
        
        /**
         * @brief dtor for DslCaps class
         */
        ~DslCaps()
        {
            gst_caps_unref(m_pGstCaps);  
        }
        
        /**
         * @brief returns the const C string used to create the caps.
         * @return returns the caps string for the GstCaps helper.
         */
        const char* c_str()
        {
            return m_ssCaps.str().c_str();
        }

        /**
         * @brief & operator for the DslCaps class
         * @return returns the GstCaps pointer created from m_ssCaps.
         */
        GstCaps* operator& ()
        {
            return m_pGstCaps;
        }
        
    private:

        /**
         * @brief Stringstream to build up the caps string from ctor input params. 
         */
        std::stringstream m_ssCaps;
        
        /**
         * @brief Pointer to GST Caps created from the Stringstream m_ssCaps. 
         */
        GstCaps* m_pGstCaps; 

    };

} // namespace 

#endif // _DSL_CAPS_H
