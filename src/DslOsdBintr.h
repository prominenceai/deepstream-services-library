
/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

#ifndef _DSL_OSD_BINTR_H
#define _DSL_OSD_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"

namespace DSL
{
    
    class OsdBintr : public Bintr
    {
    public: 
    
        OsdBintr(const char* osd, gboolean isClockEnabled);

        ~OsdBintr();

        void AddToPipeline(std::shared_ptr<Pipeline> pPipeline);
        
    private:

        gboolean m_isClockEnabled;
        
        static std::string m_sClockFont;
        static guint m_sClockFontSize;
        static guint m_sClockOffsetX;
        static guint m_sClockOffsetY;
        static guint m_sClockColor;
        
        /**
         @brief
         */
        guint m_processMode;
        
        GstElement* m_pQueue;
        GstElement* m_pVidConv;
        GstElement* m_pCapsFilter;
        GstElement* m_pConvQueue;
        GstElement* m_pOsd;
    
    };
}

#endif // _DSL_OSD_BINTR_H