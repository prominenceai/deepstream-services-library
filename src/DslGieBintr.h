

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

#ifndef _DSL_GIE_BINTR_H
#define _DSL_GIE_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"
#include "DslElementr.h"

namespace DSL
{
    
    class GieBintr : public Bintr
    {
    public: 
    
        GieBintr(const char* osd, const char* inferConfigFile,
            guint batchSize, guint interval, guint uniqueId, guint gpuId, 
            const char* modelEngineFile, const char* rawOutputDir);

        ~GieBintr();

        void LinkAll();
        
        void UnlinkAll();
        
        void AddToParent(std::shared_ptr<Bintr> pParentBintr);

    private:

        gboolean m_isClockEnabled;
        
        static std::string m_sClockFont;
        static guint m_sClockFontSize;
        
        guint m_batchSize;
        
        guint m_interval;
        
        guint m_uniqueId;

        std::string m_inferConfigFile;
        
        std::string m_modelEngineFile;

        std::string m_rawOutputDir;
        
        
        /**
         @brief
         */
        guint m_processMode;
        
        DSL_ELEMENT_PTR  m_pQueue;
        DSL_ELEMENT_PTR  m_pVidConv;
        DSL_ELEMENT_PTR  m_pClassifier;
    
    };
}

#endif // _DSL_GIE_BINTR_H