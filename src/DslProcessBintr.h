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

#ifndef _DSL_PROCESS_BINTR_H
#define _DSL_PROCESS_BINTR_H

#include "DslSinkBintr.h"
#include "DslOsdBintr.h"
    
namespace DSL 
{

    /**
     * @class ProcessBintr
     * @brief 
     */
    class ProcessBintr : public Bintr
    {
    public:
    
        /** 
         * 
         */
        ProcessBintr(const char* name);
        ~ProcessBintr();

        /**
         * @brief Adds a Sink Bintr to this Process Bintr
         * @param[in] pSinkBintr
         */
        void AddSinkBintr(std::shared_ptr<Bintr> pSinkBintr);

        /**
         * @brief 
         * @param[in] pOsdBintr
         */
        void AddOsdBintr(std::shared_ptr<Bintr> pOsdBintr);
        
        /**
         * @brief 
         */
        void AddSinkGhostPad();

        
    private:
    
        /**
         * @brief one or more Sinks for this Process bintr
         */
        std::shared_ptr<SinksBintr> m_pSinksBintr;
        
        /**
         * @brief optional OSD for this Process bintr
         */
        std::shared_ptr<OsdBintr> m_pOsdBintr;
        
    };
}

#endif // _DSL_PROCESS_BINTR_H