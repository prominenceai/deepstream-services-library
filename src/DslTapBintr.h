/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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

#ifndef _DSL_TAP_BINTR_H
#define _DSL_TAP_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslElementr.h"
#include "DslRecordMgr.h"

#include <gst-nvdssr.h>

namespace DSL
{
    #define DSL_TAP_PTR std::shared_ptr<TapBintr>

    #define DSL_RECORD_TAP_PTR std::shared_ptr<RecordTapBintr>
    #define DSL_RECORD_TAP_NEW(name, outdir, container, clientListener) std::shared_ptr<RecordTapBintr>( \
        new RecordTapBintr(name, outdir, container, clientListener))

    class TapBintr : public Bintr
    {
    public: 
    
        TapBintr(const char* name);

        ~TapBintr();
  
        bool LinkToSourceTee(DSL_NODETR_PTR pTee);

        bool UnlinkFromSourceTee();
        
    protected:
    
        /**
         * @brief Queue element as sink for all Tap Bintrs.
         */
        DSL_ELEMENT_PTR m_pQueue;
    };

    //-------------------------------------------------------------------------

    class RecordTapBintr : public TapBintr, public RecordMgr
    {
    public: 
    
        RecordTapBintr(const char* name, const char* outdir, uint container, dsl_record_client_listener_cb clientListener);

        ~RecordTapBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

    private:

        /**
         * @brief Node to wrap NVIDIA's Record Bin
         */
        DSL_NODETR_PTR m_pRecordBin;
    };

}
#endif // _DSL_TAP_BINTR_H
