/*
The MIT License

Copyright (c) 2021-2023, Prominence AI, Inc.

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

#ifndef _DSL_PIPELINE_XWIN_MGR_H
#define _DSL_PIPELINE_XWIN_MGR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"

namespace DSL
{

     class PipelineBusSyncMgr
    {
    public: 
    
        PipelineBusSyncMgr(const GstObject* pGstPipeline);

        ~PipelineBusSyncMgr();
        
        /**
         * @brief handles incoming sync messages
         * @param[in] message incoming message to process
         * @return [GST_BUS_PASS|GST_BUS_FAIL]
         */
        GstBusSyncReply HandleBusSyncMessage(GstMessage* pMessage);
        
    protected:
    
        /**
         * @brief Shared client cb mutex - owned by the the PipelineBusSyncMgr
         * but shared amoungst all child Window Sinks. Mutex will clear
         * on last unreference.
         */
        std::shared_ptr<DslMutex> m_pSharedClientCbMutex;

    private:

        /**
         * @brief mutex to prevent callback reentry
         */
        DslMutex m_busSyncMutex;
    };
    
    /**
     * @brief 
     * @param[in] bus instance pointer
     * @param[in] message incoming message packet to process
     * @param[in] pData pipeline instance pointer
     * @return [GST_BUS_PASS|GST_BUS_FAIL]
     */
    static GstBusSyncReply bus_sync_handler(
        GstBus* bus, GstMessage* pMessage, gpointer pData);
}

#endif //  DSL_PIPELINE_XWIN_MGR_H
