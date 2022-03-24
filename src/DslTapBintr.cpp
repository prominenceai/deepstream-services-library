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

#include <nvbufsurftransform.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"

#include "Dsl.h"
#include "DslTapBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    TapBintr::TapBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW("queue", name);
        AddChild(m_pQueue);
        m_pQueue->AddGhostPadToParent("sink");
    }

    TapBintr::~TapBintr()
    {
        LOG_FUNC();
    }

    bool TapBintr::LinkToSourceTee(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();

        LOG_INFO("Linking Tap '" << GetName() << "' to source Tee '" << pTee->GetName() << "'");
        return Bintr::LinkToSourceTee(pTee, "src_%u");
    }
    
    bool TapBintr::UnlinkFromSourceTee()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!IsLinkedToSource())
        {
            LOG_ERROR("TapBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        LOG_INFO("Unlinking and releasing requested Source Pad for TapBintr " << GetName());
        return Bintr::UnlinkFromSourceTee();
    }

    //-------------------------------------------------------------------------
    
    RecordTapBintr::RecordTapBintr(const char* name, const char* outdir, 
        uint container, dsl_record_client_listener_cb clientListener)
        : TapBintr(name)
        , RecordMgr(name, outdir, m_gpuId, container, clientListener)
    {
        LOG_FUNC();
    }
    
    RecordTapBintr::~RecordTapBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool RecordTapBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RecordTapBintr '" << GetName() << "' is already linked");
            return false;
        }

        if (!CreateContext())
        {
            return false;
        }
        
        m_pRecordBin = DSL_NODETR_NEW("record-bin");
        m_pRecordBin->SetGstObject(GST_OBJECT(m_pContext->recordbin));
            
        AddChild(m_pRecordBin);

        GstPad* srcPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "src");
        GstPad* sinkPad = gst_element_get_static_pad(m_pRecordBin->GetGstElement(), "sink");
        
        if (gst_pad_link(srcPad, sinkPad) != GST_PAD_LINK_OK)
        {
            LOG_ERROR("Failed to link parser to record-bin new RecordTapBintr '" << GetName() << "'");
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void RecordTapBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("RecordTapBintr '" << GetName() << "' is not linked");
            return;
        }
        DestroyContext();

        GstPad* srcPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "src");
        GstPad* sinkPad = gst_element_get_static_pad(m_pRecordBin->GetGstElement(), "sink");
        
        gst_pad_unlink(srcPad, sinkPad);

        RemoveChild(m_pRecordBin);
        
        m_pRecordBin = nullptr;
        
        m_isLinked = false;
    }
    
    void RecordTapBintr::HandleEos()
    {
        LOG_FUNC();
        
        if (IsOn())
        {
            LOG_INFO("RecordTapBintr '" << GetName() 
                << "' is in session, stopping to handle the EOS");
            StopSession(true);
        }
    }

}