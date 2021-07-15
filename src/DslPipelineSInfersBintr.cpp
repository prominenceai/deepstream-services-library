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

#include "Dsl.h"
#include "DslPipelineSInfersBintr.h"

namespace DSL
{

    PipelineSInfersBintr::PipelineSInfersBintr(const char* name)
        : Bintr(name)
        , m_stop(false)
        , m_flush(false)
        , m_primaryInferUniqueId(0)
        , m_interval(0)
    {
        LOG_FUNC();

        // Single Queue and Tee element for all Secondary GIES
        m_pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "sgies_bin_tee");
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sgies_bin_queue");

        AddChild(m_pQueue);
        AddChild(m_pTee);

        m_pGstStaticSinkPad = gst_element_get_static_pad(m_pTee->GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for SInferBintr '" << GetName() << "'");
            throw;
        }
        
        m_pGstStaticSourcePad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "src");
        if (!m_pGstStaticSourcePad)
        {
            LOG_ERROR("Failed to get Static Source Pad for SInferBintr '" << GetName() << "'");
            throw;
        }
        
        // Sink Pad Probe -- added to the Tee -- is used to wait on Stream events, and to 
        // unblock the waiting Src Pad Probe on Flush or EOS
        m_sinkPadProbeId = gst_pad_add_probe(m_pGstStaticSinkPad, 
            GST_PAD_PROBE_TYPE_EVENT_BOTH, SInfersSinkProbeCB, this, NULL);

        // Src Pad Probe -- added to the Queue -- used to block the stream and wait for
        // all SGIEs to finish processing the shared buffer.
        m_srcPadProbeId = gst_pad_add_probe(m_pGstStaticSourcePad,
            (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_BOTH),
            SInfersSrcProbeCB, this, NULL);

        gst_object_unref(m_pGstStaticSinkPad);
        gst_object_unref(m_pGstStaticSourcePad);
        
        // Float the Queue sink pad as a Ghost Pad for this PipelineSInfersBintr
        m_pTee->AddGhostPadToParent("sink");
        m_pQueue->AddGhostPadToParent("src");

        
        g_mutex_init(&m_sinkPadProbeMutex);
        g_mutex_init(&m_srcPadProbeMutex);
    }
    
    PipelineSInfersBintr::~PipelineSInfersBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    
        g_mutex_clear(&m_sinkPadProbeMutex);
        g_mutex_clear(&m_srcPadProbeMutex);
    }
     
    bool PipelineSInfersBintr::AddChild(DSL_BASE_PTR pChildElement)
    {
        LOG_FUNC();
        
        return Bintr::AddChild(pChildElement);
    }

    bool PipelineSInfersBintr::AddChild(DSL_SECONDARY_INFER_PTR pChildSecondaryInfer)
    {
        LOG_FUNC();
        
        // Ensure SecondaryInfer uniqueness
        if (IsChild(pChildSecondaryInfer))
        {
            LOG_ERROR("' " << pChildSecondaryInfer->GetName() << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Add the SGIE as a child to this PipelineSInfersBintr as a Nodetr, not Bintr, 
        // as we're not adding the SGIEs GST BIN to this PipelineSInfersBintr GST BIN. 
        // Instead, we add the SGIEs child Elementrs to this PipelineSInfersBintr as a Bintr
        if (!Bintr::AddChild(pChildSecondaryInfer))
        {
            LOG_ERROR("Failed to add SecondaryInfer' " << pChildSecondaryInfer->GetName() 
                << "' as a child of '" << GetName() << "'");
            return false;
        }
        //add the SecondaryInfer's Elements as children of this Bintr
        if (!Bintr::AddChild(pChildSecondaryInfer->GetQueueElementr()) or
            !Bintr::AddChild(pChildSecondaryInfer->GetInferEngineElementr()) or
            !Bintr::AddChild(pChildSecondaryInfer->GetFakeSinkElementr()))
        {
            LOG_ERROR("Failed to add the elementrs from SecondaryInfer' " << 
                pChildSecondaryInfer->GetName() << "' as childern of '" << GetName() << "'");
            return false;
        }
        // Add the SecondaryInfer to the SInfer collection mapped by SecondaryInfer name
        m_pChildSInfers[pChildSecondaryInfer->GetName()] = pChildSecondaryInfer;

        return true;
    }
    
    bool PipelineSInfersBintr::IsChild(DSL_SECONDARY_INFER_PTR pChildSecondaryInfer)
    {
        LOG_FUNC();
        
        return (m_pChildSInfers.find(pChildSecondaryInfer->GetName()) != m_pChildSInfers.end());
    }

    bool PipelineSInfersBintr::RemoveChild(DSL_BASE_PTR pChildElement)
    {
        LOG_FUNC();
        
        // call the base function to handle the remove for Elementrs
        return Bintr::RemoveChild(pChildElement);
    }

    bool PipelineSInfersBintr::RemoveChild(DSL_SECONDARY_INFER_PTR pChildSecondaryInfer)
    {
        LOG_FUNC();

        if (!IsChild(pChildSecondaryInfer))
        {
            LOG_ERROR("' " << pChildSecondaryInfer->GetName() << "' is NOT a child of '" << GetName() << "'");
            throw;
        }
        if (pChildSecondaryInfer->IsLinkedToSource())
        {
            // unlink the sink from the Tee
            pChildSecondaryInfer->UnlinkFromSource();
        }
        // remove the SecondaryInfer's Elements as children of this Bintr
        // remove the SecondaryInfer's Elements as children of this Bintr
        if (!Bintr::RemoveChild(pChildSecondaryInfer->GetQueueElementr()) or
            !Bintr::RemoveChild(pChildSecondaryInfer->GetInferEngineElementr()) or
            !Bintr::RemoveChild(pChildSecondaryInfer->GetFakeSinkElementr()))
        {
            LOG_ERROR("Failed to add the elementrs from SecondaryInfer' " << 
                pChildSecondaryInfer->GetName() << "' as childern of '" << GetName() << "'");
            return false;
        }
        // unreference and remove from the collection
        m_pChildSInfers.erase(pChildSecondaryInfer->GetName());
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildSecondaryInfer);
    }


    bool PipelineSInfersBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("PipelineSInfersBintr '" << GetName() << "' is already linked");
            return false;
        }
        if (!m_primaryInferUniqueId)
        {
            LOG_ERROR("Unable to link PipelineSInfersBintr '" << GetName() << "' - PrimaryGie Not Set");
            return false;
        }
        if (!m_batchSize)
        {
            LOG_ERROR("Unable to link PipelineSInfersBintr '" << GetName() << "' - batch size Not Set");
            return false;
        }
        // Get a dynamic "src" pad for the Tee to link to the static "sink" pad of the shared-buffer-Queue
        GstPad* pGstRequestedSourcePad = gst_element_get_request_pad(m_pTee->GetGstElement(), "src_%u");
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        // Get the static "sink" pad for the Queue to link back with the dynamic "src" pad of the Tee
        m_pGstStaticSinkPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for SInfersBintr '" << GetName() << "'");
            return false;
        }
        // Always Link from "sink" pad back to "src" pad when linking Tees - link state is managed
        // by each individual "sink" in the one-to-many relationship 
        if (!m_pQueue->LinkToSource(m_pTee))
        {
            return false;
        }
        // TODO - recursively handle multiple levels of Secondary Inference
        for (auto const& imap: m_pChildSInfers)
        {
            if (imap.second->GetInferOnUniqueId() == m_primaryInferUniqueId)
            {
                // batch size and infer-on-gie are set to that of the Primary GIE
                if (!imap.second->SetBatchSize(m_batchSize))
                {
                    return false;
                }
                
                LOG_INFO("Linking " << m_pTee->GetName() << " from " << imap.second->GetName());
                
                // Link all SGIE Elementrs and Link back with the Primary Tee
                if (!imap.second->LinkAll() or !imap.second->LinkToSource(m_pTee))
                {
                    LOG_ERROR("PipelineSInfersBintr '" << GetName() 
                        << "' failed to Link Child SecondaryInfer '" << imap.second->GetName() << "'");
                    return false;
                }
            }
        }
        m_isLinked = true;
        return true;
    }

    void PipelineSInfersBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OsdBintr '" << GetName() << "' is not linked");
            return;
        }
        for (auto const& imap: m_pChildSInfers)
        {
            // unlink from the Tee Element
            LOG_INFO("Unlinking " << m_pTee->GetName() << " from " << imap.second->GetName());
            if (!imap.second->UnlinkFromSource())
            {
                LOG_ERROR("PipelineSInfersBintr '" << GetName() 
                    << "' failed to Unlink Child SecondaryInfer '" << imap.second->GetName() << "'");
            }
            // unink all of the ChildSecondaryInfer's Elementrs
            imap.second->UnlinkAll();
        }
        m_pQueue->UnlinkFromSource();
        m_isLinked = false;
    }

    void PipelineSInfersBintr::SetInferOnId(int id)
    {
        LOG_FUNC();
        
        m_primaryInferUniqueId = id;
    }
    
    uint PipelineSInfersBintr::GetInterval()
    {
        LOG_FUNC();
        
        return m_interval;
    }
    
    void PipelineSInfersBintr::SetInterval(uint interval)
    {
        LOG_FUNC();
        
        m_interval = interval;
    }
    
    GstPadProbeReturn PipelineSInfersBintr::HandleSInfersSinkProbe(
        GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_sinkPadProbeMutex);

        if (pInfo->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)
        {
            GstEvent *event = (GstEvent*)pInfo->data;
            
            if (event->type == GST_EVENT_EOS)
            {
                m_stop = true;
            }
        }
        return GST_PAD_PROBE_OK;
    }

    GstPadProbeReturn PipelineSInfersBintr::HandleSInfersSrcProbe(
        GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_srcPadProbeMutex);

        if (pInfo->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)
        {
            GstEvent *event = (GstEvent*)pInfo->data;
            
            if (event->type == GST_EVENT_EOS)
            {
                return GST_PAD_PROBE_OK;
            }
        }
        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            while (GST_OBJECT_REFCOUNT_VALUE(GST_BUFFER(pInfo->data)) > 1 && !m_stop && !m_flush)
            {
                gint64 endtime(g_get_monotonic_time() + G_TIME_SPAN_SECOND / 1000);
                g_cond_wait_until(&m_padWaitLock, &m_srcPadProbeMutex, endtime);
            }
        }
        return GST_PAD_PROBE_OK;
    }

    // ************************************************************************
    // Sink and Src Pad Probe Callback functions for all Pipelines...
    // Callback user-data points to the Instance of PipelineSInfersBintr
    
    static GstPadProbeReturn SInfersSinkProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSInfersBintr)
    {
        return static_cast<PipelineSInfersBintr*>(pSInfersBintr)->
            HandleSInfersSinkProbe(pPad, pInfo);
    }
    
    static GstPadProbeReturn SInfersSrcProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSInfersBintr)
    {
        return static_cast<PipelineSInfersBintr*>(pSInfersBintr)->
            HandleSInfersSrcProbe(pPad, pInfo);
    }
    
}