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

#include "Dsl.h"
#include "DslPipelineSGiesBintr.h"

namespace DSL
{

    PipelineSecondaryGiesBintr::PipelineSecondaryGiesBintr(const char* name)
        : Bintr(name)
        , m_stop(false)
        , m_flush(false)
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
            LOG_ERROR("Failed to get Static Sink Pad for SecondaryGiesBintr '" << GetName() << "'");
            throw;
        }
        
        m_pGstStaticSourcePad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "src");
        if (!m_pGstStaticSourcePad)
        {
            LOG_ERROR("Failed to get Static Source Pad for SecondaryGiesBintr '" << GetName() << "'");
            throw;
        }
        
        // Sink Pad Probe -- added to the Tee -- is used to wait on Stream events, and to 
        // unblock the waiting Src Pad Probe on Flush or EOS
        m_sinkPadProbeId = gst_pad_add_probe(m_pGstStaticSinkPad, 
            GST_PAD_PROBE_TYPE_EVENT_BOTH, SecondaryGiesSinkProbeCB, this, NULL);

        // Src Pad Probe -- added to the Queue -- used to block the stream and wait for
        // all SGIEs to finish processing the shared buffer.
        m_srcPadProbeId = gst_pad_add_probe(m_pGstStaticSourcePad,
            (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_BOTH),
            SecondaryGiesSrcProbeCB, this, NULL);

        gst_object_unref(m_pGstStaticSinkPad);
        gst_object_unref(m_pGstStaticSourcePad);
        
        // Float the Queue sink pad as a Ghost Pad for this PipelineSecondaryGiesBintr
        m_pTee->AddGhostPadToParent("sink");
        m_pQueue->AddGhostPadToParent("src");

        
        g_mutex_init(&m_sinkPadProbeMutex);
        g_mutex_init(&m_srcPadProbeMutex);
    }
    
    PipelineSecondaryGiesBintr::~PipelineSecondaryGiesBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    
        g_mutex_clear(&m_sinkPadProbeMutex);
        g_mutex_clear(&m_srcPadProbeMutex);
    }
     
    void PipelineSecondaryGiesBintr::SetPrimaryGieName(const char* name)
    {
        LOG_FUNC();
        
        m_primaryGieName = name;
    }
    
    bool PipelineSecondaryGiesBintr::AddChild(DSL_NODETR_PTR pChildElement)
    {
        LOG_FUNC();
        
        return Bintr::AddChild(pChildElement);
    }

    bool PipelineSecondaryGiesBintr::AddChild(DSL_SECONDARY_GIE_PTR pChildSecondaryGie)
    {
        LOG_FUNC();
        
        // Ensure SecondaryGie uniqueness
        if (IsChild(pChildSecondaryGie))
        {
            LOG_ERROR("' " << pChildSecondaryGie->GetName() << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Add the SGIE as a child to this PipelineSecondaryGiesBintr as a Nodetr, not Bintr, 
        // as we're not adding the SGIEs GST BIN to this PipelineSecondaryGiesBintr GST BIN. 
        // Instead, we add the SGIEs child Elementrs to this PipelineSecondaryGiesBintr as a Bintr
        if (!Bintr::AddChild(pChildSecondaryGie))
        {
            LOG_ERROR("Failed to add SecondaryGie' " << pChildSecondaryGie->GetName() 
                << "' as a child of '" << GetName() << "'");
            return false;
        }
        //add the SecondaryGie's Elements as children of this Bintr
        if (!Bintr::AddChild(pChildSecondaryGie->GetQueueElementr()) or
            !Bintr::AddChild(pChildSecondaryGie->GetInferEngineElementr()) or
            !Bintr::AddChild(pChildSecondaryGie->GetFakeSinkElementr()))
        {
            LOG_ERROR("Failed to add the elementrs from SecondaryGie' " << 
                pChildSecondaryGie->GetName() << "' as childern of '" << GetName() << "'");
            return false;
        }
        // Add the SecondaryGie to the SecondaryGies collection mapped by SecondaryGie name
        m_pChildSecondaryGies[pChildSecondaryGie->GetName()] = pChildSecondaryGie;

        return true;
    }
    
    bool PipelineSecondaryGiesBintr::IsChild(DSL_SECONDARY_GIE_PTR pChildSecondaryGie)
    {
        LOG_FUNC();
        
        return (bool)m_pChildSecondaryGies[pChildSecondaryGie->GetName()];
    }

    bool PipelineSecondaryGiesBintr::RemoveChild(DSL_NODETR_PTR pChildElement)
    {
        LOG_FUNC();
        
        // call the base function to handle the remove for Elementrs
        return Bintr::RemoveChild(pChildElement);
    }

    bool PipelineSecondaryGiesBintr::RemoveChild(DSL_SECONDARY_GIE_PTR pChildSecondaryGie)
    {
        LOG_FUNC();

        if (!IsChild(pChildSecondaryGie))
        {
            LOG_ERROR("' " << pChildSecondaryGie->GetName() << "' is NOT a child of '" << GetName() << "'");
            throw;
        }
        if (pChildSecondaryGie->IsLinkedToSource())
        {
            // unlink the sink from the Tee
            pChildSecondaryGie->UnlinkFromSource();
        }

        // remove the SecondaryGie's Elements as children of this Bintr
        // remove the SecondaryGie's Elements as children of this Bintr
        if (!Bintr::RemoveChild(pChildSecondaryGie->GetQueueElementr()) or
            !Bintr::RemoveChild(pChildSecondaryGie->GetInferEngineElementr()) or
            !Bintr::RemoveChild(pChildSecondaryGie->GetFakeSinkElementr()))
        {
            LOG_ERROR("Failed to add the elementrs from SecondaryGie' " << 
                pChildSecondaryGie->GetName() << "' as childern of '" << GetName() << "'");
            return false;
        }
        
        // unreference and remove from the collection
        m_pChildSecondaryGies.erase(pChildSecondaryGie->GetName());
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildSecondaryGie);
    }


    bool PipelineSecondaryGiesBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("PipelineSecondaryGiesBintr '" << GetName() << "' is already linked");
            return false;
        }
        
        GstPad* pGstRequestedSourcePad = gst_element_get_request_pad(m_pTee->GetGstElement(), "src_%u");
            
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        
        m_pGstStaticSinkPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for SecondaryGiesBintr '" << GetName() << "'");
            return false;
        }
        
        if (!gst_element_link(m_pTee->GetGstElement(), m_pQueue->GetGstElement()))
        {
            return false;
        }
        
        for (auto const& imap: m_pChildSecondaryGies)
        {
            if (!imap.second->LinkAll() or !imap.second->LinkToSource(m_pTee))
            {
                LOG_ERROR("PipelineSecondaryGiesBintr '" << GetName() 
                    << "' failed to Link Child SecondaryGie '" << imap.second->GetName() << "'");
                return false;
            }
        }
        m_isLinked = true;
        return true;
    }

    void PipelineSecondaryGiesBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OsdBintr '" << GetName() << "' is not linked");
            return;
        }
        for (auto const& imap: m_pChildSecondaryGies)
        {
            // unlink from the Tee Element
            LOG_INFO("Unlinking " << m_pTee->GetName() << " from " << imap.second->GetName());
            if (!imap.second->UnlinkFromSource())
            {
                LOG_ERROR("PipelineSecondaryGiesBintr '" << GetName() 
                    << "' failed to Unlink Child SecondaryGie '" << imap.second->GetName() << "'");
                return;
            }
            // unink all of the ChildSecondaryGie's Elementrs and reset the unique Id
            imap.second->UnlinkAll();
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }
    
    void PipelineSecondaryGiesBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        for (auto const& imap: m_pChildSecondaryGies)
        {
            imap.second->SetBatchSize(batchSize);
        }
    }
    
    GstPadProbeReturn PipelineSecondaryGiesBintr::HandleSecondaryGiesSinkProbe(
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

    GstPadProbeReturn PipelineSecondaryGiesBintr::HandleSecondaryGiesSrcProbe(
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

    static GstPadProbeReturn SecondaryGiesSinkProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pGiesBintr)
    {
        return static_cast<PipelineSecondaryGiesBintr*>(pGiesBintr)->
            HandleSecondaryGiesSinkProbe(pPad, pInfo);
    }
    
    static GstPadProbeReturn SecondaryGiesSrcProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pGiesBintr)
    {
        return static_cast<PipelineSecondaryGiesBintr*>(pGiesBintr)->
            HandleSecondaryGiesSrcProbe(pPad, pInfo);
    }
    
}