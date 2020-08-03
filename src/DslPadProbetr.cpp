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
#include "DslPadProbetr.h"

namespace DSL
{
    PadProbetr::PadProbetr(const char* name, const char* factoryName, DSL_ELEMENT_PTR parentElement)
        : m_name(name)
        , m_factoryName(factoryName)
        , m_pParentGstElement(parentElement->GetGstElement())
        , m_padProbeId(0)
    {
        g_mutex_init(&m_padProbeMutex);
    }

    PadProbetr::~PadProbetr()
    {
        LOG_FUNC();

        g_mutex_clear(&m_padProbeMutex);
    }

    bool PadProbetr::AddBatchMetaHandler(dsl_batch_meta_handler_cb pClientBatchMetaHandler, 
        void* pClientUserData)
    {
        LOG_FUNC();
        
        if (IsChild(pClientBatchMetaHandler))
        {
            LOG_ERROR("Client Meta Batch Handler is already a child of PadProbetr '" << m_name << "'");
            return false;
        }
        
        if (!m_padProbeId)
        {
            GstPad* pStaticPad = gst_element_get_static_pad(m_pParentGstElement, m_factoryName.c_str());
            if (!pStaticPad)
            {
                LOG_ERROR("Failed to get Static Pad for PadProbetr '" << m_name << "'");
                return false;
            }
        
            GstPadProbeType probeType = (GstPadProbeType)(GST_PAD_PROBE_TYPE_BUFFER);
            
            // Src Pad Probe notified on Buffer ready
            m_padProbeId = gst_pad_add_probe(pStaticPad, probeType,
                PadProbeCB, this, NULL);

            gst_object_unref(pStaticPad);
        }
        
        m_pClientBatchMetaHandlers[pClientBatchMetaHandler] = pClientUserData;

        return true;
    }
    
    bool PadProbetr::RemoveBatchMetaHandler(dsl_batch_meta_handler_cb pClientBatchMetaHandler)
    {
        LOG_FUNC();
        
        if (!IsChild(pClientBatchMetaHandler))
        {
            LOG_ERROR("Client Meta Batch Handler is not owned by PadProbetr '" << m_name << "'");
            return false;
        }
        m_pClientBatchMetaHandlers.erase(pClientBatchMetaHandler);
        
        return true;
    }

    bool PadProbetr::IsChild(dsl_batch_meta_handler_cb pClientBatchMetaHandler)
    {
        LOG_FUNC();
        
        return (m_pClientBatchMetaHandlers.find(pClientBatchMetaHandler) != m_pClientBatchMetaHandlers.end());
    }

    GstPadProbeReturn PadProbetr::HandlePadProbe(GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padProbeMutex);
        
        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            if (m_pClientBatchMetaHandlers.size())
            {
                GstBuffer* pBuffer = (GstBuffer*)pInfo->data;
                if (!pBuffer)
                {
                    LOG_WARN("Unable to get data buffer for PadProbetr '" << m_name << "'");
                    return GST_PAD_PROBE_OK;
                }
                for (auto const& imap: m_pClientBatchMetaHandlers)
                {
                    // Remove the client on false return
                    if (!imap.first(pBuffer, imap.second))
                    {
                        LOG_INFO("Removing client batch meta handler for PadProbetr '" << m_name << "'");
                        RemoveBatchMetaHandler(imap.first);
                    }
                }
            }
        }
        return GST_PAD_PROBE_OK;
    }

    static GstPadProbeReturn PadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pPadProbetr)
    {
        return static_cast<PadProbetr*>(pPadProbetr)->
            HandlePadProbe(pPad, pInfo);
    }
} // DSL
    
