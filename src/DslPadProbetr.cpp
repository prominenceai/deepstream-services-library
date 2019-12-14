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
        , m_pClientBatchMetaHandler(NULL)
        , m_pClientUserData(NULL)
    {
        GstPad* pStaticPad = gst_element_get_static_pad(parentElement->GetGstElement(), factoryName);
        if (!pStaticPad)
        {
            LOG_ERROR("Failed to get Static Pad for PadProbetr '" << name << "'");
            throw;
        }
        
        // Src Pad Probe notified on Buffer ready
        m_padProbeId = gst_pad_add_probe(pStaticPad, GST_PAD_PROBE_TYPE_BUFFER,
            PadProbeCB, this, NULL);

        gst_object_unref(pStaticPad);
        
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
        
        LOG_INFO("m_pClientBatchMetaHandler = " << pClientBatchMetaHandler);
        if (m_pClientBatchMetaHandler)
        {
            LOG_ERROR("PadProbetr '" << m_name << "' already has a Client Meta Batch Handler");
            return false;
        }
        m_pClientBatchMetaHandler = pClientBatchMetaHandler;
        m_pClientUserData = pClientUserData;
        
        return true;
    }
    
    bool PadProbetr::RemoveBatchMetaHandler()
    {
        LOG_FUNC();
        
        if (!m_pClientBatchMetaHandler)
        {
            LOG_ERROR("PadProbetr '" << m_name << "' has no Client Meta Batch Handler");
            return false;
        }
        m_pClientBatchMetaHandler = NULL;
        m_pClientUserData = NULL;
        
        return true;
    }

    dsl_batch_meta_handler_cb PadProbetr::GetBatchMetaHandler()
    {
        LOG_FUNC();
        
        return m_pClientBatchMetaHandler;
    }

    GstPadProbeReturn PadProbetr::HandlePadProbe(GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padProbeMutex);

        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            if (m_pClientBatchMetaHandler) // TODO or write ouput enabled
            {
                GstBuffer* pBuffer = (GstBuffer*)pInfo->data;
                if (!pBuffer)
                {
                    LOG_WARN("Unable to get data buffer for PadProbetr '" << m_name << "'");
                    return GST_PAD_PROBE_OK;
                }
                if (m_pClientBatchMetaHandler)
                {
                    // Remove the client on false return
                    if (!m_pClientBatchMetaHandler(pBuffer, m_pClientUserData))
                    {
                        LOG_INFO("Removing client batch meta handler for PadProbetr '" << m_name << "'");
                        m_pClientBatchMetaHandler = NULL;
                        m_pClientUserData = NULL;
                    }
                }
                // TODO if write output
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
    
