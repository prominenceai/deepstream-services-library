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
        , m_kittiOutputEnabled(false)
    {
        GstPad* pStaticPad = gst_element_get_static_pad(parentElement->GetGstElement(), factoryName);
        if (!pStaticPad)
        {
            LOG_ERROR("Failed to get Static Pad for PadProbetr '" << name << "'");
            throw;
        }
        
        GstPadProbeType probeType = (GstPadProbeType)(GST_PAD_PROBE_TYPE_BLOCK | GST_PAD_PROBE_TYPE_BUFFER);
        
        // Src Pad Probe notified on Buffer ready
        m_padProbeId = gst_pad_add_probe(pStaticPad, probeType,
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
        
        if (IsChild(pClientBatchMetaHandler))
        {
            LOG_ERROR("Client Meta Batch Handler is already a child of PadProbetr '" << m_name << "'");
            return false;
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

    bool PadProbetr::SetKittiOutputEnabled(bool enabled, const char* path)
    {
        LOG_FUNC();
        
        if (enabled)
        {
            struct stat info;
            if( stat(path, &info) != 0 )
            {
                LOG_ERROR("Unable to access path '" << path << "' for PadProbetr '" << m_name << "'");
                return false;
            }
            else if(info.st_mode & S_IFDIR)
            {
                LOG_INFO("Enabling Kitti output to path '" << path << "' for PadProbet '" << m_name << "'");
                m_kittiOutputPath.assign(path);
            }
            else
            {
                LOG_ERROR("Unable to access path '" << path << "' for GieBintr '" << m_name << "'");
                return false;
            }
        }
        else
        {
            LOG_INFO("Disabling Kitti output for PadProbetr '" << m_name << "'");
            m_kittiOutputPath.clear();
        }
        m_kittiOutputEnabled = enabled;
        return true;
    }

    GstPadProbeReturn PadProbetr::HandlePadProbe(GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padProbeMutex);
        
        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            if (m_pClientBatchMetaHandlers.size() or m_kittiOutputEnabled)
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
                if (m_kittiOutputEnabled)
                {
                    
                }
                // TODO if write output
            }
        }
        return GST_PAD_PROBE_PASS;
    }

    static GstPadProbeReturn PadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pPadProbetr)
    {
        return static_cast<PadProbetr*>(pPadProbetr)->
            HandlePadProbe(pPad, pInfo);
    }
} // DSL
    
