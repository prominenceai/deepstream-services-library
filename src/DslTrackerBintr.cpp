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
#include "DslTrackerBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{
    TrackerBintr::TrackerBintr(const char* name, 
        const char* llLibFileName, guint width, guint height)
        : Bintr(name)
        , m_llLibFile(llLibFileName)
        , m_width(width)
        , m_height(height)
        , m_pClientBatchMetaHandler(NULL)
        , m_pClientUserData(NULL)
    {
        LOG_FUNC();
        m_pTracker = DSL_ELEMENT_NEW(NVDS_ELEM_TRACKER, "tracker-tracker");

        m_pTracker->SetAttribute("tracker-width", m_width);
        m_pTracker->SetAttribute("tracker-height", m_height);
        m_pTracker->SetAttribute("gpu-id", m_gpuId);
        m_pTracker->SetAttribute("ll-lib-file", llLibFileName);

        AddChild(m_pTracker);

        m_pTracker->AddGhostPadToParent("sink");
        m_pTracker->AddGhostPadToParent("src");
        
        m_pGstStaticSourcePad = gst_element_get_static_pad(m_pTracker->GetGstElement(), "src");
        if (!m_pGstStaticSourcePad)
        {
            LOG_ERROR("Failed to get Static Source Pad for TrackerBintr '" << GetName() << "'");
            throw;
        }
        
        // Src Pad Probe notified on Buffer ready
        m_srcPadProbeId = gst_pad_add_probe(m_pGstStaticSourcePad, GST_PAD_PROBE_TYPE_BUFFER,
            TrackerSrcProbeCB, this, NULL);

        gst_object_unref(m_pGstStaticSourcePad);
        
        g_mutex_init(&m_srcPadProbeMutex);
    }

    TrackerBintr::~TrackerBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    
        g_mutex_clear(&m_srcPadProbeMutex);
    }

    bool TrackerBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' display to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddTrackerBintr(shared_from_this());
    }
    
    bool TrackerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("TrackerBintr '" << m_name << "' is already linked");
            return false;
        }
        // Nothing to link with single Elementr
        m_isLinked = true;
        
        return true;
    }
    
    void TrackerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("TrackerBintr '" << m_name << "' is not linked");
            return;
        }
        // Nothing to unlink with single Elementr
        m_isLinked = false;
    }

    const char* TrackerBintr::GetLibFile()
    {
        LOG_FUNC();
        
        return m_llLibFile.c_str();
    }
    
    const char* TrackerBintr::GetConfigFile()
    {
        LOG_FUNC();
        
        return m_llConfigFile.c_str();
    }
    
    void TrackerBintr::GetMaxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        m_pTracker->GetAttribute("tracker-width", &m_width);
        m_pTracker->GetAttribute("tracker-height", &m_height);
        
        *width = m_width;
        *height = m_height;
    }

    bool TrackerBintr::SetMaxDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Tiles for TrackerBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pTracker->SetAttribute("tracker-width", m_width);
        m_pTracker->SetAttribute("tracker-height", m_height);
        
        return true;
    }
    
    bool TrackerBintr::AddBatchMetaHandler(dsl_batch_meta_handler_cb pClientBatchMetaHandler, 
        void* pClientUserData)
    {
        LOG_FUNC();
        
        LOG_INFO("m_pClientBatchMetaHandler = " << pClientBatchMetaHandler);
        if (m_pClientBatchMetaHandler)
        {
            LOG_ERROR("TrackerBintr '" << GetName() << "' already has a Client Meta Batch Handler");
            return false;
        }
        m_pClientBatchMetaHandler = pClientBatchMetaHandler;
        m_pClientUserData = pClientUserData;
        
        return true;
    }
    
    bool TrackerBintr::RemoveBatchMetaHandler()
    {
        LOG_FUNC();
        
        if (!m_pClientBatchMetaHandler)
        {
            LOG_ERROR("TrackerBintr '" << GetName() << "' has no Client Meta Batch Handler");
            return false;
        }
        m_pClientBatchMetaHandler = NULL;
        m_pClientUserData = NULL;
        
        return true;
    }

    dsl_batch_meta_handler_cb TrackerBintr::GetBatchMetaHandler()
    {
        LOG_FUNC();
        
        return m_pClientBatchMetaHandler;
    }
    
    GstPadProbeReturn TrackerBintr::HandleTrackerSrcProbe(
        GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_srcPadProbeMutex);

        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            if (m_pClientBatchMetaHandler) // TODO or write ouput enabled
            {
                GstBuffer* pBuffer = (GstBuffer*)pInfo->data;
                if (!pBuffer)
                {
                    LOG_WARN("Unable to get data buffer for Tracker '" << GetName() << "'");
                    return GST_PAD_PROBE_OK;
                }
                if (m_pClientBatchMetaHandler)
                {
                    m_pClientBatchMetaHandler(pBuffer, m_pClientUserData);
                }
                // TODO if write output
            }
        }
        return GST_PAD_PROBE_OK;
    }
    
    KtlTrackerBintr::KtlTrackerBintr(const char* name, guint width, guint height)
        : TrackerBintr(name, NVDS_KLT_LIB, width, height)
    {
        LOG_FUNC();
    }
    
    IouTrackerBintr::IouTrackerBintr(const char* name, const char* configFile, guint width, guint height)
        : TrackerBintr(name, NVDS_IOU_LIB, width, height)
    {
        LOG_FUNC();

        m_llConfigFile = configFile;

        std::ifstream streamConfigFile(configFile);
        if (!streamConfigFile.good())
        {
            LOG_ERROR("IOU Tracker Config File '" << configFile << "' Not found");
            throw;
        }
        m_pTracker->SetAttribute("ll-config-file", configFile);
    }
    
    static GstPadProbeReturn TrackerSrcProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pTrackerBintr)
    {
        return static_cast<TrackerBintr*>(pTrackerBintr)->
            HandleTrackerSrcProbe(pPad, pInfo);
    }
    
}