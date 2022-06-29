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
#include "DslPadProbeHandler.h"
#include "DslBase.h"
#include "DslBintr.h"

namespace DSL
{
    //-------------------------------------------------------------------------------------------------------
    
    PadProbeHandler::PadProbeHandler(const char* name)
        : Base(name)
        , m_isEnabled(false)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_padHandlerMutex);
    }

    PadProbeHandler::~PadProbeHandler()
    {
        LOG_FUNC();

        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);
            RemoveAllChildren();
        }
        g_mutex_clear(&m_padHandlerMutex);
    }
    
    bool PadProbeHandler::AddToParent(DSL_BASE_PTR pParent, uint pad)
    {
        LOG_FUNC();
        
        DSL_BINTR_PTR pParentBintr = 
            std::dynamic_pointer_cast<Bintr>(pParent);
            
        if (!pParentBintr->AddPadProbeHandler(shared_from_this(), pad))
        {
            LOG_ERROR("Failed to add PadProbeHandler '" << GetName() << 
                "' to Parent '" << pParentBintr->GetName() << "'");
            return false;
        }
        AssignParentName(pParentBintr->GetName());
        return true;
    }

    bool PadProbeHandler::RemoveFromParent(DSL_BASE_PTR pParent, uint pad)
    {
        LOG_FUNC();
        
        DSL_BINTR_PTR pParentBintr = 
            std::dynamic_pointer_cast<Bintr>(pParent);
        
        if (!pParentBintr->RemovePadProbeHandler(shared_from_this(), pad))
        {
            LOG_ERROR("Failed to remove PadProbeHandler '" << GetName() << 
                "' from Parent '" << pParentBintr->GetName() << "'");
            return false;
        }
        ClearParentName();
        return true;
    }
    

    bool PadProbeHandler::GetEnabled()
    {
        LOG_FUNC();
        
        return m_isEnabled;
    }
    
    bool PadProbeHandler::SetEnabled(bool enabled)
    {
        LOG_FUNC();

        if (m_isEnabled == enabled)
        {
            LOG_ERROR("Can't set Handler Enabled to the same value of " 
                << enabled << " for PadProbeHandler '" << GetName() << "' ");
            return false;
        }
        m_isEnabled = enabled;
        return true;
    }

    //----------------------------------------------------------------------------------------------

    OdePadProbeHandler::OdePadProbeHandler(const char* name)
        : PadProbeHandler(name)
        , m_nextTriggerIndex(0)
        , m_displayMetaAllocSize(1)
    {
        LOG_FUNC();
        
        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    OdePadProbeHandler::~OdePadProbeHandler()
    {
        LOG_FUNC();
    }

    bool OdePadProbeHandler::AddChild(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (!Base::AddChild(pChild))
        {
            return false;
        }
        
        // increment next index, assign to the Trigger
        pChild->SetIndex(++m_nextTriggerIndex);

        // Add the child to the Indexed map 
        m_pChildrenIndexed[m_nextTriggerIndex] = pChild;
        
        return true;
    }

    bool OdePadProbeHandler::RemoveChild(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (!Base::RemoveChild(pChild))
        {
            return false;
        }
        
        // Remove the the child from Indexed map
        m_pChildrenIndexed.erase(pChild->GetIndex());
        
        return true;
    }

    void OdePadProbeHandler::RemoveAllChildren()
    {
        LOG_FUNC();
        
        Base::RemoveAllChildren();
        
        // Remove all children from Indexed map
        m_pChildrenIndexed.clear();
    }

    uint OdePadProbeHandler::GetDisplayMetaAllocSize()
    {
        LOG_FUNC();
        
        return m_displayMetaAllocSize;
    }
    
    void OdePadProbeHandler::SetDisplayMetaAllocSize(uint size)
    {
        LOG_FUNC();
        
        m_displayMetaAllocSize = size;
    }
    
    GstPadProbeReturn OdePadProbeHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }
        GstBuffer* pBuffer = (GstBuffer*)pInfo->data;
        
        NvDsBatchMeta* pBatchMeta = gst_buffer_get_nvds_batch_meta(pBuffer);
        
        // For each frame in the batched meta data
        for (NvDsMetaList* pFrameMetaList = pBatchMeta->frame_meta_list; 
            pFrameMetaList; pFrameMetaList = pFrameMetaList->next)
        {
            // Check for valid frame data
            NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta*) (pFrameMetaList->data);
            if (pFrameMeta != NULL)
            {
                std::vector<NvDsDisplayMeta*> displayMetaData;
                
                for (auto i=0; i<m_displayMetaAllocSize; i++)
                {
                    // Acquire new Display meta for this frame, with each Trigger/Action(s)
                    // adding meta as needed
                    NvDsDisplayMeta* pDisplayMeta = 
                        nvds_acquire_display_meta_from_pool(pBatchMeta);
                    displayMetaData.push_back(pDisplayMeta);
                }
                // Preprocess the frame
                for (const auto &imap: m_pChildrenIndexed)
                {
                    DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                        std::dynamic_pointer_cast<OdeTrigger>(imap.second);
                    pOdeTrigger->PreProcessFrame(pBuffer, displayMetaData, pFrameMeta);
                }

                // For each detected object in the frame.
                for (NvDsMetaList* pMeta = pFrameMeta->obj_meta_list; 
                    pMeta != NULL; pMeta = pMeta->next)
                {
                    // Check for valid object data
                    NvDsObjectMeta* pObjectMeta = (NvDsObjectMeta*) (pMeta->data);
                    if (pObjectMeta != NULL)
                    {
                        // For each ODE Trigger owned by this ODE Manager, check for ODE
                        for (const auto &imap: m_pChildrenIndexed)
                        {
                            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                                std::dynamic_pointer_cast<OdeTrigger>(imap.second);
                            try
                            {
                                pOdeTrigger->CheckForOccurrence(pBuffer, 
                                    displayMetaData, pFrameMeta, pObjectMeta);
                            }
                            catch(...)
                            {
                                LOG_ERROR("Trigger '" << pOdeTrigger->GetName() 
                                    << "' threw exception");
                            }                            
                        }
                    }
                }
                
                // After each detected object is checked for ODE individually, post process 
                // each frame for Absence events, Limit events, etc. (i.e. frame level events).
                for (const auto &imap: m_pChildrenIndexed)
                {
                    DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                        std::dynamic_pointer_cast<OdeTrigger>(imap.second);
                    pOdeTrigger->PostProcessFrame(pBuffer, displayMetaData, pFrameMeta);
                }
                
                for (const auto & ivec: displayMetaData)
                {
                    // Add the updated display data to the frame
                    nvds_add_display_meta_to_frame(pFrameMeta, ivec);
                }
            }
        }
        return GST_PAD_PROBE_OK;
    }

    //----------------------------------------------------------------------------------------------

    CustomPadProbeHandler::CustomPadProbeHandler(const char* name, 
        dsl_pph_custom_client_handler_cb clientHandler, void* clientData)
        : PadProbeHandler(name)
        , m_clientHandler(clientHandler)
        , m_clientData(clientData)
    {
        LOG_FUNC();
        
        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    CustomPadProbeHandler::~CustomPadProbeHandler()
    {
        LOG_FUNC();
    }
    
    GstPadProbeReturn CustomPadProbeHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }

        GstBuffer* pBuffer = (GstBuffer*)pInfo->data;
        try
        {
            return (GstPadProbeReturn)m_clientHandler(pBuffer, m_clientData);
        }
        catch(...)
        {
            LOG_ERROR("CustomPadProbeHandler '" << GetName() << "' threw an exception processing Pad Buffer");
            return GST_PAD_PROBE_REMOVE;
        }
    }
    
    //----------------------------------------------------------------------------------------------

    MeterPadProbeHandler::MeterPadProbeHandler(const char* name, 
        uint interval, dsl_pph_meter_client_handler_cb clientHandler, void* clientData)
        : PadProbeHandler(name)
        , m_interval(interval)
        , m_clientHandler(clientHandler)
        , m_clientData(clientData)
        , m_timerId(0)
    {
        LOG_FUNC();

        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    MeterPadProbeHandler::~MeterPadProbeHandler()
    {
        LOG_FUNC();

        if (m_timerId)
        {
            g_source_remove(m_timerId);
        }
    }
    
    bool MeterPadProbeHandler::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);
        
        if (m_isEnabled == enabled)
        {
            LOG_ERROR("Can't set Meter Enabled to the same value of " 
                << enabled << " for MeterPadProbeHandler '" << GetName() << "' ");
            return false;
        }
        m_isEnabled = enabled;
        
        if (enabled)
        {
            LOG_INFO("Enabling performance measurements for MeterPadProbeHandler '" << GetName() << "'");

            // if have Source Meters, i.e we are currently linked, reset each.
            for (auto const &imap: m_sourceMeters)
            {
                imap.second->SessionReset();
                imap.second->IntervalReset();
            }

            return true;
        }
        LOG_INFO("Disabling performance measurements for MeterPadProbeHandler '" << GetName() << "'");
        
        if (m_timerId and !g_source_remove(m_timerId))
        {
            LOG_ERROR("Interval-timer shutdown failed for MeterPadProbeHandler '" << GetName() << "' ");
            return false;
        }
        m_timerId = 0;
        
        return true;
    }
    
    uint MeterPadProbeHandler::GetInterval()
    {
        LOG_FUNC();
        
        return m_interval;
    }
    
    bool MeterPadProbeHandler::SetInterval(uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        if (IsInUse() and m_isEnabled)
        {
            LOG_ERROR("Unable to set Interval for MeterPadProbeHandler '" << GetName() 
                << "' as it's currently linked and enabled. Disable or stop Pipeline first");
            return false;
        }
        
        m_interval = interval;
        return true;
    }

    GstPadProbeReturn MeterPadProbeHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }

        GstBuffer* pBuffer = (GstBuffer*)pInfo->data;
        NvDsBatchMeta* pBatchMeta = gst_buffer_get_nvds_batch_meta(pBuffer);

        // Don't start the report timer until we get the first buffer
        if (!m_timerId)
        {    
            LOG_INFO("Setting interval timer to " << m_interval*1000);
            m_timerId = g_timeout_add(m_interval*1000, MeterIntervalTimeoutHandler, this);
        }
        try
        {
            for (NvDsMetaList* pFrame = pBatchMeta->frame_meta_list; pFrame; pFrame = pFrame->next)
            {
                NvDsFrameMeta *pFrameMeta = (NvDsFrameMeta*) pFrame->data;
                if (m_sourceMeters.find(pFrameMeta->pad_index) == m_sourceMeters.end())
                {
                    m_sourceMeters[pFrameMeta->pad_index] = DSL_SOURCE_METER_NEW(pFrameMeta->pad_index);
                }

                m_sourceMeters[pFrameMeta->pad_index]->Timestamp();
                // increment the frame counters, calculations will be made based on last timestamp and frame counts.
                m_sourceMeters[pFrameMeta->pad_index]->IncrementFrameCounts();
            }
        }
        catch(...)
        {
            LOG_ERROR("MeterBatchMetaHandler '" 
                << GetName() << "' threw an exception processing Pad Buffer");
            return GST_PAD_PROBE_REMOVE;
        }
        return GST_PAD_PROBE_OK;
    }
    
    int MeterPadProbeHandler::HandleIntervalTimeout()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);
        
        // TODO Handle dewarper serfaces
        
        std::vector<double> sessionAverages;
        std::vector<double> intervalAverages;

        for (auto const &imap: m_sourceMeters)
        {
            sessionAverages.push_back(imap.second->GetSessionFpsAvg());
            intervalAverages.push_back(imap.second->GetIntervalFpsAvg());

            imap.second->IntervalReset();
        }
        
        try
        {
            return m_clientHandler((double*)&sessionAverages[0], (double*)&intervalAverages[0], 
                (uint)m_sourceMeters.size(), m_clientData);
        }
        catch(...)
        {
            LOG_ERROR("MeterPadProbeHandler '" << GetName() 
                << "' threw exception calling client callback... disabling!");
            return false;
        }
    }
    
    //----------------------------------------------------------------------------------------------
    
    static int MeterIntervalTimeoutHandler(void* user_data)
    {
        return static_cast<MeterPadProbeHandler*>(user_data)->
            HandleIntervalTimeout();
    }

    //----------------------------------------------------------------------------------------------

    TimestampPadProbeHandler::TimestampPadProbeHandler(const char* name)
        : PadProbeHandler(name)
        , m_timestamp{0}
    {
        LOG_FUNC();
        
        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    TimestampPadProbeHandler::~TimestampPadProbeHandler()
    {
        LOG_FUNC();
    }
    
    void TimestampPadProbeHandler::GetTime(struct timeval& timestamp)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);
        timestamp = m_timestamp;
    }
    
    void TimestampPadProbeHandler::SetTime(struct timeval& timestamp)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);
        
        m_timestamp = timestamp;
    }
    
    GstPadProbeReturn TimestampPadProbeHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }
        
        gettimeofday(&m_timestamp, NULL);
        return GST_PAD_PROBE_OK;
    }

    //----------------------------------------------------------------------------------------------

    EosConsumerPadProbeEventHandler::EosConsumerPadProbeEventHandler(const char* name)
        : PadProbeHandler(name)
    {
        LOG_FUNC();
        
        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    EosConsumerPadProbeEventHandler::~EosConsumerPadProbeEventHandler()
    {
        LOG_FUNC();
    }
    
    GstPadProbeReturn EosConsumerPadProbeEventHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padHandlerMutex);

        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }
        
        GstEvent *event = (GstEvent*)pInfo->data;
        if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
        {
            LOG_INFO("EOS Consumer -- dropping EOS Event");
            return GST_PAD_PROBE_DROP;
        }
        return GST_PAD_PROBE_OK;
    }
 
    //----------------------------------------------------------------------------------------------

    EosHandlerPadProbeEventHandler::EosHandlerPadProbeEventHandler(const char* name, 
        dsl_pph_custom_client_handler_cb clientHandler, void* clientData)
        : PadProbeHandler(name)
        , m_clientHandler(clientHandler)
        , m_clientData(clientData)
    {
        LOG_FUNC();
        
        m_clientData = clientData;

        // Enable now
        if (!SetEnabled(true))
        {
            throw;
        }
    }

    EosHandlerPadProbeEventHandler::~EosHandlerPadProbeEventHandler()
    {
        LOG_FUNC();
    }
    
    GstPadProbeReturn EosHandlerPadProbeEventHandler::HandlePadData(GstPadProbeInfo* pInfo)
    {
        if (!m_isEnabled)
        {
            return GST_PAD_PROBE_OK;
        }

        GstEvent *pEvent = (GstEvent*)pInfo->data;
        if (GST_EVENT_TYPE(pEvent) == GST_EVENT_EOS)
        {
            try
            {
                return (GstPadProbeReturn)m_clientHandler(pEvent, m_clientData);
            }
            catch(...)
            {
                LOG_ERROR("EosHandlerPadProbeEventHandler '" << GetName() 
                    << "' threw an exception processing Pad Buffer");
                return GST_PAD_PROBE_REMOVE;
            }
        }
        return GST_PAD_PROBE_OK;
    }
 
    //----------------------------------------------------------------------------------------------

    PadProbetr::PadProbetr(const char* name, 
        const char* factoryName, DSL_ELEMENT_PTR parentElement, GstPadProbeType padProbeType)
        : Base(name)
        , m_factoryName(factoryName)
        , m_pParentGstElement(parentElement->GetGstElement())
        , m_padProbeId(0)
        , m_padProbeType(padProbeType)
        , m_pStaticPad(NULL)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_padProbeMutex);
    }

    PadProbetr::~PadProbetr()
    {
        LOG_FUNC();
        
        RemoveAllChildren();
        
        if (m_pStaticPad)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padProbeMutex);
            
            if (m_padProbeId)
            {
                gst_pad_remove_probe(m_pStaticPad, m_padProbeId);
            }
            gst_object_unref(m_pStaticPad);
        }

        g_mutex_clear(&m_padProbeMutex);
    }

    bool PadProbetr::AddPadProbeHandler(DSL_BASE_PTR pPadProbeHandler)
    {
        LOG_FUNC();
        
        if (IsChild(pPadProbeHandler))
        {
            LOG_ERROR("Pad Probe Handler is already a child of PadProbetr '" << m_name << "'");
            return false;
        }
        
        if (!m_padProbeId)
        {
            m_pStaticPad = gst_element_get_static_pad(m_pParentGstElement, m_factoryName.c_str());
            if (!m_pStaticPad)
            {
                LOG_ERROR("Failed to get Static Pad for PadProbetr '" << m_name << "'");
                return false;
            }
     
            // Src Pad Probe notified on Buffer ready
            m_padProbeId = gst_pad_add_probe(m_pStaticPad, m_padProbeType,
                PadProbeCB, this, NULL);
        }
        
        return AddChild(pPadProbeHandler);
    }
    
    bool PadProbetr::RemovePadProbeHandler(DSL_BASE_PTR pPadProbeHandler)
    {
        LOG_FUNC();
        
        if (!IsChild(pPadProbeHandler))
        {
            LOG_ERROR("Pad Probe Handler is not in use by PadProbetr '" << m_name << "'");
            return false;
        }
        
        return RemoveChild(pPadProbeHandler);
    }

    //----------------------------------------------------------------------------------------------

    PadBufferProbetr::PadBufferProbetr(const char* name, 
        const char* factoryName, DSL_ELEMENT_PTR parentElement)
        : PadProbetr(name, factoryName, parentElement, GST_PAD_PROBE_TYPE_BUFFER)
    {
        LOG_FUNC();
    }

    PadBufferProbetr::~PadBufferProbetr()
    {
        LOG_FUNC();
    }

    GstPadProbeReturn PadBufferProbetr::HandlePadProbe(GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_padProbeMutex);
        
        if ((pInfo->type & GST_PAD_PROBE_TYPE_BUFFER))
        {
            if (!(GstBuffer*)pInfo->data)
            {
                LOG_WARN("Unable to get data buffer for PadProbetr '" << m_name << "'");
                return GST_PAD_PROBE_OK;
            }
        
            for (auto const& imap: m_pChildren)
            {
                DSL_PPH_PTR pPadProbeHandler = std::dynamic_pointer_cast<PadProbeHandler>(imap.second);
                try
                {
                    if (pPadProbeHandler->HandlePadData(pInfo) == GST_PAD_PROBE_REMOVE)
                    {
                        LOG_INFO("Removing Pad Probe Handler from PadProbetr '" << m_name << "'");
                        RemovePadProbeHandler(pPadProbeHandler);
                    }
                }
                catch(...)
                {
                    LOG_INFO("Removing Pad Probe Handler for PadProbetr '" << m_name << "'");
                    RemovePadProbeHandler(pPadProbeHandler);
                }
            }
        }
        return GST_PAD_PROBE_OK;
    }
    
    //----------------------------------------------------------------------------------------------

    PadEventDownStreamProbetr::PadEventDownStreamProbetr(const char* name, 
        const char* factoryName, DSL_ELEMENT_PTR parentElement)
        : PadProbetr(name, factoryName, parentElement, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM)
    {
        LOG_FUNC();
    }

    PadEventDownStreamProbetr::~PadEventDownStreamProbetr()
    {
        LOG_FUNC();
    }

    GstPadProbeReturn PadEventDownStreamProbetr::HandlePadProbe(GstPad* pPad, GstPadProbeInfo* pInfo)   
    {
        if (pInfo->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM)
        {
            if (!(GstEvent*)pInfo->data)
            {
                LOG_WARN("Unable to get Event for PadProbetr '" << m_name << "'");
                return GST_PAD_PROBE_OK;
            }
            for (auto const& imap: m_pChildren)
            {
                DSL_PPH_PTR pPadProbeHandler = std::dynamic_pointer_cast<PadProbeHandler>(imap.second);
                try
                {
                    GstPadProbeReturn retval = pPadProbeHandler->HandlePadData(pInfo);
                    if (retval == GST_PAD_PROBE_REMOVE)
                    {
                        LOG_INFO("Removing Pad Probe Handler from PadProbetr '" << m_name << "'");
                        RemovePadProbeHandler(pPadProbeHandler);
                    }
                    else if (retval == GST_PAD_PROBE_DROP)
                    {
                        return retval;
                    }
                }
                catch(...)
                {
                    LOG_INFO("Removing Pad Probe Handler for PadProbetr '" << m_name << "'");
                    RemovePadProbeHandler(pPadProbeHandler);
                }
            }
        }
        return GST_PAD_PROBE_OK;        
    }

    //----------------------------------------------------------------------------------------------
    static GstPadProbeReturn PadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pPadProbetr)
    {
        return static_cast<PadProbetr*>(pPadProbetr)->
            HandlePadProbe(pPad, pInfo);
    }
}