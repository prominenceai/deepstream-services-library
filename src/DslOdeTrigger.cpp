/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"
#include "DslOdeArea.h"
#include "DslOdeHeatMapper.h"
#include "DslServices.h"

namespace DSL
{

    // Initialize static Event Counter
    uint64_t OdeTrigger::s_eventCount = 0;

    OdeTrigger::OdeTrigger(const char* name, const char* source, 
        uint classId, uint limit)
        : OdeBase(name)
        , m_wName(m_name.begin(), m_name.end())
        , m_source(source)
        , m_sourceId(-1)
        , m_inferId(-1)
        , m_classId(classId)
        , m_triggered(0)
        , m_eventLimit(limit)
        , m_frameCount(0)
        , m_frameLimit(0)
        , m_occurrences(0)
        , m_occurrencesAccumulated(0)
        , m_minConfidence(0)
        , m_maxConfidence(0)
        , m_minTrackerConfidence(0)
        , m_maxTrackerConfidence(0)
        , m_minWidth(0)
        , m_minHeight(0)
        , m_maxWidth(0)
        , m_maxHeight(0)
        , m_minFrameCountN(1)
        , m_minFrameCountD(1)
        , m_inferDoneOnly(false)
        , m_resetTimeout(0)
        , m_resetTimerId(0)
        , m_interval(0)
        , m_intervalCounter(0)
        , m_skipFrame(false)
        , m_nextAreaIndex(0)
        , m_nextActionIndex(0)
    {
        LOG_FUNC();

        g_mutex_init(&m_resetTimerMutex);
    }

    OdeTrigger::~OdeTrigger()
    {
        LOG_FUNC();
        
        RemoveAllActions();
        RemoveAllAreas();
        if (m_pAccumulator)
        {
            RemoveAccumulator();
        }
        
        if (m_resetTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_resetTimerMutex);
            g_source_remove(m_resetTimerId);
        }
        g_mutex_clear(&m_resetTimerMutex);
    }

    bool OdeTrigger::AddAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_pOdeActions.find(pChild->GetName()) != m_pOdeActions.end())
        {
            LOG_ERROR("ODE Area '" << pChild->GetName() 
                << "' is already a child of ODE Trigger '" << GetName() << "'");
            return false;
        }
        
        // increment next index, assign to the Action, and update parent releationship.
        pChild->SetIndex(++m_nextActionIndex);
        pChild->AssignParentName(GetName());

        // Add the shared pointer to child to both Maps, by name and index
        m_pOdeActions[pChild->GetName()] = pChild;
        m_pOdeActionsIndexed[m_nextActionIndex] = pChild;
        
        return true;
    }

    bool OdeTrigger::RemoveAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_pOdeActions.find(pChild->GetName()) == m_pOdeActions.end())
        {
            LOG_WARN("'" << pChild->GetName() 
                <<"' is not a child of ODE Trigger '" << GetName() << "'");
            return false;
        }
        
        // Erase the child from both maps
        m_pOdeActions.erase(pChild->GetName());
        m_pOdeActionsIndexed.erase(pChild->GetIndex());
        
        // Clear the parent relationship and index
        pChild->ClearParentName();
        pChild->SetIndex(0);
        return true;
    }
    
    void OdeTrigger::RemoveAllActions()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        for (auto &imap: m_pOdeActions)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() 
                <<"' from Parent '" << GetName() << "'");
            imap.second->ClearParentName();
        }
        m_pOdeActions.clear();
        m_pOdeActionsIndexed.clear();
    }
    
    bool OdeTrigger::AddArea(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_pOdeAreas.find(pChild->GetName()) != m_pOdeAreas.end())
        {
            LOG_ERROR("ODE Area '" << pChild->GetName() 
                << "' is already a child of ODE Trigger'" << GetName() << "'");
            return false;
        }
        // increment next index, assign to the Action, and update parent releationship.
        pChild->SetIndex(++m_nextAreaIndex);
        pChild->AssignParentName(GetName());
        
        // Add the shared pointer to child to both Maps, by name and index
        m_pOdeAreas[pChild->GetName()] = pChild;
        m_pOdeAreasIndexed[m_nextAreaIndex] = pChild;
        
        return true;
    }

    bool OdeTrigger::RemoveArea(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_pOdeAreas.find(pChild->GetName()) == m_pOdeAreas.end())
        {
            LOG_WARN("'" << pChild->GetName() 
                <<"' is not a child of ODE Trigger '" << GetName() << "'");
            return false;
        }
        
        // Erase the child from both maps
        m_pOdeAreas.erase(pChild->GetName());
        m_pOdeAreasIndexed.erase(pChild->GetIndex());

        // Clear the parent relationship and index
        pChild->ClearParentName();
        pChild->SetIndex(0);
        
        return true;
    }
    
    void OdeTrigger::RemoveAllAreas()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        for (auto &imap: m_pOdeAreas)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() 
                <<"' from Parent '" << GetName() << "'");
            imap.second->ClearParentName();
        }
        m_pOdeAreas.clear();
        m_pOdeAreasIndexed.clear();
    }

    bool OdeTrigger::AddAccumulator(DSL_BASE_PTR pAccumulator)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_pAccumulator)
        {
            LOG_ERROR("ODE Trigger '" << GetName() 
                << "' all ready has an Accumulator");
            return false;
        }
        m_pAccumulator = pAccumulator;
        return true;
    }
    
    bool OdeTrigger::RemoveAccumulator()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (!m_pAccumulator)
        {
            LOG_ERROR("ODE Trigger '" << GetName() 
                << "' does not have an Accumulator");
            return false;
        }
        m_pAccumulator = NULL;
        return true;
    }
        
    bool OdeTrigger::AddHeatMapper(DSL_BASE_PTR pHeatMapper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_pHeatMapper)
        {
            LOG_ERROR("ODE Trigger '" << GetName() 
                << "' all ready has a Heat-Mapper");
            return false;
        }
        m_pHeatMapper = pHeatMapper;
        return true;
    }
    
    bool OdeTrigger::RemoveHeatMapper()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (!m_pHeatMapper)
        {
            LOG_ERROR("ODE Trigger '" << GetName() 
                << "' does not have a Heat-Mapper");
            return false;
        }
        m_pHeatMapper = NULL;
        return true;
    }
        
    void OdeTrigger::Reset()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_triggered = 0;
        m_occurrencesAccumulated = 0;
        
        m_frameCount = 0;
        
        // iterate through the map of limit-event-listeners calling each
        for(auto const& imap: m_limitStateChangeListeners)
        {
            try
            {
                imap.first(DSL_ODE_TRIGGER_LIMIT_COUNTS_RESET, 
                    m_eventLimit, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Limit-State-Change-Lister");
            }
        }
    }
    
    void OdeTrigger::IncrementAndCheckTriggerCount()
    {
        LOG_FUNC();
        // internal do not lock m_propertyMutex
        
        m_triggered++;
        
        if (m_triggered >= m_eventLimit)
        {
            // iterate through the map of limit-event-listeners calling each
            for(auto const& imap: m_limitStateChangeListeners)
            {
                try
                {
                    imap.first(DSL_ODE_TRIGGER_LIMIT_EVENT_REACHED, 
                        m_eventLimit, imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("Exception calling Client Limit State-Change Lister");
                }
            }
            if (m_resetTimeout)
            {
                m_resetTimerId = g_timeout_add(1000*m_resetTimeout, 
                    TriggerResetTimeoutHandler, this);            
            }
        }
    }

    static int TriggerResetTimeoutHandler(gpointer pTrigger)
    {
        return static_cast<OdeTrigger*>(pTrigger)->
            HandleResetTimeout();
    }

    int OdeTrigger::HandleResetTimeout()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_resetTimerMutex);
        
        m_resetTimerId = 0;
        Reset();
        
        // One shot - return false.
        return false;
    }
    
    uint OdeTrigger::GetResetTimeout()
    {
        LOG_FUNC();
        
        return m_resetTimeout;
    }
        
    void OdeTrigger::SetResetTimeout(uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_resetTimerMutex);
        
        // If the timer is currently running and the new 
        // timeout value is zero (disabled), then kill the timer.
        if (m_resetTimerId and !timeout)
        {
            g_source_remove(m_resetTimerId);
            m_resetTimerId = 0;
        }
        
        // Else, if the Timer is currently running and the new
        // timeout value is non-zero, stop and restart the timer.
        else if (m_resetTimerId and timeout)
        {
            g_source_remove(m_resetTimerId);
            m_resetTimerId = g_timeout_add(1000*m_resetTimeout, 
                TriggerResetTimeoutHandler, this);            
        }
        
        // Else, if the Trigger has reached its limit and the 
        // client is setting a Timeout value, start the timer.
        else if (m_eventLimit and (m_triggered >= m_eventLimit) and timeout)
        {
            m_resetTimerId = g_timeout_add(1000*m_resetTimeout, 
                TriggerResetTimeoutHandler, this);            
        } 
        // Else, if the Trigger has reached its frame limit and the 
        // client is setting a Timeout value, start the timer.
        else if (m_frameLimit and (m_frameCount >= m_frameLimit) and timeout)
        {
            m_resetTimerId = g_timeout_add(1000*m_resetTimeout, 
                TriggerResetTimeoutHandler, this);            
        } 
        
        m_resetTimeout = timeout;
    }
    
    bool OdeTrigger::IsResetTimerRunning()
    {
        LOG_FUNC();

        return m_resetTimerId;
    }
    
    bool OdeTrigger::AddLimitStateChangeListener(
        dsl_ode_trigger_limit_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_limitStateChangeListeners.find(listener) != 
            m_limitStateChangeListeners.end())
        {   
            LOG_ERROR("Limit state change listener is not unique");
            return false;
        }
        m_limitStateChangeListeners[listener] = clientData;

        return true;
    }
    
    bool OdeTrigger::RemoveLimitStateChangeListener(
        dsl_ode_trigger_limit_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (m_limitStateChangeListeners.find(listener) == 
            m_limitStateChangeListeners.end())
        {   
            LOG_ERROR("Limit state change listener was not found");
            return false;
        }
        m_limitStateChangeListeners.erase(listener);

        return true;
    }        
        
    uint OdeTrigger::GetClassId()
    {
        LOG_FUNC();
        
        return m_classId;
    }
    
    void OdeTrigger::SetClassId(uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_classId = classId;
    }

    uint OdeTrigger::GetEventLimit()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_eventLimit;
    }
    
    void OdeTrigger::SetEventLimit(uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_eventLimit = limit;
        
        // iterate through the map of limit-event-listeners calling each
        for(auto const& imap: m_limitStateChangeListeners)
        {
            try
            {
                imap.first(DSL_ODE_TRIGGER_LIMIT_EVENT_CHANGED, 
                    m_eventLimit, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Limit-State-Change-Lister");
            }
        }
    }

    uint OdeTrigger::GetFrameLimit()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_frameLimit;
    }
    
    void OdeTrigger::SetFrameLimit(uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_frameLimit = limit;
        
        // iterate through the map of limit-event-listeners calling each
        for(auto const& imap: m_limitStateChangeListeners)
        {
            try
            {
                imap.first(DSL_ODE_TRIGGER_LIMIT_FRAME_CHANGED, 
                    m_frameLimit, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Limit-State-Change-Lister");
            }
        }
    }

    const char* OdeTrigger::GetSource()
    {
        LOG_FUNC();
        
        if (m_source.size())
        {
            return m_source.c_str();
        }
        return NULL;
    }
    
    void OdeTrigger::SetSource(const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_source.assign(source);
    }

    void OdeTrigger::_setSourceId(int id)
    {
        LOG_FUNC();
        
        m_sourceId = id;
    }
    
    const char* OdeTrigger::GetInfer()
    {
        LOG_FUNC();
        
        if (m_infer.size())
        {
            return m_infer.c_str();
        }
        return NULL;
    }
    
    void OdeTrigger::SetInfer(const char* infer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_infer.assign(infer);
    }

    void OdeTrigger::_setInferId(int id)
    {
        LOG_FUNC();
        
        m_inferId = id;
    }
    
    float OdeTrigger::GetMinConfidence()
    {
        LOG_FUNC();
        
        return m_minConfidence;
    }
    
    void OdeTrigger::SetMinConfidence(float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minConfidence = minConfidence;
    }
    
    float OdeTrigger::GetMaxConfidence()
    {
        LOG_FUNC();
        
        return m_maxConfidence;
    }
    
    void OdeTrigger::SetMaxConfidence(float maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_maxConfidence = maxConfidence;
    }
    
    float OdeTrigger::GetMinTrackerConfidence()
    {
        LOG_FUNC();
        
        return m_minTrackerConfidence;
    }
    
    void OdeTrigger::SetMinTrackerConfidence(float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minTrackerConfidence = minConfidence;
    }
    
    float OdeTrigger::GetMaxTrackerConfidence()
    {
        LOG_FUNC();
        
        return m_maxTrackerConfidence;
    }
    
    void OdeTrigger::SetMaxTrackerConfidence(float maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_maxTrackerConfidence = maxConfidence;
    }
    
    void OdeTrigger::GetMinDimensions(float* minWidth, float* minHeight)
    {
        LOG_FUNC();
        
        *minWidth = m_minWidth;
        *minHeight = m_minHeight;
    }

    void OdeTrigger::SetMinDimensions(float minWidth, float minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minWidth = minWidth;
        m_minHeight = minHeight;
    }
    
    void OdeTrigger::GetMaxDimensions(float* maxWidth, float* maxHeight)
    {
        LOG_FUNC();
        
        *maxWidth = m_maxWidth;
        *maxHeight = m_maxHeight;
    }

    void OdeTrigger::SetMaxDimensions(float maxWidth, float maxHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_maxWidth = maxWidth;
        m_maxHeight = maxHeight;
    }
    
    bool OdeTrigger::GetInferDoneOnlySetting()
    {
        LOG_FUNC();
        
        return m_inferDoneOnly;
    }
    
    void OdeTrigger::SetInferDoneOnlySetting(bool inferDoneOnly)
    {
        LOG_FUNC();
        
        m_inferDoneOnly = inferDoneOnly;
    }
    
    void OdeTrigger::GetMinFrameCount(uint* minFrameCountN, uint* minFrameCountD)
    {
        LOG_FUNC();
        
        *minFrameCountN = m_minFrameCountN;
        *minFrameCountD = m_minFrameCountD;
    }

    void OdeTrigger::SetMinFrameCount(uint minFrameCountN, uint minFrameCountD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minFrameCountN = minFrameCountN;
        m_minFrameCountD = minFrameCountD;
    }

    uint OdeTrigger::GetInterval()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_interval;
    }
    
    void OdeTrigger::SetInterval(uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_interval = interval;
        m_intervalCounter = 0;
    }
    
    bool OdeTrigger::CheckForSourceId(int sourceId)
    {
        LOG_FUNC();

        // Filter on Source id if set
        if (m_source.size())
        {
            // a "one-time-get" of the source Id from the source name
            if (m_sourceId == -1)
            {
                
                Services::GetServices()->SourceIdGet(m_source.c_str(), &m_sourceId);
            }
            if (m_sourceId != sourceId)
            {
                return false;
            }
        }
        return true;
    }

    bool OdeTrigger::CheckForInferId(int inferId)
    {
        LOG_FUNC();

        // Filter on Source id if set
        if (m_infer.size())
        {
            // a "one-time-get" of the inference component Id from the name
            if (m_inferId == -1)
            {
                Services::GetServices()->InferIdGet(m_infer.c_str(), &m_inferId);
            }
            if (m_inferId != inferId)
            {
                return false;
            }
        }
        return true;
    }

    void OdeTrigger::PreProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,
        NvDsFrameMeta* pFrameMeta)
    {
        // Reset the occurrences from the last frame, even if disabled  
        m_occurrences = 0;

        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id))
        {
            return;
        }

        // Call on each of the Trigger's Areas to (optionally) display their Rectangle
        for (const auto &imap: m_pOdeAreasIndexed)
        {
            DSL_ODE_AREA_PTR pOdeArea = 
                std::dynamic_pointer_cast<OdeArea>(imap.second);
            
            pOdeArea->AddMeta(displayMetaData, pFrameMeta);
        }
        if (m_interval)
        {
            m_intervalCounter = (m_intervalCounter + 1) % m_interval; 
            if (m_intervalCounter != 0)
            {
                m_skipFrame = true;
                return;
            }
        }
        m_skipFrame = false;
    }

    uint OdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,
        NvDsFrameMeta* pFrameMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        // Filter on skip-frame interval
        if (!m_enabled or m_skipFrame)
        {
            return 0;
        }

        // Don't start incrementing the frame-count until after the
        // first ODE occurrence. 
        if (m_triggered)
        {
            m_frameCount++;
        }

        // Check to see if frame limit is enabled and exceeded
        if (m_frameLimit and (m_frameCount > m_frameLimit))
        {
            return 0;
        }

        // Else, if frame limit is enabled and reached in this frame
        if (m_frameLimit and (m_frameCount == m_frameLimit))
        {
            // iterate through the map of limit-event-listeners calling each
            for(auto const& imap: m_limitStateChangeListeners)
            {
                try
                {
                    imap.first(DSL_ODE_TRIGGER_LIMIT_FRAME_REACHED, 
                        m_frameLimit, imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("Exception calling Client Limit State-Change Lister");
                }
            }
            if (m_resetTimeout)
            {
                m_resetTimerId = g_timeout_add(1000*m_resetTimeout, 
                    TriggerResetTimeoutHandler, this);            
            }
        }

        // If the client has added an accumulator, 
        if (m_pAccumulator)
        {
            m_occurrencesAccumulated += m_occurrences;
            
            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_ACTIVE_INDEX] = 
                DSL_FRAME_INFO_OCCURRENCES;
            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES] = 
                m_occurrencesAccumulated;
                
            DSL_ODE_ACCUMULATOR_PTR pOdeAccumulator = 
                std::dynamic_pointer_cast<OdeAccumulator>(m_pAccumulator);
                
            pOdeAccumulator->HandleOccurrences(shared_from_this(),
                pBuffer, displayMetaData, pFrameMeta);
        }
        
        // If the client has added a heat-mapper
        if (m_pHeatMapper)
        {
            std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->AddDisplayMeta(
                displayMetaData);
        }
        
        return m_occurrences;
    }        
    
    bool OdeTrigger::CheckForMinCriteria(NvDsFrameMeta* pFrameMeta, 
        NvDsObjectMeta* pObjectMeta)
    {
        // Filter on enable and skip-frame interval
        if (!m_enabled or m_skipFrame)
        {
            return false;
        }
        
        // Ensure that the event limit has not been exceeded
        if (m_eventLimit and m_triggered >= m_eventLimit) 
        {
            return false;
        }
        // Ensure that the frame limit has not been exceeded
        if (m_frameLimit and m_frameCount >= m_frameLimit) 
        {
            return false;
        }
        // Filter on unique source-id and unique-inference-component-id
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForInferId(pObjectMeta->unique_component_id))
        {
            return false;
        }
        // Filter on Class id if set
        if ((m_classId != DSL_ODE_ANY_CLASS) and 
            (m_classId != pObjectMeta->class_id))
        {
            return false;
        }
        // Ensure that the minimum Inference confidence has been reached
        if (pObjectMeta->confidence > 0 and 
            pObjectMeta->confidence < m_minConfidence)
        {
            return false;
        }
        // Ensure that the maximum Inference confidence has been reached
        if (pObjectMeta->confidence > 0 and m_maxConfidence and
            pObjectMeta->confidence > m_maxConfidence)
        {
            return false;
        }
        // Ensure that the minimum Tracker confidence has been reached
        if (pObjectMeta->tracker_confidence > 0 and 
            pObjectMeta->tracker_confidence < m_minTrackerConfidence)
        {
            return false;
        }
        // Ensure that the maximum Tracker confidence has been reached
        if (pObjectMeta->tracker_confidence > 0 and m_maxTrackerConfidence and
            pObjectMeta->tracker_confidence > m_maxTrackerConfidence)
        {
            return false;
        }
        // If defined, check for minimum dimensions
        if ((m_minWidth > 0 and pObjectMeta->rect_params.width < m_minWidth) or
            (m_minHeight > 0 and pObjectMeta->rect_params.height < m_minHeight))
        {
            return false;
        }
        // If defined, check for maximum dimensions
        if ((m_maxWidth > 0 and pObjectMeta->rect_params.width > m_maxWidth) or
            (m_maxHeight > 0 and pObjectMeta->rect_params.height > m_maxHeight))
        {
            return false;
        }
        // If define, check if Inference was done on the frame or not
        if (m_inferDoneOnly and !pFrameMeta->bInferDone)
        {
            return false;
        }
        return true;
    }

    bool OdeTrigger::CheckForInside(NvDsObjectMeta* pObjectMeta)
    {
        // If areas are defined, check condition

        if (m_pOdeAreasIndexed.size())
        {
            for (const auto &imap: m_pOdeAreasIndexed)
            {
                DSL_ODE_AREA_PTR pOdeArea = 
                    std::dynamic_pointer_cast<OdeArea>(imap.second);
                if (pOdeArea->IsBboxInside(pObjectMeta->rect_params))
                {
                    return !pOdeArea->IsType(typeid(OdeExclusionArea));
                }
            }
            return false;
        }
        return true;
    }
    
    // *****************************************************************************
    AlwaysOdeTrigger::AlwaysOdeTrigger(const char* name, 
        const char* source, uint when)
        : OdeTrigger(name, source, DSL_ODE_ANY_CLASS, 0)
        , m_when(when)
    {
        LOG_FUNC();
    }

    AlwaysOdeTrigger::~AlwaysOdeTrigger()
    {
        LOG_FUNC();
    }
    
    void AlwaysOdeTrigger::PreProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,
        NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            m_when != DSL_ODE_PRE_OCCURRENCE_CHECK)
        {
            return;
        }
        if (m_interval)
        {
            m_intervalCounter = (m_intervalCounter + 1) % m_interval; 
            if (m_intervalCounter != 0)
            {
                return;
            }
        }
        for (const auto &imap: m_pOdeActionsIndexed)
        {
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, displayMetaData, pFrameMeta, NULL);
        }
    }

    uint AlwaysOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,
        NvDsFrameMeta* pFrameMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            m_when != DSL_ODE_POST_OCCURRENCE_CHECK)
        {
            return 0;
        }
        if (m_interval)
        {
            m_intervalCounter = (m_intervalCounter + 1) % m_interval; 
            if (m_intervalCounter != 0)
            {
                return 0;
            }
        }
        for (const auto &imap: m_pOdeActionsIndexed)
        {
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, displayMetaData, pFrameMeta, NULL);
        }
        return 1;
    }

    // *****************************************************************************

    OccurrenceOdeTrigger::OccurrenceOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    OccurrenceOdeTrigger::~OccurrenceOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool OccurrenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or 
            !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        IncrementAndCheckTriggerCount();
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        // set the primary metric as the current occurrence for this frame
        pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = m_occurrences;


        if (m_pHeatMapper)
        {
            std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                pFrameMeta, pObjectMeta);
        }

        for (const auto &imap: m_pOdeActionsIndexed)
        {
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(imap.second);
            try
            {
                pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, 
                    displayMetaData, pFrameMeta, pObjectMeta);
            }
            catch(...)
            {
                LOG_ERROR("Trigger '" << GetName() << "' => Action '" 
                    << pOdeAction->GetName() << "' threw exception");
            }
        }
        return true;
    }

    // *****************************************************************************
    
    AbsenceOdeTrigger::AbsenceOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    AbsenceOdeTrigger::~AbsenceOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool AbsenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        // Important **** we need to check for Criteria even if the Absence Trigger is disabled. 
        // This is case another Trigger enables This trigger, and it checks for the number of 
        // occurrences in the PostProcessFrame() . If the m_occurrences is not updated the Trigger 
        // will report Absence incorrectly
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;

        return true;
    }
    
    uint AbsenceOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or (m_eventLimit and m_triggered >= m_eventLimit) 
                or m_occurrences) 
            {
                return 0;
            }        
            
            // since occurrences = 0, ODE occurrence for the Absence Trigger = 1
            m_occurrences = 1;
            
            // event has been triggered 
            IncrementAndCheckTriggerCount();

            // update the total event count static variable
            s_eventCount++;

            for (const auto &imap: m_pOdeActionsIndexed)
            {
                DSL_ODE_ACTION_PTR pOdeAction = 
                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                pOdeAction->HandleOccurrence(shared_from_this(), 
                    pBuffer, displayMetaData, pFrameMeta, NULL);
            }
        }
        // mutext unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer, 
            displayMetaData, pFrameMeta);
    }

    // *****************************************************************************

    InstanceOdeTrigger::InstanceOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    InstanceOdeTrigger::~InstanceOdeTrigger()
    {
        LOG_FUNC();
    }

    void InstanceOdeTrigger::Reset()
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_instances.clear();
        }
        // call the base class to complete the Reset
        OdeTrigger::Reset();
    }
    
    bool InstanceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or 
            !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }

        std::string sourceAndClassId = std::to_string(pFrameMeta->source_id) + "_" 
            + std::to_string(pObjectMeta->class_id);
            
        // If this is the first time seeing an object of "class_id" for "source_id".
        if (m_instances.find(sourceAndClassId) == m_instances.end())
        {
            // Initialize the frame number for the new source
            m_instances[sourceAndClassId] = 0;
        }
        if (m_instances[sourceAndClassId] < pObjectMeta->object_id)
        {
            // Update the running instance
            m_instances[sourceAndClassId] = pObjectMeta->object_id;
            
            IncrementAndCheckTriggerCount();
            m_occurrences++;
            
            // update the total event count static variable
            s_eventCount++;

            if (m_pHeatMapper)
            {
                std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                    pFrameMeta, pObjectMeta);
            }

            // set the primary metric to the new instance occurrence for this frame
            pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = m_occurrences;

            for (const auto &imap: m_pOdeActionsIndexed)
            {
                DSL_ODE_ACTION_PTR pOdeAction = 
                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                try
                {
                    pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, 
                        displayMetaData, pFrameMeta, pObjectMeta);
                }
                catch(...)
                {
                    LOG_ERROR("Trigger '" << GetName() << "' => Action '" 
                        << pOdeAction->GetName() << "' threw exception");
                }
            }
            return true;
        }
        return false;
    }


    // *****************************************************************************
    
    SummationOdeTrigger::SummationOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    SummationOdeTrigger::~SummationOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool SummationOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or 
            !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;

        return true;
    }

    uint SummationOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame or (m_eventLimit and m_triggered >= m_eventLimit))
            {
                return 0;
            }
            // event has been triggered
            IncrementAndCheckTriggerCount();

             // update the total event count static variable
            s_eventCount++;

            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_ACTIVE_INDEX] = 
                DSL_FRAME_INFO_OCCURRENCES;
            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES] = m_occurrences;
            for (const auto &imap: m_pOdeActionsIndexed)
            {
                DSL_ODE_ACTION_PTR pOdeAction = 
                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                pOdeAction->HandleOccurrence(shared_from_this(), 
                    pBuffer, displayMetaData, pFrameMeta, NULL);
            }
        }
        // mutex unlocked safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
   }

    // *****************************************************************************

    CustomOdeTrigger::CustomOdeTrigger(const char* name, const char* source, 
        uint classId, uint limit, dsl_ode_check_for_occurrence_cb clientChecker, 
        dsl_ode_post_process_frame_cb clientPostProcessor, void* clientData)
        : OdeTrigger(name, source, classId, limit)
        , m_clientChecker(clientChecker)
        , m_clientPostProcessor(clientPostProcessor)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    CustomOdeTrigger::~CustomOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool CustomOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        // conditional execution
        if (!m_enabled or 
            !m_clientChecker or 
            !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        try
        {
            if (!m_clientChecker(pBuffer, pFrameMeta, pObjectMeta, m_clientData))
            {
                return false;
            }
        }
        catch(...)
        {
            LOG_ERROR("Custon ODE Trigger '" << GetName() 
                << "' threw exception calling client callback");
            return false;
        }

        IncrementAndCheckTriggerCount();
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        if (m_pHeatMapper)
        {
            std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                pFrameMeta, pObjectMeta);
        }

        for (const auto &imap: m_pOdeActionsIndexed)
        {
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, displayMetaData, pFrameMeta, pObjectMeta);
        }
        return true;
    }
    
    uint CustomOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            // conditional execution
            if (!m_enabled or m_clientPostProcessor == NULL)
            {
                return false;
            }
            try
            {
                if (!m_clientPostProcessor(pBuffer, pFrameMeta, m_clientData))
                {
                    return 0;
                }
            }
            catch(...)
            {
                LOG_ERROR("Custon ODE Trigger '" << GetName() 
                    << "' threw exception calling client callback");
                return false;
            }

            // event has been triggered
            IncrementAndCheckTriggerCount();

             // update the total event count static variable
            s_eventCount++;

            for (const auto &imap: m_pOdeActionsIndexed)
            {
                DSL_ODE_ACTION_PTR pOdeAction = 
                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                pOdeAction->HandleOccurrence(shared_from_this(), 
                    pBuffer, displayMetaData, pFrameMeta, NULL);
            }
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }

    // *****************************************************************************
    
    CountOdeTrigger::CountOdeTrigger(const char* name, const char* source,
        uint classId, uint limit, uint minimum, uint maximum)
        : OdeTrigger(name, source, classId, limit)
        , m_minimum(minimum)
        , m_maximum(maximum)
    {
        LOG_FUNC();
    }

    CountOdeTrigger::~CountOdeTrigger()
    {
        LOG_FUNC();
    }

    void CountOdeTrigger::GetRange(uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        
        *minimum = m_minimum;
        *maximum = m_maximum;
    }

    void CountOdeTrigger::SetRange(uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minimum = minimum;
        m_maximum = maximum;
    }
    
    bool CountOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        if (m_pHeatMapper)
        {
            std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                pFrameMeta, pObjectMeta);
        }
        return true;
    }

    uint CountOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame or (m_eventLimit and m_triggered >= m_eventLimit) or
                (m_occurrences < m_minimum) or (m_occurrences > m_maximum))
            {
                return 0;
            }
            // event has been triggered
            IncrementAndCheckTriggerCount();

             // update the total event count static variable
            s_eventCount++;

            for (const auto &imap: m_pOdeActionsIndexed)
            {
                DSL_ODE_ACTION_PTR pOdeAction = 
                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                pOdeAction->HandleOccurrence(shared_from_this(), 
                    pBuffer, displayMetaData, pFrameMeta, NULL);
            }
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
   }

    // *****************************************************************************
    
    SmallestOdeTrigger::SmallestOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    SmallestOdeTrigger::~SmallestOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool SmallestOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrenceMetaList.push_back(pObjectMeta);
        
        return true;
    }

    uint SmallestOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrences = 0;
            
            // need at least one object for a Minimum event
            if (m_enabled and m_occurrenceMetaList.size())
            {
                // One occurrence to return and increment the accumulative Trigger count
                m_occurrences = 1;
                IncrementAndCheckTriggerCount();
                // update the total event count static variable
                s_eventCount++;

                uint smallestArea = UINT32_MAX;
                NvDsObjectMeta* pSmallestObject(NULL);
                
                // iterate through the list of object occurrences that passed all min criteria
                for (const auto &ivec: m_occurrenceMetaList) 
                {
                    uint rectArea = ivec->rect_params.width * ivec->rect_params.height;
                    if (rectArea < smallestArea) 
                    { 
                        smallestArea = rectArea;
                        pSmallestObject = ivec;    
                    }
                }
                // conditionally add the 
                if (m_pHeatMapper)
                {
                    std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                        pFrameMeta, pSmallestObject);
                }
                // set the primary metric as the smallest bounding box by area
                pSmallestObject->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                    = smallestArea;
                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, pSmallestObject);
                }
            }   

            // reset for next frame
            m_occurrenceMetaList.clear();
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
   }

    // *****************************************************************************
    
    LargestOdeTrigger::LargestOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    LargestOdeTrigger::~LargestOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool LargestOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrenceMetaList.push_back(pObjectMeta);
        
        return true;
    }

    uint LargestOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrences = 0;
            
            // need at least one object for a Minimum event
            if (m_enabled and m_occurrenceMetaList.size())
            {
                // Once occurrence to return and increment the accumulative Trigger count
                m_occurrences = 1;
                IncrementAndCheckTriggerCount();
                // update the total event count static variable
                s_eventCount++;

                uint largestArea = 0;
                NvDsObjectMeta* pLargestObject(NULL);
                
                // iterate through the list of object occurrences that passed all min criteria
                for (const auto &ivec: m_occurrenceMetaList) 
                {
                    uint rectArea = ivec->rect_params.width * ivec->rect_params.height;
                    if (rectArea > largestArea) 
                    { 
                        largestArea = rectArea;
                        pLargestObject = ivec;    
                    }
                }

                // If the client has added a heat mapper, call to add-occurrence
                if (m_pHeatMapper)
                {
                    std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                        pFrameMeta, pLargestObject);
                }
                
                // set the primary metric as the larget area
                pLargestObject->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                    = largestArea;
                
                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, pLargestObject);
                }
            }   

            // reset for next frame
            m_occurrenceMetaList.clear();
        }
        // mutex unlocked  - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }

    // *****************************************************************************
    
    NewLowOdeTrigger::NewLowOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit, uint preset)
        : OdeTrigger(name, source, classId, limit)
        , m_preset(preset)
        , m_currentLow(preset)
        
    {
        LOG_FUNC();
    }

    NewLowOdeTrigger::~NewLowOdeTrigger()
    {
        LOG_FUNC();
    }

    void NewLowOdeTrigger::Reset()
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_currentLow = m_preset;
        }        
        // call the base class to complete the Reset
        OdeTrigger::Reset();
    }
    
    bool NewLowOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or 
            !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint NewLowOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame)
            {
                return 0;
            }
            
            if (m_occurrences < m_currentLow)
            {
                // new low
                m_currentLow = m_occurrences;
                
                // event has been triggered
                IncrementAndCheckTriggerCount();

                 // update the total event count static variable
                s_eventCount++;

                // Add the New High occurrences to the frame info
                pFrameMeta->misc_frame_info[DSL_FRAME_INFO_ACTIVE_INDEX] = 
                    DSL_FRAME_INFO_OCCURRENCES;
                pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES] = 
                    m_occurrences;

                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, NULL);
                }
                // new high m_occurrences means ODE occurrence = 1
                m_occurrences = 1;
            }
            else
            {
                m_occurrences = 0;
            }
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
   }

    // *****************************************************************************
    
    NewHighOdeTrigger::NewHighOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit, uint preset)
        : OdeTrigger(name, source, classId, limit)
        , m_preset(preset)
        , m_currentHigh(preset)
        
    {
        LOG_FUNC();
    }

    NewHighOdeTrigger::~NewHighOdeTrigger()
    {
        LOG_FUNC();
    }

    void NewHighOdeTrigger::Reset()
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_currentHigh = m_preset;
            m_occurrencesNewHighAccumulated = 0;
        }        
        // call the base class to complete the Reset
        OdeTrigger::Reset();
    }
    
    bool NewHighOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or 
            !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint NewHighOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame)
            {
                return 0;
            }
            
            if (m_occurrences > m_currentHigh)
            {
                // new high
                m_currentHigh = m_occurrences;
                
                // event has been triggered
                IncrementAndCheckTriggerCount();

                 // update the total event count static variable
                s_eventCount++;

                // Add the New High occurrences to the frame info
                pFrameMeta->misc_frame_info[DSL_FRAME_INFO_ACTIVE_INDEX] = 
                    DSL_FRAME_INFO_OCCURRENCES;
                pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES] = 
                    m_occurrences;

                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, NULL);
                }
                // new high m_occurrences means ODE occurrence = 1
                m_occurrences = 1;
            }
            else
            {
                m_occurrences = 0;
            }
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
   }

    // *****************************************************************************
    // Tracking Triggers
    // *****************************************************************************
    
    TrackingOdeTrigger::TrackingOdeTrigger(const char* name, const char* source, 
        uint classId, uint limit, uint maxTracePoints)
        : OdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
        
        m_pTrackedObjectsPerSource = std::shared_ptr<TrackedObjects>(
            new TrackedObjects(maxTracePoints));
    }

    TrackingOdeTrigger::~TrackingOdeTrigger()
    {
        LOG_FUNC();
    }

    void TrackingOdeTrigger::Reset()
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_pTrackedObjectsPerSource->Clear();
        }        
        // call the base class to complete the Reset
        OdeTrigger::Reset();
    }
   
    // *****************************************************************************
    
    CrossOdeTrigger::CrossOdeTrigger(const char* name, const char* source, 
        uint classId, uint limit, uint minFrameCount, uint maxTracePoints, 
        uint testMethod, DSL_RGBA_COLOR_PTR pColor)
        : TrackingOdeTrigger(name, source, classId, limit, maxTracePoints)
        , m_maxTracePoints(maxTracePoints)
        , m_occurrencesIn(0)
        , m_occurrencesOut(0)
        , m_occurrencesInAccumulated(0)
        , m_occurrencesOutAccumulated(0)
        , m_minFrameCount(minFrameCount)
        , m_traceEnabled(false)
        , m_testMethod(testMethod)
        , m_pTraceColor(pColor)
        , m_traceLineWidth(0)
    {
        LOG_FUNC();
    }

    CrossOdeTrigger::~CrossOdeTrigger()
    {
        LOG_FUNC();
    }

    bool CrossOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_pOdeAreasIndexed.size())
        {
            LOG_ERROR("At least one OdeArea is required for CrossOdeTrigger '" 
                << GetName() << "'");
            return false;
        }

        // Check for minimum criteria - but not for within an Area. 
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }

        // if this is the first occurrence of this object for this source
        if (!m_pTrackedObjectsPerSource->IsTracked(pFrameMeta->source_id,
            pObjectMeta->object_id))
        {
            // Create a new Tracked object and return without occurence
            m_pTrackedObjectsPerSource->Track(pFrameMeta, 
                pObjectMeta, m_pTraceColor);

            // Update the color for the next tracked object to be created.
            m_pTraceColor->SetNext();
            
            return false;
        }

        // Else, get the tracked object and update with current frame meta
        std::shared_ptr<TrackedObject> pTrackedObject = 
            m_pTrackedObjectsPerSource->GetObject(pFrameMeta->source_id,
                pObjectMeta->object_id);
                
        pTrackedObject->Update(pFrameMeta->frame_num, 
            (NvBbox_Coords*)&pObjectMeta->rect_params);
            
        // Iterate through the map of 1 or more Areas to test for line cross
        for (const auto &imap: m_pOdeAreasIndexed)
        {
            DSL_ODE_AREA_PTR pOdeArea = 
                std::dynamic_pointer_cast<OdeArea>(imap.second);
                
            uint testPoint = pOdeArea->GetBboxTestPoint();
                
            // get the first and last test-points to see if minimum requirments
            // have been met to keep testing/tracking the current object.
    
            dsl_coordinate firstCoordinate = 
                pTrackedObject->GetFirstCoordinate(testPoint);
            dsl_coordinate lastCoordinate = 
                pTrackedObject->GetLastCoordinate(testPoint);
            
            // purge tracked objects that start on the line, as well as
            // objects on the line with less than the minimim frame count
            if (pOdeArea->IsPointOnLine(firstCoordinate) or
                (pOdeArea->IsPointOnLine(lastCoordinate) and
                    pTrackedObject->preEventFrameCount < m_minFrameCount))
            {
                LOG_DEBUG("Online without sufficient pre-count " 
                    <<  pTrackedObject->preEventFrameCount << " - purging");
                    
                m_pTrackedObjectsPerSource->DeleteObject(pFrameMeta->source_id,
                    pObjectMeta->object_id);
                return false;
            }
            
            // Get the trace vector for the testpoint defined for this Area
            DSL_RGBA_MULTI_LINE_PTR pTrace = 
                pTrackedObject->GetTrace(testPoint, m_testMethod, m_traceLineWidth);

            // If the client has enabled object tracing
            if (m_traceEnabled)
            {
                // If the object has a previous trace from a line cross event.
                if (pTrackedObject->HasPreviousTrace())
                {
                    DSL_RGBA_MULTI_LINE_PTR pPreviousTrace = 
                        pTrackedObject->GetPreviousTrace(testPoint, m_testMethod, 
                            m_traceLineWidth);

                    // Add the multi-line metadata to the Frame's display-meta for 
                    // the previous trace
                    pPreviousTrace->AddMeta(displayMetaData, pFrameMeta);
                }
                // Add the multi-line metadata to the Frame's display-meta for
                // the current trace.
                pTrace->AddMeta(displayMetaData, pFrameMeta);
                pObjectMeta->rect_params.border_color = pTrace->color;
                pObjectMeta->rect_params.border_width = pTrace->line_width;
            }
            
            uint direction;

            // Check of the trace has crossed the area
            if (pOdeArea->DoesTraceCrossLine(pTrace->coordinates, 
                pTrace->num_coordinates, direction))
            {
                // If we've crosed before reaching the minimum frame count
                if (pTrackedObject->preEventFrameCount < m_minFrameCount)
                {
                    LOG_DEBUG("Crossed line without sufficient pre-count - purging");
                    
                    // delete the object - will be retracked in the next frame.
                    m_pTrackedObjectsPerSource->DeleteObject(pFrameMeta->source_id,
                        pObjectMeta->object_id);
                    return false;
                }
                if (++pTrackedObject->onEventFrameCount < m_minFrameCount)
                {
                    return false;
                }
                
                // event has been triggered
                IncrementAndCheckTriggerCount();
                m_occurrences++;
                
                if (direction == DSL_AREA_CROSS_DIRECTION_IN)
                {
                    m_occurrencesIn++;
                }
                else
                {
                    m_occurrencesOut++;
                }

                // update the total event count static variable
                s_eventCount++;

                // If the client has added a heat mapper, call to add the occurrence data
                if (m_pHeatMapper)
                {
                    std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                        pFrameMeta, pObjectMeta);
                }

                // add the persistence value to the array of misc_obj_info
                // at both the Primary and Persistence specific indecies.
                pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_DIRECTION] = 
                pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = 
                    direction;

                pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] = 
                    (uint64_t)(pTrackedObject->GetDurationMs());
                    
                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, pObjectMeta);
                }

                // Call on the tracked object to handle the occurrence as well.
                pTrackedObject->HandleOccurrence();
                    
                return true;
            }
        }
        // No trigger, so update the per-event counter
        pTrackedObject->preEventFrameCount++;
        return false;
    }

    uint CrossOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        // Filter on skip-frame interval
        if (!m_enabled or m_skipFrame)
        {
            return 0;
        }

        // If the client has added a heat-mapper, need to AddDisplayMeta here as
        // the base/super class PostProcessFrame is not called .
        if (m_pHeatMapper)
        {
            std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->AddDisplayMeta(
                displayMetaData);
        }

        // If the client has added an accumulator, 
        if (m_pAccumulator)
        {
            m_occurrencesInAccumulated += m_occurrencesIn;
            m_occurrencesOutAccumulated += m_occurrencesOut;
            
            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_ACTIVE_INDEX] = 
                DSL_FRAME_INFO_OCCURRENCES_DIRECTION_IN;

            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES] = 
                m_occurrencesInAccumulated + m_occurrencesOutAccumulated;

            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES_DIRECTION_IN] = 
                m_occurrencesInAccumulated;
            pFrameMeta->misc_frame_info[DSL_FRAME_INFO_OCCURRENCES_DIRECTION_OUT] = 
                m_occurrencesOutAccumulated;
                
            DSL_ODE_ACCUMULATOR_PTR pOdeAccumulator = 
                std::dynamic_pointer_cast<OdeAccumulator>(m_pAccumulator);
                
            pOdeAccumulator->HandleOccurrences(shared_from_this(),
                pBuffer, displayMetaData, pFrameMeta);
        }
        // clear the occurrence counters 
        m_occurrencesIn = 0;
        m_occurrencesOut = 0;

        // purge all tracked objects, for all sources that are not in the current frame.
        m_pTrackedObjectsPerSource->Purge(pFrameMeta->frame_num);
        
        return m_occurrences;
    }

    void CrossOdeTrigger::GetTestSettings(uint* minFrameCount, 
        uint* maxTracePoints, uint* testMethod)
    {
        LOG_FUNC();

        *minFrameCount = m_minFrameCount;
        *maxTracePoints = m_maxTracePoints;
        *testMethod = m_testMethod;
    }
    
    void CrossOdeTrigger::SetTestSettings(uint minFrameCount,
        uint maxTracePoints, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        m_minFrameCount = minFrameCount;
        m_maxTracePoints = maxTracePoints;
        m_testMethod = testMethod;
        m_pTrackedObjectsPerSource->SetMaxHistory(m_maxTracePoints);
    }
    
    void CrossOdeTrigger::GetViewSettings(bool* enabled, 
        const char** color, uint* lineWidth)
    {
        LOG_FUNC();

        *enabled = m_traceEnabled;
        *color = m_pTraceColor->GetName().c_str();
        *lineWidth = m_traceLineWidth;
    }
    
    void CrossOdeTrigger::SetViewSettings(bool enabled, 
        DSL_RGBA_COLOR_PTR pColor, uint lineWidth)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_traceEnabled = enabled;
        m_pTraceColor = pColor;
        m_traceLineWidth = lineWidth;
    }        

    void CrossOdeTrigger::Reset()
    {
        LOG_FUNC();
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrencesIn = 0;
            m_occurrencesOut = 0;
            m_occurrencesInAccumulated = 0;
            m_occurrencesOutAccumulated = 0;
        }        
        // call the base class to complete the Reset
        TrackingOdeTrigger::Reset();
    }

    // *****************************************************************************
    
    PersistenceOdeTrigger::PersistenceOdeTrigger(const char* name, 
        const char* source, uint classId, uint limit, uint minimum, 
        uint maximum)
        : TrackingOdeTrigger(name, source, classId, limit, 0)
        , m_minimumMs(minimum*1000.0)
        , m_maximumMs(maximum*1000.0)
    {
        LOG_FUNC();
    }

    PersistenceOdeTrigger::~PersistenceOdeTrigger()
    {
        LOG_FUNC();
    }

    void PersistenceOdeTrigger::GetRange(uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        
        *minimum = m_minimumMs/1000;
        *maximum = m_maximumMs/1000;
    }

    void PersistenceOdeTrigger::SetRange(uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minimumMs = minimum*1000.0;
        m_maximumMs = maximum*1000.0;
    }
    
    bool PersistenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }

        // if this is the first occurrence of any object for this source
        if (!m_pTrackedObjectsPerSource->IsTracked(pFrameMeta->source_id,
            pObjectMeta->object_id))
        {
            // Create a new Tracked object and return without occurence
            m_pTrackedObjectsPerSource->Track(pFrameMeta, 
                pObjectMeta, nullptr);
        }
        else
        {
            std::shared_ptr<TrackedObject> pTrackedObject = 
                m_pTrackedObjectsPerSource->GetObject(pFrameMeta->source_id,
                    pObjectMeta->object_id);
                    
            pTrackedObject->Update(pFrameMeta->frame_num, 
                (NvBbox_Coords*)&pObjectMeta->rect_params);

            double trackedTimeMs = pTrackedObject->GetDurationMs();
            
            LOG_DEBUG("Persistence for tracked object with id = " 
                << pObjectMeta->object_id << " for source = " 
                << pFrameMeta->source_id << ", = " << trackedTimeMs << " ms");
            
            // if the object's tracked time is within range. 
            if (trackedTimeMs >= m_minimumMs and trackedTimeMs <= m_maximumMs)
            {
                // event has been triggered
                IncrementAndCheckTriggerCount();
                m_occurrences++;

                // update the total event count static variable
                s_eventCount++;
    
                // If the client has added a heat mapper, call to add the occurrence data
                if (m_pHeatMapper)
                {
                    std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                        pFrameMeta, pObjectMeta);
                }

                // add the persistence value to the array of misc_obj_info
                // at both the Primary and Persistence specific indecies.
                pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] = 
                pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = 
                    (uint64_t)(trackedTimeMs/1000);
                    
                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, pObjectMeta);
                }
            }
        }
        return true;
    }

    uint PersistenceOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame or m_pTrackedObjectsPerSource->IsEmpty())
            {
                return 0;
            }
            // purge all tracked objects, for all sources that are not in the current frame.
            m_pTrackedObjectsPerSource->Purge(pFrameMeta->frame_num);
        }
        // mutext unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }

    // *****************************************************************************
    
    LatestOdeTrigger::LatestOdeTrigger(const char* name, const char* source, 
        uint classId, uint limit)
        : TrackingOdeTrigger(name, source, classId, limit, 0)
        , m_pLatestObjectMeta(NULL)
        , m_latestTrackedTimeMs(0)
    {
        LOG_FUNC();
    }

    LatestOdeTrigger::~LatestOdeTrigger()
    {
        LOG_FUNC();
    }

    bool LatestOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or !CheckForInside(pObjectMeta))
        {
            return false;
        }

        // if this is the first occurrence of any object for this source
        if (!m_pTrackedObjectsPerSource->IsTracked(pFrameMeta->source_id,
            pObjectMeta->object_id))
        {
            // Create a new Tracked object and return without occurence
            m_pTrackedObjectsPerSource->Track(pFrameMeta, 
                pObjectMeta, nullptr);
        }
        else
        {
            std::shared_ptr<TrackedObject> pTrackedObject = 
                m_pTrackedObjectsPerSource->GetObject(pFrameMeta->source_id,
                    pObjectMeta->object_id);
                    
            pTrackedObject->Update(pFrameMeta->frame_num, 
                (NvBbox_Coords*)&pObjectMeta->rect_params);

            double trackedTimeMs = pTrackedObject->GetDurationMs();
            
            if ((m_pLatestObjectMeta == NULL) or (trackedTimeMs < m_latestTrackedTimeMs))
            {
                m_pLatestObjectMeta = pObjectMeta;
                m_latestTrackedTimeMs = trackedTimeMs;
            }
        }
        return true;
    }
    
    uint LatestOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame or m_pTrackedObjectsPerSource->IsEmpty())
            {
                return 0;
            }
            
            // If we a Latest Object ODE 
            if (m_pLatestObjectMeta != NULL)
            {
                // event has been triggered
                IncrementAndCheckTriggerCount();
                m_occurrences++;

                // update the total event count static variable
                s_eventCount++;

                // If the client has added a heat mapper, call to add the occurrence data
                if (m_pHeatMapper)
                {
                    std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                        pFrameMeta, m_pLatestObjectMeta);
                }
                
                // add the persistence value to the array of misc_obj_info
                // as both the Primary and Persistence specific indecies.
                m_pLatestObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] = 
                m_pLatestObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = 
                    (uint64_t)(m_latestTrackedTimeMs/1000);

                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, m_pLatestObjectMeta);
                }
            
                // clear the Newest Object data for the next frame 
                m_pLatestObjectMeta = NULL;
                m_latestTrackedTimeMs = 0;
            }
            // purge all tracked objects, for all sources that are not in the current frame.
            m_pTrackedObjectsPerSource->Purge(pFrameMeta->frame_num);
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }

    // *****************************************************************************
    
    EarliestOdeTrigger::EarliestOdeTrigger(const char* name, const char* source, 
        uint classId, uint limit)
        : TrackingOdeTrigger(name, source, classId, limit, 0)
        , m_pEarliestObjectMeta(NULL)
        , m_earliestTrackedTimeMs(0)
    {
        LOG_FUNC();
    }

    EarliestOdeTrigger::~EarliestOdeTrigger()
    {
        LOG_FUNC();
    }

    bool EarliestOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta) or 
            !CheckForInside(pObjectMeta))
        {
            return false;
        }

        // if this is the first occurrence of any object for this source
        if (!m_pTrackedObjectsPerSource->IsTracked(pFrameMeta->source_id,
            pObjectMeta->object_id)) 
        {
            // Create a new Tracked object and return without occurence
            m_pTrackedObjectsPerSource->Track(pFrameMeta, 
                pObjectMeta, nullptr);
        }
        else
        {
            std::shared_ptr<TrackedObject> pTrackedObject = 
                m_pTrackedObjectsPerSource->GetObject(pFrameMeta->source_id,
                    pObjectMeta->object_id);
                    
            pTrackedObject->Update(pFrameMeta->frame_num, 
                (NvBbox_Coords*)&pObjectMeta->rect_params);

            double trackedTimeMs = pTrackedObject->GetDurationMs();
                
            if ((m_pEarliestObjectMeta == NULL) or 
                (trackedTimeMs > m_earliestTrackedTimeMs))
            {
                m_pEarliestObjectMeta = pObjectMeta;
                m_earliestTrackedTimeMs = trackedTimeMs;
                
            }
        }
        return true;
    }
    
    uint EarliestOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            if (!m_enabled or m_skipFrame or m_pTrackedObjectsPerSource->IsEmpty())
            {
                return 0;
            }
            
            if (m_pEarliestObjectMeta != NULL)
            {
                // event has been triggered
                IncrementAndCheckTriggerCount();
                m_occurrences++;

                // update the total event count static variable
                s_eventCount++;

                // If the client has added a heat mapper, call to add the occurrence data
                if (m_pHeatMapper)
                {
                    std::dynamic_pointer_cast<OdeHeatMapper>(m_pHeatMapper)->HandleOccurrence(
                        pFrameMeta, m_pEarliestObjectMeta);
                }

                // add the persistence value to the array of misc_obj_info
                // as both the Primary and Persistence specific indecies.
                m_pEarliestObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PERSISTENCE] = 
                m_pEarliestObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = 
                    (uint64_t)(m_earliestTrackedTimeMs/1000);

                for (const auto &imap: m_pOdeActionsIndexed)
                {
                    DSL_ODE_ACTION_PTR pOdeAction = 
                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                    pOdeAction->HandleOccurrence(shared_from_this(), 
                        pBuffer, displayMetaData, pFrameMeta, m_pEarliestObjectMeta);
                }
            
                // clear the Earliest Object data for the next frame 
                m_pEarliestObjectMeta = NULL;
                m_earliestTrackedTimeMs = 0;
            }
            
            // purge all tracked objects, for all sources that are not in the current frame.
            m_pTrackedObjectsPerSource->Purge(pFrameMeta->frame_num);
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }

    // *****************************************************************************
    // AB Trigger Types
    // *****************************************************************************
    
    ABOdeTrigger::ABOdeTrigger(const char* name, 
        const char* source, uint classIdA, uint classIdB, uint limit)
        : OdeTrigger(name, source, classIdA, limit)
        , m_classIdA(classIdA)
        , m_classIdB(classIdB)
    {
        LOG_FUNC();
        
        m_classIdAOnly = (m_classIdA == m_classIdB);
    }

    ABOdeTrigger::~ABOdeTrigger()
    {
        LOG_FUNC();
    }

    void ABOdeTrigger::GetClassIdAB(uint* classIdA, uint* classIdB)
    {
        LOG_FUNC();
        
        *classIdA = m_classIdA;
        *classIdB = m_classIdB;
    }
    
    void ABOdeTrigger::SetClassIdAB(uint classIdA, uint classIdB)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_classIdA = classIdA;
        m_classIdB = classIdB;
        m_classIdAOnly = (m_classIdA == m_classIdB);
    }
    
    bool ABOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id))
        {
            return false;
        }
        
        bool occurrenceAdded(false);
        
        m_classId = m_classIdA;
        if (CheckForMinCriteria(pFrameMeta, pObjectMeta) and 
            CheckForInside(pObjectMeta))
        {
            m_occurrenceMetaListA.push_back(pObjectMeta);
            occurrenceAdded = true;
        }
        else if (!m_classIdAOnly)
        {
            m_classId = m_classIdB;
            if (CheckForMinCriteria(pFrameMeta, pObjectMeta) and 
                CheckForInside(pObjectMeta))
            {
                m_occurrenceMetaListB.push_back(pObjectMeta);
                occurrenceAdded = true;
            }
        }
        
        return occurrenceAdded;
    }

    uint ABOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        if (m_classIdAOnly)
        {
            return PostProcessFrameA(pBuffer, displayMetaData, pFrameMeta);
        }
        return  PostProcessFrameAB(pBuffer, displayMetaData, pFrameMeta);
    }

    // *****************************************************************************
    
    DistanceOdeTrigger::DistanceOdeTrigger(const char* name, const char* source, 
        uint classIdA, uint classIdB, uint limit, uint minimum, uint maximum, 
        uint testPoint, uint testMethod)
        : ABOdeTrigger(name, source, classIdA, classIdB, limit)
        , m_minimum(minimum)
        , m_maximum(maximum)
        , m_testPoint(testPoint)
        , m_testMethod(testMethod)
    {
        LOG_FUNC();
    }

    DistanceOdeTrigger::~DistanceOdeTrigger()
    {
        LOG_FUNC();
    }

    void DistanceOdeTrigger::GetRange(uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        
        *minimum = m_minimum;
        *maximum = m_maximum;
    }

    void DistanceOdeTrigger::SetRange(uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minimum = minimum;
        m_maximum = maximum;
    }

    void DistanceOdeTrigger::GetTestParams(uint* testPoint, uint* testMethod)
    {
        LOG_FUNC();

        *testPoint = m_testPoint;
        *testMethod = m_testMethod;
    }

    void DistanceOdeTrigger::SetTestParams(uint testPoint, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_testPoint = testPoint;
        m_testMethod = testMethod;
    }
    
    
    uint DistanceOdeTrigger::PostProcessFrameA(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrences = 0;
            
            // need at least two objects for intersection to occur
            while (m_enabled and m_occurrenceMetaListA.size() > 1)
            {
                // iterate through the list of object occurrences that passed all min criteria
                for (uint i = 0; i < m_occurrenceMetaListA.size()-1 ; i++) 
                {
                    for (uint j = i+1; j < m_occurrenceMetaListA.size() ; j++) 
                    {
                        if (CheckDistance(m_occurrenceMetaListA[i], 
                            m_occurrenceMetaListA[j]))
                        {
                            // event has been triggered
                            m_occurrences++;
                            IncrementAndCheckTriggerCount();
                            
                             // update the total event count static variable
                            s_eventCount++;

                            // set the primary metric as the current occurrence for this frame
                            m_occurrenceMetaListA[i]->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                = m_occurrences;
                            m_occurrenceMetaListA[j]->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                = m_occurrences;

                            for (const auto &imap: m_pOdeActionsIndexed)
                            {
                                DSL_ODE_ACTION_PTR pOdeAction = 
                                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                                
                                // Invoke each action twice, once for each object in the tested pair
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, displayMetaData, pFrameMeta, m_occurrenceMetaListA[i]);
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, displayMetaData, pFrameMeta, m_occurrenceMetaListA[j]);
                            }
                            if (m_eventLimit and m_triggered >= m_eventLimit)
                            {
                                m_occurrenceMetaListA.clear();
                                break;
                            }
                        }
                    }
                }
                break;
            }   

            // reset for next frame
            m_occurrenceMetaListA.clear();
        }
        // mutext unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }
   
    uint DistanceOdeTrigger::PostProcessFrameAB(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrences = 0;
            
            // need at least one object from each of the two Classes 
            while (m_enabled and m_occurrenceMetaListA.size() and m_occurrenceMetaListB.size())
            {
                // iterate through the list of object occurrences that passed all min criteria
                for (const auto &iterA: m_occurrenceMetaListA) 
                {
                    for (const auto &iterB: m_occurrenceMetaListB) 
                    {
                        // ensure we are not testing the same object which can be in both vectors
                        // if Class Id A and B are specified to be the same.
                        if (iterA != iterB)
                        {
                            if (CheckDistance(iterA, iterB))
                            {
                                // event has been triggered
                                m_occurrences++;
                                IncrementAndCheckTriggerCount();
                                
                                 // update the total event count static variable
                                s_eventCount++;

                                // set the primary metric as the current occurrence 
                                // for this frame
                                iterA->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                    = m_occurrences;
                                iterB->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                    = m_occurrences;

                                for (const auto &imap: m_pOdeActionsIndexed)
                                {
                                    DSL_ODE_ACTION_PTR pOdeAction = 
                                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                                    
                                    // Invoke each action twice, once for each object 
                                    // in the tested pair
                                    pOdeAction->HandleOccurrence(shared_from_this(), 
                                        pBuffer, displayMetaData, pFrameMeta, iterA);
                                    pOdeAction->HandleOccurrence(shared_from_this(), 
                                        pBuffer, displayMetaData, pFrameMeta, iterB);
                                }
                                if (m_eventLimit and m_triggered >= m_eventLimit)
                                {
                                    m_occurrenceMetaListA.clear();
                                    m_occurrenceMetaListB.clear();
                                    break;
                                }
                            }
                        }
                    }
                }
                break;
            }   

            // reset for next frame
            m_occurrenceMetaListA.clear();
            m_occurrenceMetaListB.clear();
        }
        // mutext unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }

    bool DistanceOdeTrigger::CheckDistance(NvDsObjectMeta* pObjectMetaA, 
        NvDsObjectMeta* pObjectMetaB)
    {
        uint distance(0);
        if (m_testPoint == DSL_BBOX_POINT_ANY)
        {
            GeosRectangle rectA(pObjectMetaA->rect_params);
            GeosRectangle rectB(pObjectMetaB->rect_params);
            distance = rectA.Distance(rectB);
        }
        else{
            uint xa(0), ya(0), xb(0), yb(0);
            switch (m_testPoint)
            {
            case DSL_BBOX_POINT_CENTER :
                xa = round(pObjectMetaA->rect_params.left + pObjectMetaA->rect_params.width/2);
                ya = round(pObjectMetaA->rect_params.top + pObjectMetaA->rect_params.height/2);
                xb = round(pObjectMetaB->rect_params.left + pObjectMetaB->rect_params.width/2);
                yb = round(pObjectMetaB->rect_params.top + pObjectMetaB->rect_params.height/2);
                break;
            case DSL_BBOX_POINT_NORTH_WEST :
                xa = round(pObjectMetaA->rect_params.left);
                ya = round(pObjectMetaA->rect_params.top);
                xb = round(pObjectMetaB->rect_params.left);
                yb = round(pObjectMetaB->rect_params.top);
                break;
            case DSL_BBOX_POINT_NORTH :
                xa = round(pObjectMetaA->rect_params.left + pObjectMetaA->rect_params.width/2);
                ya = round(pObjectMetaA->rect_params.top);
                xb = round(pObjectMetaB->rect_params.left + pObjectMetaB->rect_params.width/2);
                yb = round(pObjectMetaB->rect_params.top);
                break;
            case DSL_BBOX_POINT_NORTH_EAST :
                xa = round(pObjectMetaA->rect_params.left + pObjectMetaA->rect_params.width);
                ya = round(pObjectMetaA->rect_params.top);
                xb = round(pObjectMetaB->rect_params.left + pObjectMetaB->rect_params.width);
                yb = round(pObjectMetaB->rect_params.top);
                break;
            case DSL_BBOX_POINT_EAST :
                xa = round(pObjectMetaA->rect_params.left + pObjectMetaA->rect_params.width);
                ya = round(pObjectMetaA->rect_params.top + pObjectMetaA->rect_params.height/2);
                xb = round(pObjectMetaB->rect_params.left + pObjectMetaB->rect_params.width);
                yb = round(pObjectMetaB->rect_params.top + pObjectMetaB->rect_params.height/2);
                break;
            case DSL_BBOX_POINT_SOUTH_EAST :
                xa = round(pObjectMetaA->rect_params.left + pObjectMetaA->rect_params.width);
                ya = round(pObjectMetaA->rect_params.top + pObjectMetaA->rect_params.height);
                xb = round(pObjectMetaB->rect_params.left + pObjectMetaB->rect_params.width);
                yb = round(pObjectMetaB->rect_params.top + pObjectMetaB->rect_params.height);
                break;
            case DSL_BBOX_POINT_SOUTH :
                xa = round(pObjectMetaA->rect_params.left + pObjectMetaA->rect_params.width/2);
                ya = round(pObjectMetaA->rect_params.top + pObjectMetaA->rect_params.height);
                xb = round(pObjectMetaB->rect_params.left + pObjectMetaB->rect_params.width/2);
                yb = round(pObjectMetaB->rect_params.top + pObjectMetaB->rect_params.height);
                break;
            case DSL_BBOX_POINT_SOUTH_WEST :
                xa = round(pObjectMetaA->rect_params.left);
                ya = round(pObjectMetaA->rect_params.top + pObjectMetaA->rect_params.height);
                xb = round(pObjectMetaB->rect_params.left);
                yb = round(pObjectMetaB->rect_params.top + pObjectMetaB->rect_params.height);
                break;
            case DSL_BBOX_POINT_WEST :
                xa = round(pObjectMetaA->rect_params.left);
                ya = round(pObjectMetaA->rect_params.top + pObjectMetaA->rect_params.height/2);
                xb = round(pObjectMetaB->rect_params.left);
                yb = round(pObjectMetaB->rect_params.top + pObjectMetaB->rect_params.height/2);
                break;
            default:
                LOG_ERROR("Invalid DSL_BBOX_POINT = '" << m_testPoint 
                    << "' for DistanceOdeTrigger Trigger '" << GetName() << "'");
                throw;
            }

            GeosPoint pointA(xa, ya);
            GeosPoint pointB(xb, yb);
            distance = pointA.Distance(pointB);
        }
        
        uint minimum(0), maximum(0);
        switch (m_testMethod)
        {
        case DSL_DISTANCE_METHOD_FIXED_PIXELS :
            minimum = m_minimum;
            maximum = m_maximum;
            break;
        case DSL_DISTANCE_METHOD_PERCENT_WIDTH_A :
            minimum = uint((m_minimum*pObjectMetaA->rect_params.width)/100);
            maximum = uint((m_maximum*pObjectMetaA->rect_params.width)/100);
            break;
        case DSL_DISTANCE_METHOD_PERCENT_WIDTH_B :
            minimum = uint((m_minimum*pObjectMetaB->rect_params.width)/100);
            maximum = uint((m_maximum*pObjectMetaB->rect_params.width)/100);
            break;
        case DSL_DISTANCE_METHOD_PERCENT_HEIGHT_A :
            minimum = uint((m_minimum*pObjectMetaA->rect_params.height)/100);
            maximum = uint((m_maximum*pObjectMetaA->rect_params.height)/100);
            break;
        case DSL_DISTANCE_METHOD_PERCENT_HEIGHT_B :
            minimum = uint((m_minimum*pObjectMetaB->rect_params.height)/100);
            maximum = uint((m_maximum*pObjectMetaB->rect_params.height)/100);
            break;
        }    
        return (minimum > distance or maximum < distance);
    }

    // *****************************************************************************
    
    IntersectionOdeTrigger::IntersectionOdeTrigger(const char* name, 
        const char* source, uint classIdA, uint classIdB, uint limit)
        : ABOdeTrigger(name, source, classIdA, classIdB, limit)
    {
        LOG_FUNC();
    }

    IntersectionOdeTrigger::~IntersectionOdeTrigger()
    {
        LOG_FUNC();
    }
    
    uint IntersectionOdeTrigger::PostProcessFrameA(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrences = 0;
            
            // need at least two objects for intersection to occur
            if (m_enabled and m_occurrenceMetaListA.size() > 1)
            {
                // iterate through the list of object occurrences that passed all min criteria
                for (uint i = 0; i < m_occurrenceMetaListA.size()-1 ; i++) 
                {
                    for (uint j = i+1; j < m_occurrenceMetaListA.size() ; j++) 
                    {
                        // check each in turn for any frame overlap
                        GeosRectangle rectA(m_occurrenceMetaListA[i]->rect_params);
                        GeosRectangle rectB(m_occurrenceMetaListA[j]->rect_params);
                        if (rectA.Overlaps(rectB))
                        {
                            // event has been triggered
                            m_occurrences++;
                            IncrementAndCheckTriggerCount();
                            
                             // update the total event count static variable
                            s_eventCount++;

                            // set the primary metric as the current occurrence for this frame
                            m_occurrenceMetaListA[i]->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                = m_occurrences;
                            m_occurrenceMetaListA[j]->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                = m_occurrences;

                            for (const auto &imap: m_pOdeActionsIndexed)
                            {
                                DSL_ODE_ACTION_PTR pOdeAction = 
                                    std::dynamic_pointer_cast<OdeAction>(imap.second);
                                
                                // Invoke each action twice, once for each object in the tested pair
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, displayMetaData, pFrameMeta, m_occurrenceMetaListA[i]);
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, displayMetaData, pFrameMeta, m_occurrenceMetaListA[j]);
                            }
                            if (m_eventLimit and m_triggered >= m_eventLimit)
                            {
                                m_occurrenceMetaListA.clear();
                                return m_occurrences;
                            }
                        }
                    }
                }
            }   

            // reset for next frame
            m_occurrenceMetaListA.clear();
        }
        // mutext unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
   }

    uint IntersectionOdeTrigger::PostProcessFrameAB(GstBuffer* pBuffer, 
        std::vector<NvDsDisplayMeta*>& displayMetaData,  NvDsFrameMeta* pFrameMeta)
    {
        // create scope so the property-mutex can be unlocked before
        // calling the base-class PostProcessFrame which locks the mutex.
        {
            // Note: function is called from the system (callback) context
            // Gaurd against property updates from the client API
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
            
            m_occurrences = 0;
            
            // need at least one object from each of the two Classes 
            if (m_enabled and m_occurrenceMetaListA.size() and m_occurrenceMetaListB.size())
            {
                // iterate through the list of object occurrences that passed all min criteria
                for (const auto &iterA: m_occurrenceMetaListA) 
                {
                    for (const auto &iterB: m_occurrenceMetaListB) 
                    {
                        // ensure we are not testing the same object which can be in both vectors
                        // if Class Id A and B are specified to be the same.
                        if (iterA != iterB)
                        {
                            // check each in turn for any frame overlap
                            GeosRectangle rectA(iterA->rect_params);
                            GeosRectangle rectB(iterB->rect_params);
                            if (rectA.Overlaps(rectB))
                            {
                                // event has been triggered
                                m_occurrences++;
                                IncrementAndCheckTriggerCount();
                                
                                 // update the total event count static variable
                                s_eventCount++;

                                // set the primary metric as the current occurrence 
                                // for this frame
                                iterA->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                    = m_occurrences;
                                iterB->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] 
                                    = m_occurrences;
                                
                                for (const auto &imap: m_pOdeActionsIndexed)
                                {
                                    DSL_ODE_ACTION_PTR pOdeAction = 
                                        std::dynamic_pointer_cast<OdeAction>(imap.second);
                                    
                                    // Invoke each action twice, once for each object 
                                    // in the tested pair
                                    pOdeAction->HandleOccurrence(shared_from_this(), 
                                        pBuffer, displayMetaData, pFrameMeta, iterA);
                                    pOdeAction->HandleOccurrence(shared_from_this(), 
                                        pBuffer, displayMetaData, pFrameMeta, iterB);
                                }
                                if (m_eventLimit and m_triggered >= m_eventLimit)
                                {
                                    m_occurrenceMetaListA.clear();
                                    m_occurrenceMetaListB.clear();
                                    return m_occurrences;
                                }
                            }
                        }
                    }
                }
            }   

            // reset for next frame
            m_occurrenceMetaListA.clear();
            m_occurrenceMetaListB.clear();
        }
        // mutex unlocked - safe to call base class
        return OdeTrigger::PostProcessFrame(pBuffer,
            displayMetaData, pFrameMeta);
    }
}