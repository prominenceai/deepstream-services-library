/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
#include "DslDeTriggerBase.h"
#include "DslServices.h"

namespace DSL
{
   
    DeTriggerBase::DeTriggerBase(const char* name,
        const char* source, uint classId, uint limit)
        : DeBase(name)
        , m_wName(m_name.begin(), m_name.end())
        , m_nextActionIndex(0)
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
        , m_minFrameCountN(1)
        , m_minFrameCountD(1)
        , m_inferDoneOnly(false)
        , m_resetTimeout(0)
        , m_resetTimerId(0)
        , m_interval(0)
        , m_intervalCounter(0)
        , m_skipFrame(false)
    {   
        LOG_FUNC();
    }

    DeTriggerBase::~DeTriggerBase()
    {
        LOG_FUNC();
        
        RemoveAllActions();
        
        if (m_resetTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_resetTimerMutex);
            g_source_remove(m_resetTimerId);
        }
    }

    bool DeTriggerBase::AddAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_pActions.find(pChild->GetName()) != m_pActions.end())
        {
            LOG_ERROR("Action '" << pChild->GetName() 
                << "' is already a child of Trigger '" << GetName() << "'");
            return false;
        }
        
        // increment next index, assign to the Action, and update parent releationship.
        pChild->SetIndex(++m_nextActionIndex);
        pChild->AssignParentName(GetName());

        // Add the shared pointer to child to both Maps, by name and index
        m_pActions[pChild->GetName()] = pChild;
        m_pActionsIndexed[m_nextActionIndex] = pChild;
        
        return true;
    }

    bool DeTriggerBase::RemoveAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_pActions.find(pChild->GetName()) == m_pActions.end())
        {
            LOG_WARN("'" << pChild->GetName() 
                <<"' is not a child of Trigger '" << GetName() << "'");
            return false;
        }
        
        // Erase the child from both maps
        m_pActions.erase(pChild->GetName());
        m_pActionsIndexed.erase(pChild->GetIndex());
        
        // Clear the parent relationship and index
        pChild->ClearParentName();
        pChild->SetIndex(0);
        return true;
    }
    
    void DeTriggerBase::RemoveAllActions()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        for (auto &imap: m_pActions)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() 
                <<"' from Trigger '" << GetName() << "'");
            imap.second->ClearParentName();
        }
        m_pActions.clear();
        m_pActionsIndexed.clear();
    }
    
    void DeTriggerBase::Reset()
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
                imap.first(DSL_TRIGGER_LIMIT_COUNTS_RESET, 
                    m_eventLimit, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Limit-State-Change-Lister");
            }
        }
    }

    int TriggerResetTimeoutHandler(gpointer pTrigger)
    {
        return static_cast<DeTriggerBase*>(pTrigger)->
            HandleResetTimeout();
    }

    int DeTriggerBase::HandleResetTimeout()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_resetTimerMutex);
        
        m_resetTimerId = 0;
        Reset();
        
        // One shot - return false.
        return false;
    }
    
    uint DeTriggerBase::GetResetTimeout()
    {
        LOG_FUNC();
        
        return m_resetTimeout;
    }
        
    void DeTriggerBase::SetResetTimeout(uint timeout)
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
    
    bool DeTriggerBase::IsResetTimerRunning()
    {
        LOG_FUNC();

        return m_resetTimerId;
    }
    
    bool DeTriggerBase::AddLimitStateChangeListener(
        dsl_trigger_limit_state_change_listener_cb listener, void* clientData)
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
    
    bool DeTriggerBase::RemoveLimitStateChangeListener(
        dsl_trigger_limit_state_change_listener_cb listener)
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
        
    uint DeTriggerBase::GetClassId()
    {
        LOG_FUNC();
        
        return m_classId;
    }
    
    void DeTriggerBase::SetClassId(uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_classId = classId;
    }

    uint DeTriggerBase::GetEventLimit()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_eventLimit;
    }
    
    void DeTriggerBase::SetEventLimit(uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_eventLimit = limit;
        
        // iterate through the map of limit-event-listeners calling each
        for(auto const& imap: m_limitStateChangeListeners)
        {
            try
            {
                imap.first(DSL_TRIGGER_LIMIT_EVENT_CHANGED, 
                    m_eventLimit, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Limit-State-Change-Lister");
            }
        }
    }

    uint DeTriggerBase::GetFrameLimit()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_frameLimit;
    }
    
    void DeTriggerBase::SetFrameLimit(uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_frameLimit = limit;
        
        // iterate through the map of limit-event-listeners calling each
        for(auto const& imap: m_limitStateChangeListeners)
        {
            try
            {
                imap.first(DSL_TRIGGER_LIMIT_FRAME_CHANGED, 
                    m_frameLimit, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Limit-State-Change-Lister");
            }
        }
    }

    const char* DeTriggerBase::GetSource()
    {
        LOG_FUNC();
        
        if (m_source.size())
        {
            return m_source.c_str();
        }
        return NULL;
    }
    
    void DeTriggerBase::SetSource(const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_source.assign(source);
    }

    void DeTriggerBase::_setSourceId(int id)
    {
        LOG_FUNC();
        
        m_sourceId = id;
    }
    
    const char* DeTriggerBase::GetInfer()
    {
        LOG_FUNC();
        
        if (m_infer.size())
        {
            return m_infer.c_str();
        }
        return NULL;
    }
    
    void DeTriggerBase::SetInfer(const char* infer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_infer.assign(infer);
    }

    void DeTriggerBase::_setInferId(int id)
    {
        LOG_FUNC();
        
        m_inferId = id;
    }
    
    float DeTriggerBase::GetMinConfidence()
    {
        LOG_FUNC();
        
        return m_minConfidence;
    }
    
    void DeTriggerBase::SetMinConfidence(float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minConfidence = minConfidence;
    }
    
    float DeTriggerBase::GetMaxConfidence()
    {
        LOG_FUNC();
        
        return m_maxConfidence;
    }
    
    void DeTriggerBase::SetMaxConfidence(float maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_maxConfidence = maxConfidence;
    }


    bool DeTriggerBase::GetInferDoneOnlySetting()
    {
        LOG_FUNC();
        
        return m_inferDoneOnly;
    }
    
    void DeTriggerBase::SetInferDoneOnlySetting(bool inferDoneOnly)
    {
        LOG_FUNC();
        
        m_inferDoneOnly = inferDoneOnly;
    }
    
    void DeTriggerBase::GetMinFrameCount(uint* minFrameCountN, uint* minFrameCountD)
    {
        LOG_FUNC();
        
        *minFrameCountN = m_minFrameCountN;
        *minFrameCountD = m_minFrameCountD;
    }

    void DeTriggerBase::SetMinFrameCount(uint minFrameCountN, uint minFrameCountD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minFrameCountN = minFrameCountN;
        m_minFrameCountD = minFrameCountD;
    }

    uint DeTriggerBase::GetInterval()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_interval;
    }
    
    void DeTriggerBase::SetInterval(uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_interval = interval;
        m_intervalCounter = 0;
    }
    
    bool DeTriggerBase::CheckForSourceId(int sourceId)
    {
        LOG_FUNC();

        // Filter on Source id if set
        if (m_source.size())
        {
            // a "one-time-get" of the source Id from the source name
            if (m_sourceId == -1)
            {
                
                Services::GetServices()->SourceUniqueIdGet(m_source.c_str(), 
                    &m_sourceId);
            }
            if (m_sourceId != sourceId)
            {
                return false;
            }
        }
        return true;
    }

    bool DeTriggerBase::CheckForInferId(int inferId)
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

    void DeTriggerBase::IncrementAndCheckTriggerCount()
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
                    imap.first(DSL_TRIGGER_LIMIT_EVENT_REACHED, 
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
}

