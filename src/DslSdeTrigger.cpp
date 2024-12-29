/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "DslSdeTrigger.h"
//#include "DslSdeAction.h"
#include "DslServices.h"

namespace DSL
{

    // Initialize static Event Counter
    uint64_t SdeTrigger::s_eventCount = 0;

    SdeTrigger::SdeTrigger(const char* name, const char* source, 
        uint classId, uint limit)
        : DeTriggerBase(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    SdeTrigger::~SdeTrigger()
    {
        LOG_FUNC();        
    }

    void SdeTrigger::PreProcessFrame(GstBuffer* pBuffer, 
        NvDsAudioFrameMeta* pFrameMeta)
    {
        LOG_FUNC();

        // Reset the occurrences from the last frame, even if disabled  
        m_occurrences = 0;

        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id))
        {
            return;
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

    uint SdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsAudioFrameMeta* pFrameMeta)
    {
        LOG_FUNC();
        
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
                    imap.first(DSL_TRIGGER_LIMIT_FRAME_REACHED, 
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
       
        return m_occurrences;
    }        
    
    bool SdeTrigger::CheckForMinCriteria(NvDsAudioFrameMeta* pFrameMeta)
    {
        LOG_FUNC();
        
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
        // Filter on unique source-id 
        if (!CheckForSourceId(pFrameMeta->source_id))
        {
            return false;
        }
        // Filter on Class id if set
        if ((m_classId != DSL_SDE_ANY_CLASS) and 
            (m_classId != pFrameMeta->class_id))
        {
            return false;
        }
        // Ensure that the minimum Inference confidence has been reached
        if (pFrameMeta->confidence > 0 and 
            pFrameMeta->confidence < m_minConfidence)
        {
            return false;
        }
        // Ensure that the maximum Inference confidence has been reached
        if (pFrameMeta->confidence > 0 and m_maxConfidence and
            pFrameMeta->confidence > m_maxConfidence)
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

    // *****************************************************************************

    OccurrenceSdeTrigger::OccurrenceSdeTrigger(const char* name, 
        const char* source, uint classId, uint limit)
        : SdeTrigger(name, source, classId, limit)
    {
        LOG_FUNC();
    }

    OccurrenceSdeTrigger::~OccurrenceSdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool OccurrenceSdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, 
        NvDsAudioFrameMeta* pFrameMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (!CheckForMinCriteria(pFrameMeta))
        {
            return false;
        }
        
        IncrementAndCheckTriggerCount();
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        // // set the primary metric as the current occurrence for this frame
        // pObjectMeta->misc_obj_info[DSL_OBJECT_INFO_PRIMARY_METRIC] = m_occurrences;

        for (const auto &imap: m_pActionsIndexed)
        {
            DSL_SDE_ACTION_PTR pSdeAction = 
                std::dynamic_pointer_cast<SdeAction>(imap.second);
            pSdeAction->HandleOccurrence(shared_from_this(), pBuffer, 
                pFrameMeta);
            // try
            // {
            //     pSdeAction->HandleOccurrence(shared_from_this(), pBuffer, 
            //         displayMetaData, pFrameMeta, pObjectMeta);
            // }
            // catch(...)
            // {
            //     LOG_ERROR("Trigger '" << GetName() << "' => Action '" 
            //         << pSdeAction->GetName() << "' threw exception");
            // }
        }
        return true;
    }

}
