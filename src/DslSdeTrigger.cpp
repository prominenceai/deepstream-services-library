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
