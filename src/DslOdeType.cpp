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
#include "DslOdeType.h"
#include "DslOdeAction.h"
#define LIMIT_ONE 1
#define LIMIT_NONE 0

namespace DSL
{

    // Initialize static Event Counter
    uint64_t OdeType::s_eventCount = 0;

    OdeType::OdeType(const char* name, 
        uint eventType, uint classId, uint64_t limit)
        : Base(name)
        , m_wName(m_name.begin(), m_name.end())
        , m_eventType(eventType)
        , m_classId(classId)
        , m_sourceId(0)
        , m_triggered(0)
        , m_limit(limit)
        , m_occurrences(0)
        , m_minConfidence(0)
        , m_minWidth(0)
        , m_minHeight(0)
        , m_minFrameCountN(1)
        , m_minFrameCountD(1)
    {
        LOG_FUNC();

        g_mutex_init(&m_propertyMutex);
    }

    OdeType::~OdeType()
    {
        LOG_FUNC();

        g_mutex_clear(&m_propertyMutex);
    }
    
    uint OdeType::GetClassId()
    {
        LOG_FUNC();
        
        return m_classId;
    }
    
    void OdeType::SetClassId(uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_classId = classId;
    }

    uint OdeType::GetSourceId()
    {
        LOG_FUNC();
        
        return m_sourceId;
    }
    
    void OdeType::SetSourceId(uint sourceId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_sourceId = sourceId;
    }

    float OdeType::GetMinConfidence()
    {
        LOG_FUNC();
        
        return m_minConfidence;
    }
    
    void OdeType::SetMinConfidence(float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minConfidence = minConfidence;
    }
    
    void OdeType::GetMinDimensions(uint* minWidth, uint* minHeight)
    {
        LOG_FUNC();
        
        *minWidth = m_minWidth;
        *minHeight = m_minHeight;
        
    }

    void OdeType::SetMinDimensions(uint minWidth, uint minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minWidth = minWidth;
        m_minHeight = minHeight;
    }

    void OdeType::GetMinFrameCount(uint* minFrameCountN, uint* minFrameCountD)
    {
        LOG_FUNC();
        
        *minFrameCountN = m_minFrameCountN;
        *minFrameCountD = m_minFrameCountD;
    }

    void OdeType::SetMinFrameCount(uint minFrameCountN, uint minFrameCountD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minFrameCountN = minFrameCountN;
        m_minFrameCountD = minFrameCountD;
    }

    bool OdeType::CheckForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        // Don't exceed trigger count, and filter on correct Class ID
        if ((m_limit and m_triggered == m_limit) or 
            (m_classId != pObjectMeta->class_id) or
            (m_sourceId and m_sourceId != pFrameMeta->source_id))
        {
            return false;
        }
        // Ensure that the minimum confidence has been reached
        if (pObjectMeta->confidence < m_minConfidence)
        {
            return false;
        }
        // If defined, check for minimum dimensions
        if ((m_minWidth and pObjectMeta->rect_params.width < m_minWidth) or
            (m_minHeight and pObjectMeta->rect_params.height < m_minHeight))
        {
            return false;
        }
        return true;
    }


    // *****************************************************************************
    
    FirstOccurrenceEvent::FirstOccurrenceEvent(const char* name, uint classId)
        : OdeType(name, DSL_ODE_TYPE_FIRST_OCCURRENCE, classId, LIMIT_ONE)
    {
        LOG_FUNC();
    }

    FirstOccurrenceEvent::~FirstOccurrenceEvent()
    {
        LOG_FUNC();
    }
    
    bool FirstOccurrenceEvent::CheckForOccurrence(GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_triggered++;
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pChildren)
        {
            DSL_ODE_ACTION_PTR pAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, pObjectMeta);
        }
        return true;
    }

    // *****************************************************************************

    EveryOccurrenceEvent::EveryOccurrenceEvent(const char* name, uint classId)
        : OdeType(name, DSL_ODE_TYPE_FIRST_OCCURRENCE, classId, LIMIT_NONE)
    {
        LOG_FUNC();
    }

    EveryOccurrenceEvent::~EveryOccurrenceEvent()
    {
        LOG_FUNC();
    }
    
    bool EveryOccurrenceEvent::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }

        m_triggered++;
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pChildren)
        {
            DSL_ODE_ACTION_PTR pAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, pObjectMeta);
        }
        return true;
    }

    // *****************************************************************************
    
    FirstAbsenceEvent::FirstAbsenceEvent(const char* name, uint classId)
        : OdeType(name, DSL_ODE_TYPE_FIRST_ABSENCE, classId, LIMIT_ONE)
    {
        LOG_FUNC();
    }

    FirstAbsenceEvent::~FirstAbsenceEvent()
    {
        LOG_FUNC();
    }
    
    bool FirstAbsenceEvent::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    bool FirstAbsenceEvent::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (m_triggered or m_occurrences)
        {
            m_occurrences = 0;
            return false;
        }
        
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pChildren)
        {
            DSL_ODE_ACTION_PTR pAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return true;
   }

    // *****************************************************************************
    
    EveryAbsenceEvent::EveryAbsenceEvent(const char* name, uint classId)
        : OdeType(name, DSL_ODE_TYPE_FIRST_ABSENCE, classId, LIMIT_NONE)
    {
        LOG_FUNC();
    }

    EveryAbsenceEvent::~EveryAbsenceEvent()
    {
        LOG_FUNC();
    }
    
    bool EveryAbsenceEvent::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    bool EveryAbsenceEvent::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (m_occurrences)
        {
            // reset for next frame
            m_occurrences = 0;
        }

        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pChildren)
        {
            DSL_ODE_ACTION_PTR pAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return true;
   }

}