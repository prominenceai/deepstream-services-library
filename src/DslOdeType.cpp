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
        , m_eventType(eventType)
        , m_classId(classId)
        , m_triggered(0)
        , m_limit(limit)
        , m_minConfidence(0)
        , m_minWidth(0)
        , m_minHeight(0)
        , m_minFrameCountN(0)
        , m_minFrameCountD(0)
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

    void OdeType::HandleOccurrence(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // NOTE: private funtion... do not lock mutex
        
        // update the triggered count member variable
        m_triggered++;
        
        // update the total event count static variable
        s_eventCount++;
        
        // check to see if this Detection Event has any child Event actions to invoke 
        // before building the Event Occurrence Data data structure
        if (!m_pChildren.size())
        {
            return;
        }

        DSL_ODE_OCCURRENCE_PTR pOdeOccurrence = DSL_ODE_OCCURRENCE_NEW();
        m_name.copy(pOdeOccurrence->event_name, MAX_NAME_SIZE-1, 0);
        
        pOdeOccurrence->event_type = m_eventType;
        pOdeOccurrence->event_id = s_eventCount;
        pOdeOccurrence->ntp_timestamp = pFrameMeta->ntp_timestamp;
        pOdeOccurrence->source_id = pFrameMeta->source_id;
        pOdeOccurrence->frame_num = pFrameMeta->frame_num;
        pOdeOccurrence->source_frame_width = pFrameMeta->source_frame_width;
        pOdeOccurrence->source_frame_height = pFrameMeta->source_frame_height;
        pOdeOccurrence->class_id = pObjectMeta->class_id;
        pOdeOccurrence->object_id = pObjectMeta->object_id; 
        pOdeOccurrence->box.left = pObjectMeta->rect_params.left;
        pOdeOccurrence->box.top = pObjectMeta->rect_params.top;
        pOdeOccurrence->box.width = pObjectMeta->rect_params.width;
        pOdeOccurrence->box.height = pObjectMeta->rect_params.height;
        pOdeOccurrence->min_confidence = m_minConfidence;
        pOdeOccurrence->box_criteria.top = 0;
        pOdeOccurrence->box_criteria.left = 0;
        pOdeOccurrence->box_criteria.width = m_minWidth;
        pOdeOccurrence->box_criteria.height = m_minHeight;
        pOdeOccurrence->min_frame_count_n = m_minFrameCountN;
        pOdeOccurrence->min_frame_count_d = m_minFrameCountD;
        
        for (const auto &imap: m_pChildren)
        {
            DSL_ODE_ACTION_PTR pAction = std::dynamic_pointer_cast<EventAction>(imap.second);
            pAction->HandleOccurrence(pOdeOccurrence);
        }
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
    
    bool FirstOccurrenceEvent::CheckForOccurrence(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if ((m_limit and m_triggered == m_limit) or (m_classId != pObjectMeta->class_id))
        {
            return false;
        }
        HandleOccurrence(pFrameMeta, pObjectMeta);
        return true;
    }

    // *****************************************************************************

    EveryOccurrenceEvent::EveryOccurrenceEvent(const char* name, uint classId)
        : OdeType(name, DSL_ODE_TYPE_FIRST_OCCURRENCE, classId, LIMIT_ONE)
    {
        LOG_FUNC();
    }

    EveryOccurrenceEvent::~EveryOccurrenceEvent()
    {
        LOG_FUNC();
    }
    
    bool EveryOccurrenceEvent::CheckForOccurrence(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        if (m_classId != pObjectMeta->class_id)
        {
            return false;
        }
        HandleOccurrence(pFrameMeta, pObjectMeta);
        return true;
    }
    
}