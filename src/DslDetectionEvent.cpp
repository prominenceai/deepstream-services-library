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
#include "DslDetectionEvent.h"

namespace DSL
{

    DetectionEvent::DetectionEvent(const char* name, uint classId)
        : Base(name)
        , m_classId(classId)
        , m_minWidth(0)
        , m_minHeight(0)
        , m_minFrameCountN(0)
        , m_minFrameCountD(0)
    {
        LOG_FUNC();

        g_mutex_init(&m_propertyMutex);
    }

    DetectionEvent::~DetectionEvent()
    {
        LOG_FUNC();

        g_mutex_clear(&m_propertyMutex);
    }
    
    uint DetectionEvent::GetClassId()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_classId;
    }
    
    void DetectionEvent::SetClassId(uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_classId = classId;
    }
    
    void DetectionEvent::GetMinDimensions(uint* minWidth, uint* minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        *minWidth = m_minWidth;
        *minHeight = m_minHeight;
        
    }

    void DetectionEvent::SetMinDimensions(uint minWidth, uint minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minWidth = minWidth;
        m_minHeight = minHeight;
    }

    void DetectionEvent::GetMinFrameCount(uint* minFrameCountN, uint* minFrameCountD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        *minFrameCountN = m_minFrameCountN;
        *minFrameCountD = m_minFrameCountD;
    }

    void DetectionEvent::SetMinFrameCount(uint minFrameCountN, uint minFrameCountD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minFrameCountN = minFrameCountN;
        m_minFrameCountD = minFrameCountD;
    }

    FirstOccurrenceEvent::FirstOccurrenceEvent(const char* name, uint classId)
        : DetectionEvent(name, classId)
    {
        LOG_FUNC();

    }

    FirstOccurrenceEvent::~FirstOccurrenceEvent()
    {
        LOG_FUNC();

    }

    FirstAbsenceEvent::FirstAbsenceEvent(const char* name, uint classId)
        : DetectionEvent(name, classId)
    {
        LOG_FUNC();

    }

    FirstAbsenceEvent::~FirstAbsenceEvent()
    {
        LOG_FUNC();

    }
}