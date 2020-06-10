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
        uint classId, uint limit)
        : Base(name)
        , m_wName(m_name.begin(), m_name.end())
        , m_enabled(true)
        , m_classId(classId)
        , m_sourceId(0)
        , m_triggered(0)
        , m_limit(limit)
        , m_occurrences(0)
        , m_minConfidence(0)
        , m_minWidth(0)
        , m_minHeight(0)
        , m_areaParams{0}
        , m_minFrameCountN(1)
        , m_minFrameCountD(1)
    {
        LOG_FUNC();
        
        m_areaParams.has_bg_color = true;
        m_areaParams.bg_color.red = 1.0;
        m_areaParams.bg_color.green = 1.0;
        m_areaParams.bg_color.blue = 1.0;
        m_areaParams.bg_color.alpha = 0.5;

        g_mutex_init(&m_propertyMutex);
    }

    OdeType::~OdeType()
    {
        LOG_FUNC();
        
        RemoveAllChildren();
        
        g_mutex_clear(&m_propertyMutex);
    }

    void OdeType::PreProcessFrame(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled)
        {
            return;
        }
        if (m_areaParams.width and m_areaParams.height)
        {
            NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(pBuffer);
            NvDsDisplayMeta* pDisplayMeta = nvds_acquire_display_meta_from_pool(batchMeta);
            
            pDisplayMeta->num_rects = 1;
            pDisplayMeta->rect_params[0] = m_areaParams;
            nvds_add_display_meta_to_frame(pFrameMeta, pDisplayMeta);
        }
    }

    bool OdeType::GetEnabled()
    {
        LOG_FUNC();
        
        return m_enabled;
    }
    
    void OdeType::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_enabled = enabled;
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
    
    void OdeType::GetArea(uint* left, uint* top, uint* width, uint* height)
    {
        LOG_FUNC();
        
        *left = m_areaParams.left;
        *top = m_areaParams.top;
        *width = m_areaParams.width;
        *height = m_areaParams.height;
    }
    
    void OdeType::SetArea(uint left, uint top, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_areaParams.left = left;
        m_areaParams.top = top;
        m_areaParams.width = width;
        m_areaParams.height = height;
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

    bool OdeType::checkForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        // Ensure enabled, limit has not been exceeded, and filter 
        // on correct Class ID and Source ID 
        if ((!m_enabled) or
            (m_limit and m_triggered >= m_limit) or 
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
        // If area defined, check for overlay
        if ((m_areaParams.width and m_areaParams.height) and !doesOverlap(pObjectMeta->rect_params))
        {
            return false;
        }
        return true;
    }

    inline bool OdeType::valueInRange(int value, int min, int max)
    { 
        return (value >= min) && (value <= max);
    }

    inline bool OdeType::doesOverlap(NvOSD_RectParams rectParams)
    {
        bool xOverlap = valueInRange(rectParams.left, m_areaParams.left, m_areaParams.left + m_areaParams.width) ||
                        valueInRange(m_areaParams.left, rectParams.left, rectParams.left + rectParams.width);

        bool yOverlap = valueInRange(rectParams.top, m_areaParams.top, m_areaParams.top + m_areaParams.height) ||
                        valueInRange(m_areaParams.top, rectParams.top, rectParams.top + rectParams.height);

        return xOverlap && yOverlap;
    }    
    
    // *****************************************************************************

    OccurrenceOdeType::OccurrenceOdeType(const char* name, uint classId, uint limit)
        : OdeType(name, classId, limit)
    {
        LOG_FUNC();
    }

    OccurrenceOdeType::~OccurrenceOdeType()
    {
        LOG_FUNC();
    }
    
    bool OccurrenceOdeType::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
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
    
    AbsenceOdeType::AbsenceOdeType(const char* name, uint classId, uint limit)
        : OdeType(name, classId, limit)
    {
        LOG_FUNC();
    }

    AbsenceOdeType::~AbsenceOdeType()
    {
        LOG_FUNC();
    }
    
    bool AbsenceOdeType::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }
    
    bool AbsenceOdeType::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
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
    
    SummationOdeType::SummationOdeType(const char* name, uint classId, uint limit)
        : OdeType(name, classId, limit)
    {
        LOG_FUNC();
    }

    SummationOdeType::~SummationOdeType()
    {
        LOG_FUNC();
    }
    
    bool SummationOdeType::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    bool SummationOdeType::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pChildren)
        {
            DSL_ODE_ACTION_PTR pAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        // reset for next frame
        m_occurrences = 0;
        return true;
   }

}