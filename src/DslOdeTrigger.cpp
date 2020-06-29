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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"
#include "DslOdeArea.h"

namespace DSL
{

    // Initialize static Event Counter
    uint64_t OdeTrigger::s_eventCount = 0;

    OdeTrigger::OdeTrigger(const char* name, 
        uint classId, uint limit)
        : Base(name)
        , m_wName(m_name.begin(), m_name.end())
        , m_enabled(true)
        , m_classId(classId)
        , m_sourceId(DSL_ODE_ANY_SOURCE)
        , m_triggered(0)
        , m_limit(limit)
        , m_occurrences(0)
        , m_minConfidence(0)
        , m_minWidth(0)
        , m_minHeight(0)
        , m_maxWidth(0)
        , m_maxHeight(0)
        , m_minFrameCountN(1)
        , m_minFrameCountD(1)
        , m_inferDoneOnly(false)
    {
        LOG_FUNC();

        g_mutex_init(&m_propertyMutex);
    }

    OdeTrigger::~OdeTrigger()
    {
        LOG_FUNC();
        
        RemoveAllActions();
        RemoveAllAreas();
        
        g_mutex_clear(&m_propertyMutex);
    }
    
    bool OdeTrigger::AddAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeActions.find(pChild->GetName()) != m_pOdeActions.end())
        {
            LOG_ERROR("ODE Area '" << pChild->GetName() << "' is already a child of ODE Trigger'" << GetName() << "'");
            return false;
        }
        m_pOdeActions[pChild->GetName()] = pChild;
        return true;
    }

    bool OdeTrigger::RemoveAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeActions.find(pChild->GetName()) == m_pOdeActions.end())
        {
            LOG_WARN("'" << pChild->GetName() <<"' is not a child of ODE Trigger '" << GetName() << "'");
            return false;
        }
        m_pOdeActions.erase(pChild->GetName());
        return true;
    }
    
    void OdeTrigger::RemoveAllActions()
    {
        LOG_FUNC();
        
        for (auto &imap: m_pOdeActions)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() <<"' from Parent '" << GetName() << "'");
        }
        m_pOdeActions.clear();
    }
    
    bool OdeTrigger::AddArea(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeAreas.find(pChild->GetName()) != m_pOdeAreas.end())
        {
            LOG_ERROR("ODE Area '" << pChild->GetName() << "' is already a child of ODE Trigger'" << GetName() << "'");
            return false;
        }
        m_pOdeAreas[pChild->GetName()] = pChild;
        return true;
    }

    bool OdeTrigger::RemoveArea(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        m_pOdeAreas.erase(pChild->GetName());
        return true;
    }
    
    void OdeTrigger::RemoveAllAreas()
    {
        LOG_FUNC();
        
        for (auto &imap: m_pOdeAreas)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() <<"' from Parent '" << GetName() << "'");
            imap.second->ClearParentName();
        }
        m_pOdeAreas.clear();
    }
    
    void OdeTrigger::Reset()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_triggered = 0;
    }
        
    bool OdeTrigger::GetEnabled()
    {
        LOG_FUNC();
        
        return m_enabled;
    }
    
    void OdeTrigger::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        LOG_INFO("Setting enable to " << enabled << " for Trigger '" << GetName() << "'");
        m_enabled = enabled;
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

    uint OdeTrigger::GetSourceId()
    {
        LOG_FUNC();
        
        return m_sourceId;
    }
    
    void OdeTrigger::SetSourceId(uint sourceId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_sourceId = sourceId;
    }

    double OdeTrigger::GetMinConfidence()
    {
        LOG_FUNC();
        
        return m_minConfidence;
    }
    
    void OdeTrigger::SetMinConfidence(double minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minConfidence = minConfidence;
    }
    
    void OdeTrigger::GetMinDimensions(uint* minWidth, uint* minHeight)
    {
        LOG_FUNC();
        
        *minWidth = m_minWidth;
        *minHeight = m_minHeight;
    }

    void OdeTrigger::SetMinDimensions(uint minWidth, uint minHeight)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_minWidth = minWidth;
        m_minHeight = minHeight;
    }
    
    void OdeTrigger::GetMaxDimensions(uint* maxWidth, uint* maxHeight)
    {
        LOG_FUNC();
        
        *maxWidth = m_maxWidth;
        *maxHeight = m_maxHeight;
    }

    void OdeTrigger::SetMaxDimensions(uint maxWidth, uint maxHeight)
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

    void OdeTrigger::PreProcessFrame(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled)
        {
            return;
        }
        // Reset the occurrences from the last frame. 
        m_occurrences = 0;

        for (const auto &imap: m_pOdeAreas)
        {
            // If an Area is set to display, create a rectange and color representation
            DSL_ODE_AREA_PTR pOdeArea = std::dynamic_pointer_cast<OdeArea>(imap.second);
            if (pOdeArea->m_display)
            {
                // If this is the first time seeing a frame for the reported Source Id.
                if (pOdeArea->m_frameNumPerSource.find(pFrameMeta->source_id) == pOdeArea->m_frameNumPerSource.end())
                {
                    // Initial the frame number for the new source
                    pOdeArea->m_frameNumPerSource[pFrameMeta->source_id] = 0;
                }
                
                // If the last frame number for the reported source is less than the current frame
                if (pOdeArea->m_frameNumPerSource[pFrameMeta->source_id] < pFrameMeta->frame_num)
                {
                    // Update the frame number so we only add the rectangle once
                    pOdeArea->m_frameNumPerSource[pFrameMeta->source_id] = pFrameMeta->frame_num;
                    
                    NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(pBuffer);
                    NvDsDisplayMeta* pDisplayMeta = nvds_acquire_display_meta_from_pool(batchMeta);
                    
                    pDisplayMeta->rect_params[pDisplayMeta->num_rects++] = pOdeArea->m_rectParams;
                    nvds_add_display_meta_to_frame(pFrameMeta, pDisplayMeta);
                }
            }
        }
    }

    bool OdeTrigger::checkForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Note: function is called from the system (callback) context
        // Gaurd against property updates from the client API
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        // Ensure enabled, and that the limit has not been exceeded
        if (m_limit and m_triggered >= m_limit) 
        {
            return false;
        }
        // Filter on Class id if set
        if ((m_classId != DSL_ODE_ANY_CLASS) and (m_classId != pObjectMeta->class_id))
        {
            return false;
        }
        // Filter on Source id if set
        if ((m_sourceId != DSL_ODE_ANY_SOURCE) and (m_sourceId != pFrameMeta->source_id))
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
        // If defined, check for maximum dimensions
        if ((m_maxWidth and pObjectMeta->rect_params.width > m_maxWidth) or
            (m_maxHeight and pObjectMeta->rect_params.height > m_maxHeight))
        {
            return false;
        }
        // If define, check if Inference was done on the frame or not
        if (m_inferDoneOnly and !pFrameMeta->bInferDone)
        {
            return false;
        }
        // If areas are defined, check for overlay
        if (m_pOdeAreas.size())
        {
            for (const auto &imap: m_pOdeAreas)
            {
                DSL_ODE_AREA_PTR pOdeArea = std::dynamic_pointer_cast<OdeArea>(imap.second);
                if (doesOverlap(pObjectMeta->rect_params, pOdeArea->m_rectParams))
                {
                    return true;
                }
            }
            return false;
        }
        return true;
    }

    inline bool OdeTrigger::valueInRange(int value, int min, int max)
    { 
        return (value >= min) && (value <= max);
    }

    inline bool OdeTrigger::doesOverlap(NvOSD_RectParams a, NvOSD_RectParams b)
    {
        bool xOverlap = valueInRange(a.left, b.left, b.left + b.width) ||
                        valueInRange(b.left, a.left, a.left + a.width);

        bool yOverlap = valueInRange(a.top, b.top, b.top + b.height) ||
                        valueInRange(b.top, a.top, a.top + a.height);

        return xOverlap && yOverlap;
    }    
    
    // *****************************************************************************

    OccurrenceOdeTrigger::OccurrenceOdeTrigger(const char* name, uint classId, uint limit)
        : OdeTrigger(name, classId, limit)
    {
        LOG_FUNC();
    }

    OccurrenceOdeTrigger::~OccurrenceOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool OccurrenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }

        m_triggered++;
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, pObjectMeta);
        }
        return true;
    }

    // *****************************************************************************
    
    AbsenceOdeTrigger::AbsenceOdeTrigger(const char* name, uint classId, uint limit)
        : OdeTrigger(name, classId, limit)
    {
        LOG_FUNC();
    }

    AbsenceOdeTrigger::~AbsenceOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool AbsenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Important **** we need to check for Criteria even if the Absence Trigger is disabled. This is
        // case another Trigger enables This trigger, and it checks for the number of occurrences in the 
        // PostProcessFrame() . If the m_occurrences is not updated the Trigger will report Absence incorrectly
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }
    
    uint AbsenceOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or
            (m_limit and m_triggered >= m_limit) or 
            m_occurrences) 
        {
            m_occurrences = 0;
            return m_occurrences;
        }        
        
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return m_occurrences;
   }

    // *****************************************************************************
    
    SummationOdeTrigger::SummationOdeTrigger(const char* name, uint classId, uint limit)
        : OdeTrigger(name, classId, limit)
    {
        LOG_FUNC();
    }

    SummationOdeTrigger::~SummationOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool SummationOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint SummationOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled)
        {
            return 0;
        }
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return 1; // Summation ODE is triggered on every frame
   }

    // *****************************************************************************
    
    IntersectionOdeTrigger::IntersectionOdeTrigger(const char* name, uint classId, uint limit)
        : OdeTrigger(name, classId, limit)
    {
        LOG_FUNC();
    }

    IntersectionOdeTrigger::~IntersectionOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool IntersectionOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrenceMetaList.push_back(pObjectMeta);
        
        return true;
    }

    uint IntersectionOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        m_occurrences = 0;
        
        // need at least two objects for intersection to occur
        if (m_enabled and m_occurrenceMetaList.size() > 1)
        {
            // iterate through the list of object occurrences that passed all min criteria
            for (uint i = 0; i < m_occurrenceMetaList.size()-1 ; i++) 
            {
                for (uint j = i+1; j < m_occurrenceMetaList.size() ; j++) 
                {
                    // check each in turn for any frame overlap
                    if (doesOverlap(m_occurrenceMetaList[i]->rect_params, m_occurrenceMetaList[j]->rect_params))
                    {
                        // event has been triggered
                        m_occurrences++;
                        
                        // TODO: should we be testing the new trigger count against the limit here?
                        // or just wait for the next frame and leave "checkForOccurrence" to test the limit?
                        m_triggered++;
                        
                         // update the total event count static variable
                        s_eventCount++;

                        for (const auto &imap: m_pOdeActions)
                        {
                            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                            
                            // Invoke each action twice, once for each object in the tested pair
                            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, m_occurrenceMetaList[i]);
                            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, m_occurrenceMetaList[j]);
                        }
                    }
                }
            }
        }   

        // reset for next frame
        m_occurrenceMetaList.clear();
        return m_occurrences;
   }

    // *****************************************************************************

    CustomOdeTrigger::CustomOdeTrigger(const char* name, 
        uint classId, uint limit, dsl_ode_check_for_occurrence_cb clientChecker, void* clientData)
        : OdeTrigger(name, classId, limit)
        , m_clientChecker(clientChecker)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    CustomOdeTrigger::~CustomOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool CustomOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !checkForMinCriteria(pFrameMeta, pObjectMeta))
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
            LOG_ERROR("Custon ODE Trigger '" << GetName() << "' threw exception calling client callback");
            return false;
        }

        m_triggered++;
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, pObjectMeta);
        }
        return true;
    }

    // *****************************************************************************
    
    MinimumOdeTrigger::MinimumOdeTrigger(const char* name, uint classId, uint limit, uint minimum)
        : OdeTrigger(name, classId, limit)
        , m_minimum(minimum)
    {
        LOG_FUNC();
    }

    MinimumOdeTrigger::~MinimumOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool MinimumOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint MinimumOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or m_occurrences >= m_minimum)
        {
            return 0;
        }
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return m_occurrences;
    }

    // *****************************************************************************
    
    MaximumOdeTrigger::MaximumOdeTrigger(const char* name, uint classId, uint limit, uint maximum)
        : OdeTrigger(name, classId, limit)
        , m_maximum(maximum)
    {
        LOG_FUNC();
    }

    MaximumOdeTrigger::~MaximumOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool MaximumOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint MaximumOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or m_occurrences <= m_maximum)
        {
            return 0;
        }
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return m_occurrences;
   }

    // *****************************************************************************
    
    RangeOdeTrigger::RangeOdeTrigger(const char* name, uint classId, uint limit, uint lower, uint upper)
        : OdeTrigger(name, classId, limit)
        , m_lower(lower)
        , m_upper(upper)
    {
        LOG_FUNC();
    }

    RangeOdeTrigger::~RangeOdeTrigger()
    {
        LOG_FUNC();
    }
    
    bool RangeOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!checkForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint RangeOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or (m_occurrences < m_lower) or (m_occurrences > m_upper))
        {
            return 0;
        }
        // event has been triggered
        m_triggered++;

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pFrameMeta, NULL);
        }
        return m_occurrences;
   }

}