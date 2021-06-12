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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"
#include "DslOdeArea.h"
#include "DslServices.h"

namespace DSL
{

    // Initialize static Event Counter
    uint64_t OdeTrigger::s_eventCount = 0;

    OdeTrigger::OdeTrigger(const char* name, const char* source, 
        uint classId, uint limit)
        : Base(name)
        , m_wName(m_name.begin(), m_name.end())
        , m_enabled(true)
        , m_source(source)
        , m_sourceId(-1)
        , m_classId(classId)
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
        , m_resetTimeout(0)
        , m_resetTimerId(0)
    {
        LOG_FUNC();

        g_mutex_init(&m_propertyMutex);
        g_mutex_init(&m_resetTimerMutex);
    }

    OdeTrigger::~OdeTrigger()
    {
        LOG_FUNC();
        
        RemoveAllActions();
        RemoveAllAreas();
        
        if (m_resetTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_resetTimerMutex);
            g_source_remove(m_resetTimerId);
        }
        g_mutex_clear(&m_resetTimerMutex);
        g_mutex_clear(&m_propertyMutex);
    }
    
    bool OdeTrigger::AddAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeActions.find(pChild->GetName()) != m_pOdeActions.end())
        {
            LOG_ERROR("ODE Area '" << pChild->GetName() 
                << "' is already a child of ODE Trigger'" << GetName() << "'");
            return false;
        }
        m_pOdeActions[pChild->GetName()] = pChild;
        pChild->AssignParentName(GetName());
        return true;
    }

    bool OdeTrigger::RemoveAction(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeActions.find(pChild->GetName()) == m_pOdeActions.end())
        {
            LOG_WARN("'" << pChild->GetName() 
                <<"' is not a child of ODE Trigger '" << GetName() << "'");
            return false;
        }
        m_pOdeActions.erase(pChild->GetName());
        pChild->ClearParentName();
        return true;
    }
    
    void OdeTrigger::RemoveAllActions()
    {
        LOG_FUNC();
        
        for (auto &imap: m_pOdeActions)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() 
                <<"' from Parent '" << GetName() << "'");
            imap.second->ClearParentName();
        }
        m_pOdeActions.clear();
    }
    
    bool OdeTrigger::AddArea(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeAreas.find(pChild->GetName()) != m_pOdeAreas.end())
        {
            LOG_ERROR("ODE Area '" << pChild->GetName() 
                << "' is already a child of ODE Trigger'" << GetName() << "'");
            return false;
        }
        m_pOdeAreas[pChild->GetName()] = pChild;
        pChild->AssignParentName(GetName());
        return true;
    }

    bool OdeTrigger::RemoveArea(DSL_BASE_PTR pChild)
    {
        LOG_FUNC();
        
        if (m_pOdeAreas.find(pChild->GetName()) == m_pOdeAreas.end())
        {
            LOG_WARN("'" << pChild->GetName() 
                <<"' is not a child of ODE Trigger '" << GetName() << "'");
            return false;
        }
        m_pOdeAreas.erase(pChild->GetName());
        pChild->ClearParentName();
        return true;
    }
    
    void OdeTrigger::RemoveAllAreas()
    {
        LOG_FUNC();
        
        for (auto &imap: m_pOdeAreas)
        {
            LOG_DEBUG("Removing Action '" << imap.second->GetName() 
                <<"' from Parent '" << GetName() << "'");
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
    
    void OdeTrigger::IncrementAndCheckTriggerCount()
    {
        LOG_FUNC();
        // internal do not lock m_propertyMutex
        
        m_triggered++;
        
        if (m_triggered >= m_limit and m_resetTimeout)
        {
            m_resetTimerId = g_timeout_add(1000*m_resetTimeout, 
                TriggerResetTimeoutHandler, this);            
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
        // timeout value = 0 = disable, then kill the timer.
        if (m_resetTimerId and !timeout)
        {
            g_source_remove(m_resetTimerId);
            m_resetTimerId == 0;
        }
        
        m_resetTimeout = timeout;
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

    uint OdeTrigger::GetLimit()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        return m_limit;
    }
    
    void OdeTrigger::SetLimit(uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_limit = limit;
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

    bool OdeTrigger::CheckForSourceId(int sourceId)
    {
        LOG_FUNC();

        // Filter on Source id if set
        if (m_source.size())
        {
            // a "one-time-get" of the source Id from the source name as the 
            // source id is not assigned until the pipeline is played
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

    void OdeTrigger::PreProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta)
    {
        // Reset the occurrences from the last frame, even if disabled  
        m_occurrences = 0;

        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id))
        {
            return;
        }

        // Call on each of the Trigger's Areas to (optionally) display their Rectangle
        for (const auto &imap: m_pOdeAreas)
        {
            DSL_ODE_AREA_PTR pOdeArea = std::dynamic_pointer_cast<OdeArea>(imap.second);
            
            pOdeArea->AddMeta(pDisplayMeta, pFrameMeta);
        }
    }

    bool OdeTrigger::CheckForMinCriteria(NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
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
        if (m_source.size())
        {
            if (m_sourceId == -1)
            {
                Services::GetServices()->SourceIdGet(m_source.c_str(), &m_sourceId);
            }
            if (m_sourceId != pFrameMeta->source_id)
            {
                return false;
            }
        }
        // Ensure that the minimum confidence has been reached
        if (pObjectMeta->confidence > 0 and pObjectMeta->confidence < m_minConfidence)
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
        // If areas are defined, check condition
        if (m_pOdeAreas.size())
        {
            for (const auto &imap: m_pOdeAreas)
            {
                DSL_ODE_AREA_PTR pOdeArea = std::dynamic_pointer_cast<OdeArea>(imap.second);
                if (pOdeArea->CheckForWithin(pObjectMeta->rect_params))
                {
                    return true;
                }
            }
            return false;
        }
        return true;
    }
    
    // *****************************************************************************
    AlwaysOdeTrigger::AlwaysOdeTrigger(const char* name, const char* source, uint when)
        : OdeTrigger(name, source, DSL_ODE_ANY_CLASS, 0)
        , m_when(when)
    {
        LOG_FUNC();
    }

    AlwaysOdeTrigger::~AlwaysOdeTrigger()
    {
        LOG_FUNC();
    }
    
    void AlwaysOdeTrigger::PreProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta)
    {

        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            m_when != DSL_ODE_PRE_OCCURRENCE_CHECK)
        {
            return;
        }
        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
    }

    uint AlwaysOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            m_when != DSL_ODE_POST_OCCURRENCE_CHECK)
        {
            return 0;
        }
        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
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
    
    bool OccurrenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) 
            or !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }

        IncrementAndCheckTriggerCount();
        m_occurrences++;
        
        // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            try
            {
                pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, 
                    pDisplayMeta, pFrameMeta, pObjectMeta);
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
    
    bool AbsenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Important **** we need to check for Criteria even if the Absence Trigger is disabled. 
        // This is case another Trigger enables This trigger, and it checks for the number of 
        // occurrences in the PostProcessFrame() . If the m_occurrences is not updated the Trigger 
        // will report Absence incorrectly
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }
    
    uint AbsenceOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or (m_limit and m_triggered >= m_limit) or m_occurrences) 
        {
            return 0;
        }        
        
        // event has been triggered 
        IncrementAndCheckTriggerCount();

        // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
        return 1;
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
    
    bool InstanceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }

        std::string sourceAndClassId = std::to_string(pFrameMeta->source_id) + "_" 
            + std::to_string(pObjectMeta->class_id);
            
        // If this is the first time seeing an object of "class_id" for "source_id".
        if (m_instances.find(sourceAndClassId) == m_instances.end())
        {
            // Initial the frame number for the new source
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

            for (const auto &imap: m_pOdeActions)
            {
                DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                try
                {
                    pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, 
                        pDisplayMeta, pFrameMeta, pObjectMeta);
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
    
    bool SummationOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint SummationOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or (m_limit and m_triggered >= m_limit))
        {
            return 0;
        }
        // event has been triggered
        IncrementAndCheckTriggerCount();

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
        return 1; // Summation ODE is triggered on every frame
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
    
    bool CustomOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // conditional execution
        if (!m_enabled or !m_clientChecker or !CheckForSourceId(pFrameMeta->source_id) 
            or !CheckForMinCriteria(pFrameMeta, pObjectMeta))
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

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, pObjectMeta);
        }
        return true;
    }
    
    uint CustomOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
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

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
        return 1;
    }

    // *****************************************************************************
    
    PersistenceOdeTrigger::PersistenceOdeTrigger(const char* name, const char* source, 
        uint classId, uint limit, uint minimum, uint maximum)
        : OdeTrigger(name, source, classId, limit)
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
    
    bool PersistenceOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }

		// if this is the first occurrence of any object for this source
		if (m_trackedObjectsPerSource.find(pFrameMeta->source_id) == 
			m_trackedObjectsPerSource.end())
		{
			LOG_DEBUG("First object detected with id = " << pObjectMeta->object_id 
				<< " for source = " << pFrameMeta->source_id);
			
			// create a new tracked object for this tracking Id and source
			std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
				(new TrackedObject(pObjectMeta->object_id, pFrameMeta->frame_num));
				
			// create a map of tracked objects for this source	
			std::shared_ptr<TrackedObjects> pTrackedObjects = 
				std::shared_ptr<TrackedObjects>(new TrackedObjects());
				
			// insert the new tracked object into the new map	
			pTrackedObjects->insert(std::pair<uint64_t, 
				std::shared_ptr<TrackedObject>>(pObjectMeta->object_id, pTrackedObject));
				
			// add the map of tracked objects for this source to the map of all tracked objects.
			m_trackedObjectsPerSource[pFrameMeta->source_id] = pTrackedObjects;
		}
		else
		{
			std::shared_ptr<TrackedObjects> pTrackedObjects = 
				m_trackedObjectsPerSource[pFrameMeta->source_id];
				
			// else, if this is the first occurrence of a specific object for this source
			if (pTrackedObjects->find(pObjectMeta->object_id) == pTrackedObjects->end())
			{
				LOG_DEBUG("New object detected with id = " << pObjectMeta->object_id 
					<< " for source = " << pFrameMeta->source_id);
				
				// create a new tracked object for this tracking Id and source
				std::shared_ptr<TrackedObject> pTrackedObject = std::shared_ptr<TrackedObject>
					(new TrackedObject(pObjectMeta->object_id, pFrameMeta->frame_num));

				// insert the new tracked object into the new map	
				pTrackedObjects->insert(std::pair<uint64_t, 
					std::shared_ptr<TrackedObject>>(pObjectMeta->object_id, pTrackedObject));		
			}
			else
			{
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
                
				LOG_DEBUG("Tracked objected detected with id = " << pObjectMeta->object_id 
					<< " for source = " << pFrameMeta->source_id);
				// else, the object is currently being tracked - so update the frame number
				pTrackedObjects->at(pObjectMeta->object_id)->m_frameNumber = pFrameMeta->frame_num;
				
				timeval currentTime;
				gettimeofday(&currentTime, NULL);
				
				double currentTimeMs = currentTime.tv_sec*1000.0 + currentTime.tv_usec/1000.0;
				double trackedTimeMs = currentTimeMs - pTrackedObjects->at(pObjectMeta->object_id)->m_creationTimeMs;
				
				LOG_DEBUG("Persistence for tracked object with id = " << pObjectMeta->object_id 
					<< " for source = " << pFrameMeta->source_id << ", = " << trackedTimeMs << " ms");
				
				// if the objects tracked time is within range. 
				if (trackedTimeMs >= m_minimumMs and trackedTimeMs <= m_maximumMs)
				{
					// event has been triggered
					IncrementAndCheckTriggerCount();
					m_occurrences++;

					// update the total event count static variable
					s_eventCount++;
		
					for (const auto &imap: m_pOdeActions)
					{
						DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
						pOdeAction->HandleOccurrence(shared_from_this(), 
							pBuffer, pDisplayMeta, pFrameMeta, pObjectMeta);
					}
				}
			}
		}
		return true;
		
    }

    uint PersistenceOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        if (m_trackedObjectsPerSource.empty())
        {
            return 0;
        }
		
		// purge all tracked objects, for all sources that are not in the current frame.
		for (const auto &trackedObjects: m_trackedObjectsPerSource)
		{
			std::shared_ptr<TrackedObjects> pTrackedObjects = trackedObjects.second;

			auto trackedObject = pTrackedObjects->cbegin();
			while (trackedObject != pTrackedObjects->cend())
			{
				if (trackedObject->second->m_frameNumber != pFrameMeta->frame_num)
				{
					LOG_DEBUG("Purging tracked object with id = " << trackedObject->first 
						<< " for source = " << trackedObjects.first);
						
					// use the return value to update the iterator, as erase invalidates it
					trackedObject = pTrackedObjects->erase(trackedObject);
				}
				else {
					++trackedObject;
				}			
			}
		}
        return m_occurrences;
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
    
    bool CountOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint CountOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

        if (!m_enabled or (m_occurrences < m_minimum) or (m_occurrences > m_maximum))
        {
            return 0;
        }
        // event has been triggered
        IncrementAndCheckTriggerCount();

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
        return m_occurrences;
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
    
    bool SmallestOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrenceMetaList.push_back(pObjectMeta);
        
        return true;
    }

    uint SmallestOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
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
            NvDsObjectMeta* smallestObject(NULL);
            
            // iterate through the list of object occurrences that passed all min criteria
            for (const auto &ivec: m_occurrenceMetaList) 
            {
                uint rectArea = ivec->rect_params.width * ivec->rect_params.width;
                if (rectArea < smallestArea) 
                { 
                    smallestArea = rectArea;
                    smallestObject = ivec;    
                }
            }
            for (const auto &imap: m_pOdeActions)
            {
                DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                
                pOdeAction->HandleOccurrence(shared_from_this(), 
                    pBuffer, pDisplayMeta, pFrameMeta, smallestObject);
            }
        }   

        // reset for next frame
        m_occurrenceMetaList.clear();
        return m_occurrences;
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
    
    bool LargestOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrenceMetaList.push_back(pObjectMeta);
        
        return true;
    }

    uint LargestOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
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
            NvDsObjectMeta* largestObject(NULL);
            
            // iterate through the list of object occurrences that passed all min criteria
            for (const auto &ivec: m_occurrenceMetaList) 
            {
                uint rectArea = ivec->rect_params.width * ivec->rect_params.width;
                if (rectArea > largestArea) 
                { 
                    largestArea = rectArea;
                    largestObject = ivec;    
                }
            }
            for (const auto &imap: m_pOdeActions)
            {
                DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                
                pOdeAction->HandleOccurrence(shared_from_this(), 
                    pBuffer, pDisplayMeta, pFrameMeta, largestObject);
            }
        }   

        // reset for next frame
        m_occurrenceMetaList.clear();
        return m_occurrences;
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
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_triggered = 0;
        m_currentLow = m_preset;
    }
    
    bool NewLowOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint NewLowOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or m_occurrences >= m_currentLow)
        {
            return 0;
        }
        // new low
        m_currentLow = m_occurrences;
        
        // event has been triggered
        IncrementAndCheckTriggerCount();

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
        return 1; // At most once per frame
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
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);
        
        m_triggered = 0;
        m_currentHigh = m_preset;
    }
    
    bool NewHighOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id) or 
            !CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            return false;
        }
        
        m_occurrences++;
        
        return true;
    }

    uint NewHighOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        if (!m_enabled or m_occurrences <= m_currentHigh)
        {
            return 0;
        }
        // new high
        m_currentHigh = m_occurrences;
        
        // event has been triggered
        IncrementAndCheckTriggerCount();

         // update the total event count static variable
        s_eventCount++;

        for (const auto &imap: m_pOdeActions)
        {
            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
            pOdeAction->HandleOccurrence(shared_from_this(), 
                pBuffer, pDisplayMeta, pFrameMeta, NULL);
        }
        return 1; // At most once per frame
   }
   
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
    
    bool ABOdeTrigger::CheckForOccurrence(GstBuffer* pBuffer, NvDsDisplayMeta* pDisplayMeta, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled or !CheckForSourceId(pFrameMeta->source_id))
        {
            return false;
        }
        
        bool occurrenceAdded(false);
        
        m_classId = m_classIdA;
        if (CheckForMinCriteria(pFrameMeta, pObjectMeta))
        {
            m_occurrenceMetaListA.push_back(pObjectMeta);
            occurrenceAdded = true;
        }
        else if (!m_classIdAOnly)
        {
            m_classId = m_classIdB;
            if (CheckForMinCriteria(pFrameMeta, pObjectMeta))
            {
                m_occurrenceMetaListB.push_back(pObjectMeta);
                occurrenceAdded = true;
            }
        }
        
        return occurrenceAdded;
    }

    uint ABOdeTrigger::PostProcessFrame(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        if (m_classIdAOnly)
        {
            return PostProcessFrameA(pBuffer, pDisplayMeta, pFrameMeta);
        }
        return  PostProcessFrameAB(pBuffer, pDisplayMeta, pFrameMeta);
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
        
        LOG_WARN("min = " << m_minimum << ", max = " << m_maximum);
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
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
        m_occurrences = 0;
        
        // need at least two objects for intersection to occur
        if (m_enabled and m_occurrenceMetaListA.size() > 1)
        {
            // iterate through the list of object occurrences that passed all min criteria
            for (uint i = 0; i < m_occurrenceMetaListA.size()-1 ; i++) 
            {
                for (uint j = i+1; j < m_occurrenceMetaListA.size() ; j++) 
                {
                    if (CheckDistance(m_occurrenceMetaListA[i], m_occurrenceMetaListA[j]))
                    {
                        // event has been triggered
                        m_occurrences++;
                        IncrementAndCheckTriggerCount();
                        
                         // update the total event count static variable
                        s_eventCount++;

                        for (const auto &imap: m_pOdeActions)
                        {
                            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                            
                            // Invoke each action twice, once for each object in the tested pair
                            pOdeAction->HandleOccurrence(shared_from_this(), 
                                pBuffer, pDisplayMeta, pFrameMeta, m_occurrenceMetaListA[i]);
                            pOdeAction->HandleOccurrence(shared_from_this(), 
                                pBuffer, pDisplayMeta, pFrameMeta, m_occurrenceMetaListA[j]);
                        }
                        if (m_limit and m_triggered >= m_limit)
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
        return m_occurrences;
    }
   
    uint DistanceOdeTrigger::PostProcessFrameAB(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
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
                        if (CheckDistance(iterA, iterB))
                        {
                            LOG_WARN("min = " << m_minimum << ", max = " << m_maximum);
                            // event has been triggered
                            m_occurrences++;
                            IncrementAndCheckTriggerCount();
                            
                             // update the total event count static variable
                            s_eventCount++;

                            for (const auto &imap: m_pOdeActions)
                            {
                                DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                                
                                // Invoke each action twice, once for each object in the tested pair
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, pDisplayMeta, pFrameMeta, iterA);
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, pDisplayMeta, pFrameMeta, iterB);
                            }
                            if (m_limit and m_triggered >= m_limit)
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
        return m_occurrences;
    }

    bool DistanceOdeTrigger::CheckDistance(NvDsObjectMeta* pObjectMetaA, NvDsObjectMeta* pObjectMetaB)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_propertyMutex);

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
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
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

                        for (const auto &imap: m_pOdeActions)
                        {
                            DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                            
                            // Invoke each action twice, once for each object in the tested pair
                            pOdeAction->HandleOccurrence(shared_from_this(), 
                                pBuffer, pDisplayMeta, pFrameMeta, m_occurrenceMetaListA[i]);
                            pOdeAction->HandleOccurrence(shared_from_this(), 
                                pBuffer, pDisplayMeta, pFrameMeta, m_occurrenceMetaListA[j]);
                        }
                        if (m_limit and m_triggered >= m_limit)
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
        return m_occurrences;
   }

    uint IntersectionOdeTrigger::PostProcessFrameAB(GstBuffer* pBuffer, 
        NvDsDisplayMeta* pDisplayMeta,  NvDsFrameMeta* pFrameMeta)
    {
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

                            for (const auto &imap: m_pOdeActions)
                            {
                                DSL_ODE_ACTION_PTR pOdeAction = std::dynamic_pointer_cast<OdeAction>(imap.second);
                                
                                // Invoke each action twice, once for each object in the tested pair
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, pDisplayMeta, pFrameMeta, iterA);
                                pOdeAction->HandleOccurrence(shared_from_this(), 
                                    pBuffer, pDisplayMeta, pFrameMeta, iterB);
                            }
                            if (m_limit and m_triggered >= m_limit)
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
        return m_occurrences;
    }
}