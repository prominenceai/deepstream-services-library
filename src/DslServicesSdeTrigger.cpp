/*
The MIT License

Copyright (c)   2024, Prominence AI, Inc.

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
#include "DslApi.h"
#include "DslServices.h"
#include "DslServicesValidate.h"

namespace DSL
{
   
    DslReturnType Services::SdeTriggerOccurrenceNew(const char* name, 
        const char* source, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_sdeTriggers.find(name) != m_sdeTriggers.end())
            {   
                LOG_ERROR("SDE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_SDE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_sdeTriggers[name] = DSL_SDE_TRIGGER_OCCURRENCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Occurrence SDE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Occurrence SDE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeTriggerReset(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->Reset();
            
            LOG_INFO("SDE Trigger '" << name << "' Reset successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerResetTimeoutGet(const char* name, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *timeout = pSdeTrigger->GetResetTimeout();
            
            LOG_INFO("Trigger '" << name << "' returned Timeout = " 
                << *timeout << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception getting Reset Timer");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerResetTimeoutSet(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->SetResetTimeout(timeout);

            LOG_INFO("Trigger '" << name << "' set Timeout = " 
                << timeout << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception setting Reset Timer");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerLimitStateChangeListenerAdd(const char* name,
        dsl_trigger_limit_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            if (!pSdeTrigger->AddLimitStateChangeListener(listener, clientData))
            {
                LOG_ERROR("SDE Trigger '" << name 
                    << "' failed to add a Limit State Change Listener");
                return DSL_RESULT_SDE_TRIGGER_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("SDE Trigger '" << name 
                << "' successfully added a Limit State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception adding a Limit State Change Listener");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SdeTriggerLimitStateChangeListenerRemove(const char* name,
        dsl_trigger_limit_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            if (!pSdeTrigger->RemoveLimitStateChangeListener(listener))
            {
                LOG_ERROR("SDE Trigger '" << name 
                    << "' failed to remove a Limit State Change Listener");
                return DSL_RESULT_SDE_TRIGGER_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("SDE Trigger '" << name 
                << "' successfully removed a Limit State Change Listener");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception removing a Limit State Change Listener");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeTriggerEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *enabled = pSdeTrigger->GetEnabled();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->SetEnabled(enabled);
            
            LOG_INFO("Trigger '" << name << "' returned Enabled = "
                << enabled << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception setting Enabled");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerEnabledStateChangeListenerAdd(const char* name,
        dsl_enabled_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            if (!pSdeTrigger->AddEnabledStateChangeListener(listener, clientData))
            {
                LOG_ERROR("SDE Trigger '" << name 
                    << "' failed to add an Enabled State Change Listener");
                return DSL_RESULT_SDE_TRIGGER_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("SDE Trigger '" << name 
                << "' successfully added an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception adding an Enabled State Change Listener");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SdeTriggerEnabledStateChangeListenerRemove(const char* name,
        dsl_enabled_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            if (!pSdeTrigger->RemoveEnabledStateChangeListener(listener))
            {
                LOG_ERROR("SDE Trigger '" << name 
                    << "' failed to remove an Enabled State Change Listener");
                return DSL_RESULT_SDE_TRIGGER_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("SDE Trigger '" << name 
                << "' successfully removed an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception removing an Enabled State Change Listener");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeTriggerSourceGet(const char* name, const char** source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *source = pSdeTrigger->GetSource();
            
            LOG_INFO("Trigger '" << name << "' returned Source = " 
                << *source << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting source name");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerSourceSet(const char* name, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);

            pSdeTrigger->SetSource(source);
            
            LOG_INFO("Trigger '" << name << "' set Source = " 
                << source << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception setting source name");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerInferGet(const char* name, const char** infer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *infer = pSdeTrigger->GetInfer();
            
            LOG_INFO("Trigger '" << name << "' returned inference component name = " 
                << *infer << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting inference component name");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerInferSet(const char* name, const char* infer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);

            pSdeTrigger->SetInfer(infer);
            
            LOG_INFO("Trigger '" << name << "' set inference component name = " 
                << infer << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting inference component name");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerClassIdGet(const char* name, uint* classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *classId = pSdeTrigger->GetClassId();
            
            LOG_INFO("Trigger '" << name << "' returned Class Id = " 
                << *classId << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting class id");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerClassIdSet(const char* name, uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->SetClassId(classId);
            
            LOG_INFO("Trigger '" << name << "' set Class Id = " 
                << classId << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting class id");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerLimitEventGet(const char* name, uint* limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *limit = pSdeTrigger->GetEventLimit();

            LOG_INFO("Trigger '" << name << "' returned Event Limit = " 
                << *limit << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting Event Limit");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerLimitEventSet(const char* name, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->SetEventLimit(limit);
            
            LOG_INFO("Trigger '" << name << "' set Evemt Limit = " 
                << limit << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting Event Limit");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }    
            
    DslReturnType Services::SdeTriggerLimitFrameGet(const char* name, uint* limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *limit = pSdeTrigger->GetFrameLimit();

            LOG_INFO("Trigger '" << name << "' returned Frame Limit = " 
                << *limit << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting Frame Limit");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerLimitFrameSet(const char* name, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->SetFrameLimit(limit);
            
            LOG_INFO("Trigger '" << name << "' set Frame Limit = " 
                << limit << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception getting limit");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }    
            
    DslReturnType Services::SdeTriggerConfidenceMinGet(const char* 
        name, float* minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *minConfidence = pSdeTrigger->GetMinConfidence();
            
            LOG_INFO("Trigger '" << name << "' returned minimum confidence = " 
                << *minConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting minimum confidence");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerConfidenceMinSet(const char* name, 
        float minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);

            pSdeTrigger->SetMinConfidence(minConfidence);

            LOG_INFO("Trigger '" << name << "' set minimum confidence = " 
                << minConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting minimum confidence");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerConfidenceMaxGet(const char* 
        name, float* maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *maxConfidence = pSdeTrigger->GetMaxConfidence();
            
            LOG_INFO("Trigger '" << name << "' returned maximum confidence = " 
                << *maxConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting maximum confidence");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerConfidenceMaxSet(const char* name, 
        float maxConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);

            pSdeTrigger->SetMaxConfidence(maxConfidence);

            LOG_INFO("Trigger '" << name << "' set maximum confidence = " 
                << maxConfidence << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw exception getting minimum confidence");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeTriggerIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            *interval = pSdeTrigger->GetInterval();
            
            LOG_INFO("Trigger '" << name << "' returned Interval = " 
                << *interval << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception getting Interval");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::SdeTriggerIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            DSL_SDE_TRIGGER_PTR pSdeTrigger = 
                std::dynamic_pointer_cast<SdeTrigger>(m_sdeTriggers[name]);
         
            pSdeTrigger->SetInterval(interval);

            LOG_INFO("Trigger '" << name << "' set Interval = " 
                << interval << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw exception setting Interval");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::SdeTriggerActionAdd(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, action);

            // Note: Actions can be added when in use, i.e. shared between
            // multiple SDE Triggers

            if (!m_sdeTriggers[name]->AddAction(m_sdeActions[action]))
            {
                LOG_ERROR("SDE Trigger '" << name
                    << "' failed to add SDE Action '" << action << "'");
                return DSL_RESULT_SDE_TRIGGER_ACTION_ADD_FAILED;
            }
            LOG_INFO("SDE Action '" << action
                << "' was added to SDE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name
                << "' threw exception adding SDE Action '" << action << "'");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SdeTriggerActionRemove(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, action);

            if (!m_sdeActions[action]->IsParent(m_sdeTriggers[name]))
            {
                LOG_ERROR("SDE Action'" << action << 
                    "' is not in use by SDE Trigger '" << name << "'");
                return DSL_RESULT_SDE_TRIGGER_ACTION_NOT_IN_USE;
            }

            if (!m_sdeTriggers[name]->RemoveAction(m_sdeActions[action]))
            {
                LOG_ERROR("SDE Trigger '" << name
                    << "' failed to remove SDE Action '" << action << "'");
                return DSL_RESULT_SDE_TRIGGER_ACTION_REMOVE_FAILED;
            }
            LOG_INFO("SDE Action '" << action
                << "' was removed from SDE Trigger '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name
                << "' threw exception remove SDE Action '" << action << "'");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SdeTriggerActionRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);

            m_sdeTriggers[name]->RemoveAllActions();

            LOG_INFO("All Events Actions removed from SDE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name 
                << "' threw an exception removing All Events Actions");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeTriggerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_TRIGGER_NAME_NOT_FOUND(m_sdeTriggers, name);
            
            if (m_sdeTriggers[name]->IsInUse())
            {
                LOG_INFO("SDE Trigger '" << name << "' is in use");
                return DSL_RESULT_SDE_TRIGGER_IN_USE;
            }
            m_sdeTriggers.erase(name);

            LOG_INFO("SDE Trigger '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger '" << name << "' threw an exception on deletion");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeTriggerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_sdeTriggers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_sdeTriggers)
            {
                // In the case of Delete all
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("SDE Trigger '" << imap.second->GetName() << "' is currently in use");
                    return DSL_RESULT_SDE_TRIGGER_IN_USE;
                }
            }
            m_sdeTriggers.clear();

            LOG_INFO("All SDE Triggers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Trigger threw an exception on delete all");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }

    uint Services::SdeTriggerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_sdeTriggers.size();
    }

}