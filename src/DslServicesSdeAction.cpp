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
#include "DslSdeAction.h"

namespace DSL
{
    DslReturnType Services::SdeActionPrintNew(const char* name,
        boolean forceFlush)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure action name uniqueness 
            if (m_sdeActions.find(name) != m_sdeActions.end())
            {   
                LOG_ERROR("SDE Action name '" << name << "' is not unique");
                return DSL_RESULT_SDE_ACTION_NAME_NOT_UNIQUE;
            }
            m_sdeActions[name] = DSL_SDE_ACTION_PRINT_NEW(name, forceFlush);

            LOG_INFO("New SDE Print Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New SDE Print Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeActionMonitorNew(const char* name,
        dsl_sde_monitor_occurrence_cb clientMonitor, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure action name uniqueness 
            if (m_sdeActions.find(name) != m_sdeActions.end())
            {   
                LOG_ERROR("SDE Action name '" << name << "' is not unique");
                return DSL_RESULT_SDE_ACTION_NAME_NOT_UNIQUE;
            }
            m_sdeActions[name] = DSL_SDE_ACTION_MONITOR_NEW(name, 
                clientMonitor, clientData);

            LOG_INFO("New SDE Monitor Action '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New SDE Monitor Action '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeActionEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, name);
            
            DSL_SDE_ACTION_PTR pSdeAction = 
                std::dynamic_pointer_cast<SdeAction>(m_sdeActions[name]);
         
            *enabled = pSdeAction->GetEnabled();

            LOG_INFO("SDE Action '" << name << "' returned Enabed = " 
                << *enabled  << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Action '" << name 
                << "' threw exception getting Enabled setting");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeActionEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, name);
            
            DSL_SDE_ACTION_PTR pSdeAction = 
                std::dynamic_pointer_cast<SdeAction>(m_sdeActions[name]);
         
            pSdeAction->SetEnabled(enabled);

            LOG_INFO("SDE Action '" << name << "' set Enabed = " 
                << enabled  << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Action '" << name << "' threw exception setting Enabled");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SdeActionEnabledStateChangeListenerAdd(const char* name,
        dsl_enabled_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, name);
            
            DSL_SDE_ACTION_PTR pSdeAction = 
                std::dynamic_pointer_cast<SdeAction>(m_sdeActions[name]);
         
            if (!pSdeAction->AddEnabledStateChangeListener(listener, clientData))
            {
                LOG_ERROR("SDE Action '" << name 
                    << "' failed to add an Enabled State Change Listener");
                return DSL_RESULT_SDE_ACTION_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("SDE Action '" << name 
                << "' successfully added an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Action '" << name 
                << "' threw exception adding an Enabled State Change Listener");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SdeActionEnabledStateChangeListenerRemove(
        const char* name, dsl_enabled_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, name);
            
            DSL_SDE_ACTION_PTR pSdeAction = 
                std::dynamic_pointer_cast<SdeAction>(m_sdeActions[name]);
         
            if (!pSdeAction->RemoveEnabledStateChangeListener(listener))
            {
                LOG_ERROR("SDE Action '" << name 
                    << "' failed to remove an Enabled State Change Listener");
                return DSL_RESULT_SDE_ACTION_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("SDE Action '" << name 
                << "' successfully removed an Enabled State Change Listener");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Action '" << name 
                << "' threw exception removing an Enabled State Change Listener");
            return DSL_RESULT_SDE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SdeActionDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_SDE_ACTION_NAME_NOT_FOUND(m_sdeActions, name);
            
            if (m_sdeActions[name].use_count() > 1)
            {
                LOG_INFO("SDE Action'" << name << "' is in use");
                return DSL_RESULT_SDE_ACTION_IN_USE;
            }
            m_sdeActions.erase(name);

            LOG_INFO("SDE Action '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Action '" << name << "' threw exception on deletion");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SdeActionDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_sdeActions.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_sdeActions)
            {
                // In the case of Delete all
                if (imap.second.use_count() > 1)
                {
                    LOG_ERROR("SDE Action '" << imap.second->GetName() 
                        << "' is currently in use");
                    return DSL_RESULT_SDE_ACTION_IN_USE;
                }
            }
            m_sdeActions.clear();

            LOG_INFO("All SDE Actions deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("SDE Action threw exception on delete all");
            return DSL_RESULT_SDE_ACTION_THREW_EXCEPTION;
        }
    }

    uint Services::SdeActionListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_sdeActions.size();
    }
}