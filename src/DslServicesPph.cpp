/*
The MIT License

Copyright (c)   2021-2022, Prominence AI, Inc.

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
#include "DslPadProbeHandler.h"

namespace DSL
{
    DslReturnType Services::PphCustomNew(const char* name,
        dsl_pph_custom_client_handler_cb clientHandler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("Custom Pad Probe Handler name '" << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPH_CUSTOM_NEW(name, clientHandler, clientData);

            LOG_INFO("New Custom Pad Probe Handler '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Custom Pad Prove handler '" << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphMeterNew(const char* name, uint interval, 
        dsl_pph_meter_client_handler_cb clientHandler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("Meter Pad Probe Handler name '" << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            if (!interval)
            {
                LOG_ERROR("Meter Pad Probe Handler '" << name << "' failed to set property, interval must be greater than 0");
                return DSL_RESULT_PPH_METER_INVALID_INTERVAL;
            }
            m_padProbeHandlers[name] = DSL_PPH_METER_NEW(name, 
                interval, clientHandler, clientData);

            LOG_INFO("New Meter Pad Probe Handler '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Meter Pad Prove handler '" << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    

    DslReturnType Services::PphMeterIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, MeterPadProbeHandler);

            DSL_PPH_METER_PTR pMeter = 
                std::dynamic_pointer_cast<MeterPadProbeHandler>(m_padProbeHandlers[name]);

            *interval = pMeter->GetInterval();

            LOG_INFO("Meter Pad Probe Handler '" << name << "' returned Interval = "
                << *interval << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Meter Sink '" << name << "' threw an exception getting reporting interval");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphMeterIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, MeterPadProbeHandler);
            
            if (!interval)
            {
                LOG_ERROR("Meter Pad Probe Handler '" << name << "' failed to set property, interval must be greater than 0");
                return DSL_RESULT_PPH_METER_INVALID_INTERVAL;
            }

            DSL_PPH_METER_PTR pMeter = 
                std::dynamic_pointer_cast<MeterPadProbeHandler>(m_padProbeHandlers[name]);

            if (!pMeter->SetInterval(interval))
            {
                LOG_ERROR("Meter Pad Probe Handler '" << name << "' failed to set reporting interval");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            LOG_INFO("Meter Pad Probe Handler '" << name << "' set Interval = "
                << interval << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Meter Pad Probe Handler '" << name << "' threw an exception setting reporting interval");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PphOdeNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("ODE Pad Probe Handler name '" << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPH_ODE_NEW(name);
            
            LOG_INFO("New ODE Pad Probe Handler '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Pad Probe Handler '" << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphOdeTriggerAdd(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

            // Can't add Events if they're In use by another Handler
            if (m_odeTriggers[trigger]->IsInUse())
            {
                LOG_ERROR("Unable to add ODE Trigger '" << trigger 
                    << "' as it is currently in use");
                return DSL_RESULT_ODE_TRIGGER_IN_USE;
            }

            DSL_PPH_ODE_PTR pOde = 
                std::dynamic_pointer_cast<OdePadProbeHandler>(m_padProbeHandlers[name]);

            if (!pOde->AddChild(m_odeTriggers[trigger]))
            {
                LOG_ERROR("ODE Pad Probe Handler '" << name
                    << "' failed to add ODE Trigger '" << trigger << "'");
                return DSL_RESULT_PPH_ODE_TRIGGER_ADD_FAILED;
            }
            LOG_INFO("ODE Trigger '" << trigger 
                << "' was added to ODE Pad Probe Handler '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name
                << "' threw exception adding ODE Trigger '" << trigger << "'");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphOdeTriggerRemove(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

            if (!m_odeTriggers[trigger]->IsParent(m_padProbeHandlers[name]))
            {
                LOG_ERROR("ODE Trigger '" << trigger << 
                    "' is not in use by ODE Pad Probe Handler '" << name << "'");
                return DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE;
            }
            
            if (!m_padProbeHandlers[name]->RemoveChild(m_odeTriggers[trigger]))
            {
                LOG_ERROR("ODE Pad Probe Handler '" << name
                    << "' failed to remove ODE Trigger '" << trigger << "'");
                return DSL_RESULT_PPH_ODE_TRIGGER_REMOVE_FAILED;
            }
            LOG_INFO("ODE Trigger '" << trigger 
                << "' was removed from ODE Pad Probe Handler '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name 
                << "' threw an exception removing ODE Trigger");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PphOdeTriggerRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            
            m_padProbeHandlers[name]->RemoveAllChildren();

            LOG_INFO("All ODE Triggers removed from ODE Pad Probe Handler '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name 
                << "' threw an exception removing All ODE Triggers");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphOdeDisplayMetaAllocSizeGet(const char* name, uint* size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);

            DSL_PPH_ODE_PTR pOde = 
                std::dynamic_pointer_cast<OdePadProbeHandler>(m_padProbeHandlers[name]);
            
            *size = pOde->GetDisplayMetaAllocSize();

            LOG_INFO("ODE Pad Probe Handler '" << name 
                << "' returned a Display Meta size of " << *size << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name 
                << "' threw an exception getting Display Meta size");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphOdeDisplayMetaAllocSizeSet(const char* name, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            
            DSL_PPH_ODE_PTR pOde = 
                std::dynamic_pointer_cast<OdePadProbeHandler>(m_padProbeHandlers[name]); 

            pOde->SetDisplayMetaAllocSize(size);

            LOG_INFO("ODE Pad Probe Handler '" << name 
                << "' set its Display Meta size to " << size << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name 
                << "' threw an exception setting Display Meta size");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphBufferTimeoutNew(const char* name,
        uint timeout, dsl_pph_buffer_timeout_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("Buffer Timeout Pad Probe Handler name '" 
                    << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPH_BUFFER_TIMEOUR_NEW(name,
                timeout, handler, clientData);
            
            LOG_INFO("New Buffer Timeout Pad Probe Handler '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Buffer Timeout Pad Probe Handler '" 
                << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphEosNew(const char* name,
        dsl_pph_eos_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("End of Stream Pad Probe Handler name '" 
                    << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPEH_EOS_HANDLER_NEW(name,
                handler, clientData);
            
            LOG_INFO("End of Stream Pad Probe Handler '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New End of Stream Pad Probe Handler '" 
                << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);

            *enabled = m_padProbeHandlers[name]->GetEnabled();

            LOG_INFO("Pad Probe Handler '" << name << "' returned Enabled = "
                << *enabled << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler '" << name
                << "' threw exception getting the Enabled state");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

   DslReturnType Services::PphEnabledSet(const char* name, boolean enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);

            if (!m_padProbeHandlers[name]->SetEnabled(enabled))
            {
                LOG_ERROR("Pad Probe Handler '" << name
                    << "' failed to set enabled state");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << name << "' set Enabled = "
                << enabled << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler '" << name
                << "' threw exception setting the Enabled state");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            
            if (m_padProbeHandlers[name]->IsInUse())
            {
                LOG_INFO("Pad Probe Handler '" << name << "' is in use");
                return DSL_RESULT_PPH_IS_IN_USE;
            }
            m_padProbeHandlers.erase(name);

            LOG_INFO("Pad Probe Handler '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler '" << name << "' threw an exception on deletion");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PphDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_padProbeHandlers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_padProbeHandlers)
            {
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("Pad Probe Handler '" << imap.second->GetName() 
                        << "' is currently in use");
                    return DSL_RESULT_PPH_IS_IN_USE;
                }
            }
            m_padProbeHandlers.clear();

            LOG_INFO("All Pad Probe Handlers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler threw an exception on delete all");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    uint Services::PphListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_padProbeHandlers.size();
    }

}