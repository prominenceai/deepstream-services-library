/*
The MIT License

Copyright (c)   2021-2024, Prominence AI, Inc.

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
    DslReturnType Services::ComponentDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        if (m_components[name]->IsInUse())
        {
            LOG_INFO("Component '" << name << "' is in use");
            return DSL_RESULT_COMPONENT_IN_USE;
        }
        m_components.erase(name);

        LOG_INFO("Component '" << name << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_components.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            // Only if there are Pipelines do we check if the component is in use.
            if (m_pipelines.size())
            {
                for (auto const& imap: m_components)
                {
                    // In the case of Delete all
                    if (imap.second->IsInUse())
                    {
                        LOG_ERROR("Component '" << imap.second->GetName() 
                            << "' is currently in use");
                        return DSL_RESULT_COMPONENT_IN_USE;
                    }
                }
            }

            m_components.clear();
            LOG_INFO("All Components deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw exception on Delete All Components");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }

    uint Services::ComponentListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_components.size();
    }

    DslReturnType Services::ComponentQueueCurrentLevelGet(const char* name, 
        uint unit, uint64_t* currentLevel)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (unit > DSL_COMPONENT_QUEUE_UNIT_OF_TIME)
            {
                LOG_ERROR("Invalid queue measurement unit = " << unit 
                    << " for Component '"  << name << "'");
                DSL_RESULT_COMPONENT_GET_QUEUE_PROPERTY_FAILED;
            }
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            *currentLevel = pQBintrComponent->GetQueueCurrentLevel(unit);

            LOG_INFO("Current queue level = " << *currentLevel 
                << " in units of = " << unit <<  " returned for Component '" 
                << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting current queue level");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueCurrentLevelPrint(const char* name, 
        uint unit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (unit > DSL_COMPONENT_QUEUE_UNIT_OF_TIME)
            {
                LOG_ERROR("Invalid queue measurement unit = " << unit 
                    << " for Component '"  << name << "'");
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            pQBintrComponent->PrintQueueCurrentLevel(unit);
 
            LOG_INFO("Component '" << name << "' print current-level for unit = " 
                << unit << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception printing queue current-level");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
 
    DslReturnType Services::ComponentQueueLeakyGet(const char* name, uint* leaky)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)

            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            *leaky = pQBintrComponent->GetQueueLeaky();

            LOG_INFO("Queue leaky setting = " << *leaky 
                 <<  " returned for Component '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting leaky setting");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueLeakySet(const char* name, uint leaky)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (leaky > DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM)
            {
                LOG_ERROR("Invalid queue leaky setting = " << leaky 
                    << " for Component '"  << name << "'");
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }

            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            if (!pQBintrComponent->SetQueueLeaky(leaky))
            {
                LOG_INFO("Component '" << name 
                    << "' failed to set queue leaky setting = " << leaky);
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }

            LOG_INFO("Component '" << name << "' set queue leaky = " << leaky 
                 <<  "successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception setting leaky setting");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueMaxSizeGet(const char* name, 
        uint unit, uint64_t* maxSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (unit > DSL_COMPONENT_QUEUE_UNIT_OF_TIME)
            {
                LOG_ERROR("Invalid queue measurement unit = " << unit 
                    << " for Component '"  << name << "'");
                DSL_RESULT_COMPONENT_GET_QUEUE_PROPERTY_FAILED;
            }
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            *maxSize = pQBintrComponent->GetQueueMaxSize(unit);

            LOG_INFO("Queue max-size = " << *maxSize 
                << " in units of = " << unit <<  " returned for Component '" 
                << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting queue max-size");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueMaxSizeSet(const char* name, 
        uint unit, uint64_t maxSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (unit > DSL_COMPONENT_QUEUE_UNIT_OF_TIME)
            {
                LOG_ERROR("Invalid queue measurement unit = " << unit 
                    << " for Component '"  << name << "'");
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            if (!pQBintrComponent->SetQueueMaxSize(unit, maxSize))
            {
                LOG_INFO("Component '" << name 
                    << "' failed to set queue in max-size = " << maxSize);
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }

            LOG_INFO("Component '" << name << "' set queue max-size = "  
                << maxSize << " for unit = " << unit << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting queue max-size");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueMinThresholdGet(const char* name, 
        uint unit, uint64_t* minThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (unit > DSL_COMPONENT_QUEUE_UNIT_OF_TIME)
            {
                LOG_ERROR("Invalid queue measurement unit = " << unit 
                    << " for Component '"  << name << "'");
                DSL_RESULT_COMPONENT_GET_QUEUE_PROPERTY_FAILED;
            }
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            *minThreshold = pQBintrComponent->GetQueueMinThreshold(unit);

            LOG_INFO("Queue min-threshold = " << *minThreshold 
                << " in units of = " << unit <<  " returned for Component '" 
                << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting queue min-threshold");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueMinThresholdSet(const char* name, 
        uint unit, uint64_t minThreshold)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            if (unit > DSL_COMPONENT_QUEUE_UNIT_OF_TIME)
            {
                LOG_ERROR("Invalid queue measurement unit = " << unit 
                    << " for Component '"  << name << "'");
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            if (!pQBintrComponent->SetQueueMinThreshold(unit, minThreshold))
            {
                LOG_INFO("Component '" << name 
                    << "' failed to set queue in min-threshold = " << minThreshold);
                return DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED;
            }

            LOG_INFO("Component '" << name << "' set queue min-threshold = " 
                << minThreshold << " for unit = " << unit << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception setting queue min-threshold");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueOverrunListenerAdd(const char* name, 
        dsl_component_queue_overrun_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            if (!pQBintrComponent->AddQueueOverrunListener(listener, clientData))
            {
                LOG_ERROR("Component '" << name 
                    << "' failed to add a queue overrun listener");
                return DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Component '" << name 
                << "' added queue overrun listener successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw an exception adding a queue overrun lister");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::ComponentQueueOverrunListenerRemove(const char* name, 
        dsl_component_queue_overrun_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);
            
            if (!pQBintrComponent->RemoveQueueOverrunListener(listener))
            {
                LOG_ERROR("Component '" << name 
                    << "' failed to remove a queue overrun listener");
                return DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Component '" << name 
                << "' removed queue overrun listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw an exception removing queue overrun lister");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentQueueUnderrunListenerAdd(const char* name, 
        dsl_component_queue_underrun_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);

            if (!pQBintrComponent->AddQueueUnderrunListener(listener, clientData))
            {
                LOG_ERROR("Component '" << name 
                    << "' failed to add a queue underrun listener");
                return DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Component '" << name 
                << "' added queue underrun listener successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw an exception adding a queue underrun lister");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::ComponentQueueUnderrunListenerRemove(const char* name, 
        dsl_component_queue_underrun_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_QBINTR(m_components, name)
            
            DSL_QBINTR_PTR pQBintrComponent = 
                std::dynamic_pointer_cast<QBintr>(m_components[name]);
            
            if (!pQBintrComponent->RemoveQueueUnderrunListener(listener))
            {
                LOG_ERROR("Component '" << name 
                    << "' failed to remove a queue overrun listener");
                return DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Component '" << name 
                << "' removed queue overrun listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw an exception removing queue overrun lister");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentGpuIdGet(const char* name, uint* gpuid)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            
            *gpuid = m_components[name]->GetGpuId();

            LOG_INFO("Current GPU ID = " << *gpuid 
                << " for component '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting GPU Id");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentGpuIdSet(const char* name, uint gpuid)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            
            if (!m_components[name]->SetGpuId(gpuid))
            {
                LOG_INFO("Component '" << name 
                    << "' faild to set GPU Id = " << gpuid);
                return DSL_RESULT_COMPONENT_SET_GPUID_FAILED;
            }

            LOG_INFO("New GPU ID = " << gpuid 
                << " for component '" << name << "'");

            return DSL_RESULT_SUCCESS;
            }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception setting GPU Id");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
}

DslReturnType Services::ComponentNvbufMemTypeGet(const char* name, uint* type)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            
            *type = m_components[name]->GetNvbufMemType();

            LOG_INFO("Current NVIDIA buffer memory type = " << *type 
                << " for component '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception getting NVIDIA buffer memory type");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentNvbufMemTypeSet(const char* name, uint type)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);

            if (type > DSL_NVBUF_MEM_TYPE_SURFACE_ARRAY)
            {
                LOG_ERROR("Invalid NVIDIA buffer memory type = " << type 
                    << " for component '"  << name << "'");
                return DSL_RESULT_COMPONENT_SET_NVBUF_MEM_TYPE_FAILED;
            }
            
            if (!m_components[name]->SetNvbufMemType(type))
            {
                LOG_INFO("Component '" << name 
                    << "' faild to set NVIDIA buffer memory type = " << type);
                return DSL_RESULT_COMPONENT_SET_NVBUF_MEM_TYPE_FAILED;
            }

            LOG_INFO("NVIDIA buffer memorytype = " << type 
                << " set for component '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name 
                << "' threw exception setting NVIDIA buffer memory type");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
}
