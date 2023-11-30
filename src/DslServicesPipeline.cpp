/*
The MIT License

Copyright (c)   2021, Prominence AI, Inc.

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
#include "DslPipelineBintr.h"

namespace DSL
{
    DslReturnType Services::PipelineNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            if (m_pipelines[name])
            {   
                LOG_ERROR("Pipeline name '" << name << "' is not unique");
                return DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE;
            }
            
            m_pipelines[name] = std::shared_ptr<PipelineBintr>(new PipelineBintr(name));
            LOG_INFO("New PIPELINE '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Pipeline '" << name << "' threw exception on create");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            m_pipelines[name]->RemoveAllChildren();
            m_pipelines.erase(name);

            LOG_INFO("Pipeline '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name << "' threw an exception on Delete");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
            
    }

    DslReturnType Services::PipelineDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            for (auto &imap: m_pipelines)
            {
                imap.second->RemoveAllChildren();
                imap.second = nullptr;
            }
            m_pipelines.clear();

            LOG_INFO("All Pipelines deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on PipelineDeleteAll");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    uint Services::PipelineListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_pipelines.size();
    }
    
    DslReturnType Services::PipelineComponentAdd(const char* name, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
            
            // Can't add components if they're In use by another Pipeline
            if (m_components[component]->IsInUse())
            {
                LOG_ERROR("Unable to add component '" << component 
                    << "' as it's currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }

            if (!m_components[component]->AddToParent(m_pipelines[name]))
            {
                LOG_ERROR("Pipeline '" << name
                    << "' failed component '" << component << "'");
                return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
            }
            LOG_INFO("Component '" << component 
                << "' was added to Pipeline '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name
                << "' threw exception adding component '" << component << "'");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }    
    
    DslReturnType Services::PipelineComponentRemove(const char* name, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

            if (!m_components[component]->IsParent(m_pipelines[name]))
            {
                LOG_ERROR("Component '" << component << 
                    "' is not in use by Pipeline '" << name << "'");
                return DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE;
            }
            m_components[component]->RemoveFromParent(m_pipelines[name]);

            LOG_INFO("Component '" << component 
                << "' was removed from Pipeline '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception removing component");
            return DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED;
        }
    }
    
    DslReturnType Services::PipelineStreammuxBatchSizeGet(const char* name,
        uint* batchSize)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            *batchSize = m_pipelines[name]->GetStreammuxBatchSize();
            
            LOG_INFO("Pipeline '" << name 
                << "' returned Streammuxer batch-size = " 
                << *batchSize << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception getting the Streammux batch-size");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineStreammuxBatchSizeSet(const char* name,
        uint batchSize)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->SetStreammuxBatchSize(batchSize))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to set Streammux batch-size = "
                    << batchSize );
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            LOG_INFO("Pipeline '" << name << "' set batch-size = " 
                << batchSize << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception setting the Streammux batch-size");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineStreammuxNumSurfacesPerFrameGet(const char* name,
        uint* num)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            *num = m_pipelines[name]->GetStreammuxNumSurfacesPerFrame();

            LOG_INFO("Pipeline '" << name 
                << "' returned Streammux num-surfaces-per-frame = " << *num 
                << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name
                << "' threw an exception getting Streammux num-surfaces-per-frame");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreammuxNumSurfacesPerFrameSet(const char* name,
        uint num)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (num > 4)
            {
                LOG_ERROR("The value of '" << num 
                    << "' is invalid for Streammux num-surfaces-per-frame setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            
            if (!m_pipelines[name]->SetStreammuxNumSurfacesPerFrame(num))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to set the Streammux num-surfaces-per-frame setting = "
                    << num);
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            LOG_INFO("Pipeline '" << name << "' set the num-surfaces-per-frame setting = " 
                << num << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception setting Streammux num-surfaces-per-frame");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineStreammuxSyncInputsEnabledGet(const char* name,
        boolean* enabled)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            *enabled = m_pipelines[name]->GetStreammuxSyncInputsEnabled();
            
            LOG_INFO("Pipeline '" << name 
                << "' returned Streammuxer sync-inputs enabled = " 
                << *enabled << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name
                << "' threw an exception getting Streammux sync-inputs enabled");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreammuxSyncInputsEnabledSet(const char* name,
        boolean enabled)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->SetStreammuxSyncInputsEnabled(enabled))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to Set the Streammux sync-inputs enabled setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' set the Streammuxer sync-inputs enabled setting = " 
                << enabled << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception setting Streammux sync-inputs enabled");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreammuxMaxLatencyGet(const char* name, 
        uint* maxLatency)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            *maxLatency = m_pipelines[name]->GetStreammuxMaxLatency();

            LOG_INFO("Pipeline '" << name 
                << "' returned Streammuxer max-latency = " 
                << *maxLatency << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw exception getting max-latency");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineStreammuxMaxLatencySet(const char* name, 
        uint maxLatency)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->SetStreammuxMaxLatency(maxLatency))
            {
                LOG_INFO("Pipeline '" << name 
                    << "' faild to set Streammuxer max-latency = " 
                    << maxLatency);
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }

            LOG_INFO("Pipeline '" << name 
                << "' set the Streammuxer max-latency setting = " 
                << maxLatency << "' successfully");

            return DSL_RESULT_SUCCESS;
            }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw exception setting GPU Id");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineStreammuxTilerAdd(const char* name,
        const char* tiler)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tiler);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, tiler, TilerBintr)
            
            if (!m_pipelines[name]->AddStreammuxTiler(m_components[tiler]))
            {
                LOG_ERROR("Pipeline '" << name << "' failed to add Tiler '" 
                    << tiler << "' to the Streammux's output");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            LOG_INFO("Pipeline '" << name << "' added Tiler '" 
                << tiler << "' to the Streammux's output successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception adding a Tiler to Streammux's output");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineStreammuxTilerRemove(const char* name)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->RemoveStreammuxTiler())
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to remove a Tiler from the Streammux's output");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' removed Tiler from to the Streammux's output successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception removing a Tiler from Streammux's output");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::PipelinePause(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[name])->Pause())
            {
                return DSL_RESULT_PIPELINE_FAILED_TO_PAUSE;
            }

            LOG_INFO("Player '" << name 
                << "' transitioned to a state of PAUSED successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception on Pause");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelinePlay(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[name])->Play())
            {
                return DSL_RESULT_PIPELINE_FAILED_TO_PLAY;
            }

            LOG_INFO("Pipeline '" << name 
                << "' transitioned to a state of PLAYING successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception on Play");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineStop(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[name])->Stop())
            {
                return DSL_RESULT_PIPELINE_FAILED_TO_STOP;
            }

            LOG_INFO("Pipeline '" << name 
                << "' transitioned to a state of NULL successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception on Play");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineStateGet(const char* name, uint* state)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            GstState gstState;
            std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[name])->GetState(gstState, 0);
            *state = (uint)gstState;

            LOG_INFO("Pipeline '" << name 
                << "' returned a current state of '" << StateValueToString(*state) << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception getting state");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineIsLive(const char* name, boolean* isLive)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            *isLive = std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[name])->IsLive();

            LOG_INFO("Pipeline '" << name 
                << "' returned is-live = " << *isLive << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception getting 'is-live'");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineDumpToDot(const char* name, 
        const char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[name]->DumpToDot(const_cast<char*>(filename));
        
        return DSL_RESULT_SUCCESS;
    }   
    
    DslReturnType Services::PipelineDumpToDotWithTs(const char* name, 
        const char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[name]->DumpToDot(const_cast<char*>(filename));

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStateChangeListenerAdd(const char* name, 
        dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->AddStateChangeListener(listener, clientData))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to add a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' added State Change Listener successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception adding a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStateChangeListenerRemove(const char* name, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->RemoveStateChangeListener(listener))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' removed State Change Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception removing a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineEosListenerAdd(const char* name, 
        dsl_eos_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->AddEosListener(listener, clientData))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to add an EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' added End of Stream Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception adding an EOS Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineEosListenerRemove(const char* name, 
        dsl_eos_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->RemoveEosListener(listener))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to remove an EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' removed End of Stream Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception removing an EOS Listener");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineErrorMessageHandlerAdd(const char* name, 
        dsl_error_message_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            if (!m_pipelines[name]->AddErrorMessageHandler(handler, clientData))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to add an Error Message Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' added End of Stream Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception adding an Error Message Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineErrorMessageHandlerRemove(const char* name, 
        dsl_error_message_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            if (!m_pipelines[name]->RemoveErrorMessageHandler(handler))
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to remove an Error Message Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' added End of Stream Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception removing an Error Message Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineErrorMessageLastGet(const char* name,
        std::wstring& source, std::wstring& message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);
            
            m_pipelines[name]->GetLastErrorMessage(source, message);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception removing an Error Message Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineMainLoopNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            if (!m_pipelines[name]->NewMainLoop())
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to create a new Main-Loop");
                return DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' created its own Main-Loop successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception creating Main-Loop");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineMainLoopRun(const char* name)
    {
        LOG_FUNC();
        
        // Note: do not lock mutex - blocking call
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            if (!m_pipelines[name]->RunMainLoop())
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to run its own Main-Loop");
                return DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' returned from running its own Main-Loop successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception running its own Main-Loop");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineMainLoopQuit(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            if (!m_pipelines[name]->QuitMainLoop())
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to quit running its own Main-Loop");
                return DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' quit running its own Main-Loop successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception quiting its own Main-Loop");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }


    DslReturnType Services::PipelineMainLoopDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, name);

            if (!m_pipelines[name]->DeleteMainLoop())
            {
                LOG_ERROR("Pipeline '" << name 
                    << "' failed to delete its own Main-Loop");
                return DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED;
            }
            LOG_INFO("Pipeline '" << name 
                << "' deleted its own Main-Loop successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << name 
                << "' threw an exception deleting its own Main-Loop");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
}