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
#include "DslInferBintr.h"

namespace DSL
{
    DslReturnType Services::PrimaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("GIE name '" << name << "' is not unique");
                return DSL_RESULT_INFER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Infer config file: " << inferConfigFile);
            
            std::ifstream configFile(inferConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Infer Config File not found");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            std::string testPath(modelEngineFile);
            if (testPath.size())
            {
                LOG_INFO("Model engine file: " << modelEngineFile);
                
                std::ifstream modelFile(modelEngineFile);
                if (!modelFile.good())
                {
                    LOG_ERROR("Model Engine File not found");
                    return DSL_RESULT_INFER_MODEL_FILE_NOT_FOUND;
                }
            }
            m_components[name] = DSL_PRIMARY_GIE_NEW(name, 
                inferConfigFile, modelEngineFile, interval);
            LOG_INFO("New Primary GIE '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PrimaryTisNew(const char* name, 
        const char* inferConfigFile, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("TIS name '" << name << "' is not unique");
                return DSL_RESULT_INFER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Infer config file: " << inferConfigFile);
            
            std::ifstream configFile(inferConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Infer Config File not found");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            m_components[name] = DSL_PRIMARY_TIS_NEW(name, 
                inferConfigFile, interval);
            LOG_INFO("New Primary TIS '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Primary TIS '" << name << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SecondaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, const char* inferOnGieName, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("GIE name '" << name << "' is not unique");
                return DSL_RESULT_INFER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Infer config file: " << inferConfigFile);
            
            std::ifstream configFile(inferConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Infer Config File not found");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            LOG_INFO("Model engine file: " << modelEngineFile);
            
            std::string testPath(modelEngineFile);
            if (testPath.size())
            {
                std::ifstream modelFile(modelEngineFile);
                if (!modelFile.good())
                {
                    LOG_ERROR("Model Engine File not found");
                    return DSL_RESULT_INFER_MODEL_FILE_NOT_FOUND;
                }
            }
            m_components[name] = DSL_SECONDARY_GIE_NEW(name, 
                inferConfigFile, modelEngineFile, inferOnGieName, interval);

            LOG_INFO("New Secondary GIE '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SecondaryTisNew(const char* name, const char* inferConfigFile,
        const char* inferOnTieName, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Secondary TIS name '" << name << "' is not unique");
                return DSL_RESULT_INFER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Infer config file: " << inferConfigFile);
            
            std::ifstream configFile(inferConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Infer Config File not found");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            m_components[name] = DSL_SECONDARY_TIS_NEW(name, 
                inferConfigFile, inferOnTieName, interval);

            LOG_INFO("New Secondary TIS '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Secondary TIS '" << name << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PrimaryInferPphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_PRIMARY_INFER_TYPE(m_components, name)
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for PrimaryInfer '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("Primary Infer'" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_INFER_HANDLER_ADD_FAILED;
            }
            LOG_INFO("New Primary Infer '" << name << "' added Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::PrimaryInferPphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_PRIMARY_INFER_TYPE(m_components, name);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Primary GIE '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the PrimaryInfer
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Primary GIE '" << name << "'");
                return DSL_RESULT_INFER_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("New Primary Infer '" << name << "' added Pad Probe Handler successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferRawOutputEnabledSet(const char* name, boolean enabled,
        const char* path)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);
                
            if (!pInferBintr->SetRawOutputEnabled(enabled, path))
            {
                LOG_ERROR("GIE '" << name << "' failed to enable raw output");
                return DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST;
            }
            LOG_INFO("Infer '" << name << "' set Raw Output Enabled = "
                << " successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on raw output enabled set");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferConfigFileGet(const char* name, const char** inferConfigFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            *inferConfigFile = pInferBintr->GetInferConfigFile();
            
            LOG_INFO("Infer '" << name << "' returned Config File = '"
                << *inferConfigFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferConfigFileSet(const char* name, const char* inferConfigFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            if (!pInferBintr->SetInferConfigFile(inferConfigFile))
            {
                LOG_ERROR("GIE '" << name << "' failed to set the Infer Config file");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("Infer '" << name << "' set Config File = '"
                << inferConfigFile << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::GieModelEngineFileGet(const char* name, const char** modelEngineFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_INFER_PTR pGieBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            *modelEngineFile = pGieBintr->GetModelEngineFile();

            LOG_INFO("GIE Infer '" << name << "' returned Model Engine File = '"
                << *modelEngineFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::GieModelEngineFileSet(const char* name, const char* modelEngineFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_INFER_PTR pGieBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            if (!pGieBintr->SetModelEngineFile(modelEngineFile))
            {
                LOG_ERROR("GIE '" << name << "' failed to set the Infer Config file");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("GIE Infer '" << name << "' set Model Engine File = '"
                << modelEngineFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            *interval = pInferBintr->GetInterval();

            LOG_INFO("Infer '" << name << "' returned Interval = "
                << *interval << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            if (!pInferBintr->SetInterval(interval))
            {
                LOG_ERROR("GIE '" << name << "' failed to set new Interval");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("Infer '" << name << "' set Interval = "
                << interval << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw an exception setting Interval");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

}    