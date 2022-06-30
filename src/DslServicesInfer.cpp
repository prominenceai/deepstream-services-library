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
    DslReturnType Services::InferPrimaryGieNew(const char* name, const char* inferConfigFile,
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
                LOG_ERROR("Infer Config File not found for Primary GIE InferBintr '" 
                        << name << "'");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            std::string testPath(modelEngineFile);
            if (testPath.size())
            {
                LOG_INFO("Model engine file: " << modelEngineFile);
                
                std::ifstream modelFile(modelEngineFile);
                if (!modelFile.good())
                {
                    LOG_ERROR("Model Engine File not found for Primary GIE InferBintr '" 
                        << name << "'");
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

    DslReturnType Services::InferPrimaryTisNew(const char* name, 
        const char* inferConfigFile, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Primary Tis InferBintr name '" 
                    << name << "' is not unique");
                return DSL_RESULT_INFER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Infer config file: " << inferConfigFile);
            
            std::ifstream configFile(inferConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Infer Config File not found for Primary TIS InferBintr '" 
                    << name << "'");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            m_components[name] = DSL_PRIMARY_TIS_NEW(name, 
                inferConfigFile, interval);
            LOG_INFO("New Primary TIS InferBintr '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Primary TIS InferBintr '" << name 
                << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferSecondaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, const char* inferOnGieName, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Secpmdary GIE InferBintr name '" 
                    << name << "' is not unique");
                return DSL_RESULT_INFER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Infer config file: " << inferConfigFile);
            
            std::ifstream configFile(inferConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Infer Config File not found for Secondary GIE InferBintr '" 
                    << name << "'");
                return DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND;
            }
            
            LOG_INFO("Model engine file: " << modelEngineFile);
            
            std::string testPath(modelEngineFile);
            if (testPath.size())
            {
                std::ifstream modelFile(modelEngineFile);
                if (!modelFile.good())
                {
                    LOG_ERROR("Model Engine File not found for Secondary GIE InferBintr '" 
                        << name << "'");
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
            LOG_ERROR("New Secondary GIE InferBintr '" << name 
                << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferSecondaryTisNew(const char* name, const char* inferConfigFile,
        const char* inferOnTieName, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Secondary TIS InferBintr name '" 
                    << name << "' is not unique");
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

            LOG_INFO("New Secondary TIS InferBintr '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Secondary TIS InferBintr '" << name 
                << "' threw exception on create");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferBatchSizeGet(const char* name, uint* size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            *size = pInferBintr->GetBatchSize();

            LOG_INFO("InferBintr '" << name << "' returned batch-size = "
                << *size << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception getting batch-size");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferBatchSizeSet(const char* name, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            if (!pInferBintr->SetBatchSizeByClient(size))
            {
                LOG_ERROR("InferBintr '" << name << "' failed to set batch-size");
                return DSL_RESULT_INFER_SET_FAILED;
            }

            LOG_INFO("InferBintr '" << name << "' set batch-size = "
                << size << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception setting batch-size");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferUniqueIdGet(const char* name, uint* id)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_INFER(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            *id = pInferBintr->GetUniqueId();

            LOG_INFO("InferBintr '" << name << "' returned Unique Id = "
                << *id << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception getting unique Id");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferPrimaryPphAdd(const char* name, 
        const char* handler, uint pad)
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
                LOG_ERROR("Invalid Pad type = " << pad << " for PrimaryInfer '" 
                    << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("Primary InferBintr'" << name 
                    << "' failed to add Pad Probe Handler");
                return DSL_RESULT_INFER_HANDLER_ADD_FAILED;
            }
            LOG_INFO("New Primary InferBintr '" << name 
                << "' added Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Primary InferBintr '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::InferPrimaryPphRemove(const char* name, const char* handler, uint pad) 
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
                LOG_ERROR("Invalid Pad type = " << pad << " for Primary InferBintr '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the PrimaryInfer
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Primary InferBintr '" << name << "'");
                return DSL_RESULT_INFER_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("New Primary InferBintr '" << name << "' added Pad Probe Handler successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Primary InferBintr '" << name << "' threw an exception removing Pad Probe Handler");
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
                LOG_ERROR("InferBintr '" << name << "' failed to enable raw output");
                return DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST;
            }
            LOG_INFO("InferBintr '" << name << "' set Raw Output Enabled = "
                << " successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw exception on raw output enabled set");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferConfigFileGet(const char* name, 
        const char** inferConfigFile)
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
            
            LOG_INFO("InferBintr '" << name << "' returned Config File = '"
                << *inferConfigFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name << "' threw exception on Infer Config file get");
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
                LOG_ERROR("InferBintr '" << name 
                    << "' failed to set the Infer Config file");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("InferBintr '" << name << "' set Config File = '"
                << inferConfigFile << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferGieModelEngineFileGet(const char* name, const char** modelEngineFile)
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

            LOG_INFO("GIE InferBintr '" << name << "' returned Model Engine File = '"
                << *modelEngineFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE InferBintr '" << name 
                << "' threw exception on Model Engine File get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferGieModelEngineFileSet(const char* name, 
        const char* modelEngineFile)
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
                LOG_ERROR("GIE InferBintr '" << name 
                    << "' failed to set the Model Engine file");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("GIE InferBintr '" << name << "' set Model Engine File = '"
                << modelEngineFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE InferBintr '" << name 
                << "' threw exception on Model Engine file get");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferGieTensorMetaSettingsGet(const char* name, 
        boolean* inputEnabled, boolean* outputEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);
            
            bool InputTensorMetaEnabled(false);
            bool OutputTensorMetaEnabled(false);
            
            pInferBintr->GetTensorMetaSettings(&InputTensorMetaEnabled,
                &OutputTensorMetaEnabled);
            
            *inputEnabled = InputTensorMetaEnabled;
            *outputEnabled = OutputTensorMetaEnabled;

            LOG_INFO("GIE InferBintr '" << name << "' returned input-tensor-meta = "
                << *inputEnabled << " and output-tensor-meta = "
                << *outputEnabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception getting tensor-meta settings");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::InferGieTensorMetaSettingsSet(const char* name, 
        boolean inputEnabled, boolean outputEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_INFER_PTR pInferBintr = 
                std::dynamic_pointer_cast<InferBintr>(m_components[name]);

            if (!pInferBintr->SetTensorMetaSettings(inputEnabled, outputEnabled))
            {
                LOG_ERROR("GIE InferBintr '" << name << "' failed to set new Interval");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("GIE InferBintr '" << name << "' set input-tensor-meta = "
                << inputEnabled << " and output-tensor-meta = "
                << outputEnabled << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception setting Interval");
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

            LOG_INFO("InferBintr '" << name << "' returned Interval = "
                << *interval << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception in Interval get");
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
                LOG_ERROR("InferBintr '" << name << "' failed to set new Interval");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("InferBintr '" << name << "' set Interval = "
                << interval << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("InferBintr '" << name 
                << "' threw an exception setting Interval");
            return DSL_RESULT_INFER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::InferNameGet(int inferId, const char** name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_inferNames.find(inferId) != m_inferNames.end())
        {
            *name = m_inferNames[inferId].c_str();
            return DSL_RESULT_SUCCESS;
        }
        *name = NULL;
        return DSL_RESULT_INFER_NAME_NOT_FOUND;
    }

    DslReturnType Services::InferIdGet(const char* name, int* inferId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_inferIds.find(name) != m_inferIds.end())
        {
            *inferId = m_inferIds[name];
            return DSL_RESULT_SUCCESS;
        }
        *inferId = -1;
        return DSL_RESULT_INFER_ID_NOT_FOUND;
    }

    DslReturnType Services::_inferIdGet(const char* name, int* inferId)
    {
        LOG_FUNC();
        
        // called internally, do not lock mutex

        if (m_inferIds.find(name) != m_inferIds.end())
        {
            *inferId = m_inferIds[name];
            return DSL_RESULT_SUCCESS;
        }
        *inferId = -1;
        return DSL_RESULT_INFER_ID_NOT_FOUND;
    }


    DslReturnType Services::_inferNameSet(uint inferId, const char* name)
    {
        LOG_FUNC();
        
        // called internally, do not lock mutex
        
        m_inferNames[inferId] = name;
        m_inferIds[name] = inferId;
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::_inferNameErase(uint inferId)
    {
        LOG_FUNC();

        // called internally, do not lock mutex
        
        if (m_inferNames.find(inferId) != m_inferNames.end())
        {
            m_inferIds.erase(m_inferNames[inferId]);
            m_inferNames.erase(inferId);
            return DSL_RESULT_SUCCESS;
        }
        return DSL_RESULT_SOURCE_NOT_FOUND;
    }

    DslReturnType Services::SegVisualNew(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure element name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Segmentation Visualizer name '" << name 
                    << "' is not unique");
                return DSL_RESULT_SEGVISUAL_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new SegVisualBintr(
                name, width, height));
                
            LOG_INFO("New Segmentation Visualizer '" << name 
                << "' created successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Segmentation Visualizer'" << name 
                << "' threw exception on create");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SegVisualDimensionsGet(const char* name, 
        uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);

            DSL_SEGVISUAL_PTR pSegVisual = 
                std::dynamic_pointer_cast<SegVisualBintr>(m_components[name]);

            pSegVisual->GetDimensions(width, height);
            
            LOG_INFO("Width = " << *width << " height = " << *height << 
                " returned successfully for Segmentation Visualizer '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception getting dimensions");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SegVisualDimensionsSet(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);

            DSL_SEGVISUAL_PTR pSegVisual = 
                std::dynamic_pointer_cast<SegVisualBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pSegVisual->SetDimensions(width, height))
            {
                LOG_ERROR("Segmentation Visualizer '" << name 
                    << "' failed to set dimensions");
                return DSL_RESULT_SEGVISUAL_SET_FAILED;
            }
            LOG_INFO("Width = " << width << " height = " << height << 
                " set successfully for Tiler '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception setting dimensions");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SegVisualPphAdd(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], DSL_PAD_SRC))
            {
                LOG_ERROR("Segmentation Visualizer '" << name 
                    << "' failed to add Pad Probe Handler");
                return DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED;
            }
            LOG_INFO("Segmentation Visualizer '" << name << "' added Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::SegVisualPphRemove(const char* name, const char* handler) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to remove itself from the Tee
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], DSL_PAD_SRC))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Segmentation Visualizer '" << name << "'");
                return DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("Segmentation Visualizer '" << name << "' added Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OfvNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("OFV name '" << name << "' is not unique");
                return DSL_RESULT_OFV_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new OfvBintr(name));

            LOG_INFO("New OFV '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New OFV '" << name << "' threw exception on create");
            return DSL_RESULT_OFV_THREW_EXCEPTION;
        }
    }
    
}    