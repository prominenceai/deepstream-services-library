/*
The MIT License

Copyright (c)   2021-2023, Prominence AI, Inc.

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
#include "DslRemuxerBintr.h"

namespace DSL
{
    DslReturnType Services::RemuxerNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Remuxer name '" << name << "' is not unique");
                return DSL_RESULT_REMUXER_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_REMUXER_NEW(name);
            
            LOG_INFO("New Remuxer '" << name << "' created successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Remuxer '" << name << "' threw exception on create");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBranchAddTo(const char* name, 
        const char* branch, uint* streamIds, uint numStreamIds)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_BRANCH_NAME_NOT_FOUND(m_components, branch);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_COMPONENT_IS_NOT_REMUXER_BRANCH(m_components, branch);
            
            // Can't add components if they're In use by another Branch
            if (m_components[branch]->IsInUse())
            {
                LOG_ERROR("Unable to add branch '" << branch 
                    << "' as it's currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
            // Cast the Branch to a Bintr to call the correct AddChild method.
            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);

            if (!std::dynamic_pointer_cast<RemuxerBintr>(
                    m_components[name])->AddChildTo(pBranchBintr, 
                        streamIds, numStreamIds))
            {
                LOG_ERROR("Remuxer '" << name << 
                    "' failed to add branch '" << branch 
                    << "' to select stream-ids ");
                return DSL_RESULT_REMUXER_BRANCH_ADD_FAILED;
            }
                
            LOG_INFO("Branch '" << branch 
                << "' was added to Remuxer '" << name << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception adding branch '" << branch << "'");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }    

    DslReturnType Services::RemuxerBranchAdd(const char* name, 
        const char* branch)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_BRANCH_NAME_NOT_FOUND(m_components, branch);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_COMPONENT_IS_NOT_REMUXER_BRANCH(m_components, branch);
            
            // Can't add components if they're In use by another Branch
            if (m_components[branch]->IsInUse())
            {
                LOG_ERROR("Unable to add branch '" << branch 
                    << "' as it's currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
            // Cast the Branch to a Bintr to call the correct AddChild method.
            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);

            if (!std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->AddChild(pBranchBintr))
            {
                LOG_ERROR("Remuxer '" << name << 
                    "' failed to add branch '" << branch << "'");
                return DSL_RESULT_REMUXER_BRANCH_ADD_FAILED;
            }
                
            LOG_INFO("Branch '" << branch 
                << "' was added to Remuxer '" << name << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception adding branch '" << branch << "'");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }    
    
    DslReturnType Services::RemuxerBranchRemove(const char* name, 
        const char* branch)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_BRANCH_NAME_NOT_FOUND(m_components, branch);

            DSL_REMUXER_PTR pRemuxerBintr = 
                std::dynamic_pointer_cast<RemuxerBintr>(m_components[name]);

            if (!pRemuxerBintr->IsChild(m_components[branch]))
            {
                LOG_ERROR("Branch '" << branch << 
                    "' is not in use by Remuxer '" << name << "'");
                return DSL_RESULT_REMUXER_BRANCH_IS_NOT_CHILD;
            }

            // Cast the Branch to a Bintr to call the correct AddChile method.
            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);

            if (!pRemuxerBintr->RemoveChild(pBranchBintr))
            {
                LOG_ERROR("Remuxer '" << name << 
                    "' failed to remove branch '" << branch << "'");
                return DSL_RESULT_REMUXER_BRANCH_REMOVE_FAILED;
            }
            LOG_INFO("Branch '" << branch 
                << "' was removed from Remuxer '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception removing branch '" << branch << "'");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBranchRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);

            DSL_REMUXER_PTR pRemuxerBintr = 
                std::dynamic_pointer_cast<RemuxerBintr>(m_components[name]);
                
            // pRemuxerBintr->RemoveAll();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" <<  name
                << "' threw an exception removing all branches");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBranchCountGet(const char* name, uint* count)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);

            *count = std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->GetNumChildren();
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" <<  name
                << "' threw an exception getting branch count");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBatchSizeGet(const char* name,
        uint* batchSize)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            
            *batchSize = std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->GetBatchSize();
            
            LOG_INFO("Remuxer '" << name 
                << "' returned batch-size = " 
                << *batchSize << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception getting batch-size");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBatchSizeSet(const char* name,
        uint batchSize)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            
            if (!std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->OverrideBatchSize(batchSize))
            {
                LOG_ERROR("Remuxer '" << name 
                    << "' failed to set batch-size = "
                    << batchSize);
                return DSL_RESULT_REMUXER_SET_FAILED;
            }
            LOG_INFO("Remuxer '" << name << "' set batch-size = " 
                << batchSize << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception setting batch-size");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBranchConfigFileGet(const char* name,
        const char* branch, const char** configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_COMPONENT_IS_NOT_REMUXER_BRANCH(m_components, branch);

            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);
            
            DSL_REMUXER_PTR pRemuxerBintr = 
                std::dynamic_pointer_cast<RemuxerBintr>(m_components[name]);
                
            if (!pRemuxerBintr->IsChild(pBranchBintr))
            {
                LOG_ERROR("Branch '" << branch 
                    << "' is not a child of Remuxer '" << name << "'");
                return DSL_RESULT_REMUXER_BRANCH_IS_NOT_CHILD;
            }
            *configFile = pRemuxerBintr->GetStreammuxConfigFile(pBranchBintr);

            LOG_INFO("Remuxer '" << name << "' returned Config File = '"
                << configFile << "' for Branch '" << branch << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer'" << name 
                << "' threw exception getting the Config File pathspec");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBranchConfigFileSet(const char* name,
        const char* branch, const char* configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_COMPONENT_IS_NOT_REMUXER_BRANCH(m_components, branch);

            std::string testPath(configFile);
            if (testPath.size())
            {
                LOG_INFO("Tracker config file: " << configFile);
                
                std::ifstream streamConfigFile(configFile);
                if (!streamConfigFile.good())
                {
                    LOG_ERROR("Tracker Config File not found");
                    return DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND;
                }
            }
            
            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);

            DSL_REMUXER_PTR pRemuxerBintr = 
                std::dynamic_pointer_cast<RemuxerBintr>(m_components[name]);

            if (!pRemuxerBintr->SetStreammuxConfigFile(pBranchBintr, configFile))
            {
                LOG_ERROR("Remuxer '" << name 
                    << "' failed to set the Config file");
                return DSL_RESULT_REMUXER_SET_FAILED;
            }
            LOG_INFO("Remuxer '" << name << "' set Config File = '"
                << configFile << "' for Branch '" << branch << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw exception setting Config file");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBatchPropertiesGet(const char* name,
        uint* batchSize, int* batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);

            std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->GetBatchProperties(batchSize, batchTimeout);

            LOG_INFO("Remuxer '" << name 
                << "' returned batch-size = " 
                << *batchSize << " and batch-timeout = " 
                << *batchTimeout << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception getting batch properties");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerBatchPropertiesSet(const char* name,
        uint batchSize, int batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);

            if (!std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->SetBatchProperties(batchSize, batchTimeout))
            {
                LOG_ERROR("Remuxer '" << name 
                    << "' failed to set batch-size = "
                    << batchSize << " and batch-timeout = "
                    << batchTimeout);
                return DSL_RESULT_REMUXER_SET_FAILED;
            }
            LOG_INFO("Remuxer '" << name << "' set batch-size = " 
                << batchSize << " and batch-timeout = " 
                << batchTimeout << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception setting batch properties");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerDimensionsGet(const char* name,
        uint* width, uint* height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);

            std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->GetDimensions(width, height);

            LOG_INFO("Remuxer '" << name << "' returned width = " 
                << *width << " and  height = " << *height << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception getting the output dimensions");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::RemuxerDimensionsSet(const char* name,
        uint width, uint height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);

            if (!std::dynamic_pointer_cast<RemuxerBintr>(
                m_components[name])->SetDimensions(width, height))
            {
                LOG_ERROR("Remuxer '" << name 
                    << "' failed to Set the Streammux output dimensions");
                return DSL_RESULT_REMUXER_SET_FAILED;
            }
            LOG_INFO("Remuxer '" << name << "' set width = " 
                << width << " and height = " << height << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception setting the output dimensions");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::RemuxerPphAdd(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to add itself to the Remuxer as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], 
                DSL_PAD_SINK))
            {
                LOG_ERROR("Remuxer '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_REMUXER_HANDLER_ADD_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << handler 
                << "' added to Remuxer '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::RemuxerPphRemove(const char* name, const char* handler) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RemuxerBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to remove itself from the Remuxer
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], 
                DSL_PAD_SINK))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Remuxer '" << name << "'");
                return DSL_RESULT_REMUXER_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << handler 
                << "' removed from Remuxer '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Remuxer '" << name 
                << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_REMUXER_THREW_EXCEPTION;
        }
    }
}