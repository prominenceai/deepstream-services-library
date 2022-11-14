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
#include "DslTrackerBintr.h"

namespace DSL
{
    DslReturnType Services::TrackerNew(const char* name, const char* configFile, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("IOU Tracker name '" << name << "' is not unique");
                return DSL_RESULT_TRACKER_NAME_NOT_UNIQUE;
            }
            
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
            
            m_components[name] = std::shared_ptr<Bintr>(new TrackerBintr(
                name, configFile, width, height));
                
            LOG_INFO("New Tracker '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw exception on create");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerLibFileGet(const char* name, 
        const char** libFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);
            
            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            *libFile = pTrackerBintr->GetLibFile();

            LOG_INFO("Tracker '" << name << "' returned Lib File = '"
                << *libFile << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw exception getting the Lib File pathspec");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerLibFileSet(const char* name, 
        const char* libFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            std::string testPath(libFile);
            if (testPath.size())
            {
                LOG_INFO("Tracker lib file: " << libFile);
                
                std::ifstream streamLibFile(libFile);
                if (!streamLibFile.good())
                {
                    LOG_ERROR("Tracker Lib File not found");
                    return DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND;
                }
            }
            
            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!pTrackerBintr->SetLibFile(libFile))
            {
                LOG_ERROR("Tracker '" << name << "' failed to set the Lib file");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("Tracker '" << name << "' set Lib File = '"
                << libFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception setting Lib file");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerConfigFileGet(const char* name, 
        const char** configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);
            
            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            *configFile = pTrackerBintr->GetConfigFile();

            LOG_INFO("Tracker '" << name << "' returned Config File = '"
                << *configFile << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw exception getting the Config File pathspec");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerConfigFileSet(const char* name, 
        const char* configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

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
            
            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!pTrackerBintr->SetConfigFile(configFile))
            {
                LOG_ERROR("Tracker '" << name << "' failed to set the Config file");
                return DSL_RESULT_INFER_SET_FAILED;
            }
            LOG_INFO("Tracker '" << name << "' set Config File = '"
                << configFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception setting Config file");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            trackerBintr->GetDimensions(width, height);

            LOG_INFO("Tracker '" << name << "' returned Width = " 
                << *width << " and Height = " << *height << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!trackerBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Tracker '" << name << "' failed to set dimensions");
                return DSL_RESULT_TRACKER_SET_FAILED;
            }
            LOG_INFO("Tracker '" << name << "' set Width = " 
                << width << " and Height = " << height << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerBatchProcessingEnabledGet(const char* name, 
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            *enabled = trackerBintr->GetBatchProcessingEnabled();

            LOG_INFO("DCF Tracker '" << name << "' returned Batch Processing Enabed = " 
                << *enabled  << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw an exception getting batch-process enabled setting");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerBatchProcessingEnabledSet(const char* name, 
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!trackerBintr->SetBatchProcessingEnabled(enabled))
            {
                LOG_ERROR("Tracker '" << name 
                    << "' failed to set batch-processing enabled setting");
                return DSL_RESULT_TRACKER_SET_FAILED;
            }
            LOG_INFO("Tracker '" << name << "' set Batch Processing Enabed = " 
                << enabled  << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw an exception setting batch-processing enabled setting");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerPastFrameReportingEnabledGet(const char* name, 
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            *enabled = trackerBintr->GetPastFrameReportingEnabled();

            LOG_INFO("Tracker '" << name << "' returned Past Frame Reporting Enabed = " 
                << *enabled  << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw an exception getting past-frame-reporting enabled setting");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerPastFrameReportingEnabledSet(const char* name, 
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!trackerBintr->SetPastFrameReportingEnabled(enabled))
            {
                LOG_ERROR("Tracker '" << name 
                    << "' failed to set past-frame-reporting enabled setting");
                return DSL_RESULT_TRACKER_SET_FAILED;
            }
            LOG_INFO("Tracker '" << name << "' set Past Frame Reporting Enabed = " 
                << enabled  << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw an exception setting past-frame-reporting enabled setting");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerPphAdd(const char* name, 
        const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad 
                    << " for Tracker '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(
                m_components[name], pad))
            {
                LOG_ERROR("Tracker '" << name 
                    << "' failed to add Pad Probe Handler");
                return DSL_RESULT_TRACKER_HANDLER_ADD_FAILED;
            }
            LOG_INFO("Tracker '" << name 
                << "' added Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::TrackerPphRemove(const char* name, 
        const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, TrackerBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tracker '" 
                    << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the Tracker
            if (!m_padProbeHandlers[handler]->RemoveFromParent(
                m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Tracker '" << name << "'");
                return DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("Tracker '" << name
                << "' removed Pad Probe Handler successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name 
                << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }
        

}