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
#include "DslSourceBintr.h"

namespace DSL
{
    DslReturnType Services::SourceCsiNew(const char* name,
        uint width, uint height, uint fpsN, uint fpsD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // Get the Device properties
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            
            if (!deviceProp.integrated)
            {
                LOG_ERROR("CSI Source is not supported on dGPU x86_64 builds");
                return DSL_RESULT_SOURCE_CSI_NOT_SUPPORTED;
            }
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_CSI_SOURCE_NEW(name, width, height, fpsN, fpsD);

            LOG_INFO("New CSI Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New CSI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceCsiSensorIdGet(const char* name, 
            uint* sensorId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                CsiSourceBintr);


            DSL_CSI_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<CsiSourceBintr>(m_components[name]);

            *sensorId = pSourceBintr->GetSensorId();

            LOG_INFO("CSI Source '" << name << "' returned sensor-id = '" 
                << *sensorId << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("CSI Source '" << name 
                << "' threw exception getting sensor-id");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceCsiSensorIdSet(const char* name, 
            uint sensorId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                CsiSourceBintr);

            DSL_CSI_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<CsiSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetSensorId(sensorId))
            {
                LOG_ERROR("Failed to set sensor-id '" 
                    << sensorId << "' for CSI Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("CSI Source '" << name << "' set sensor-id = '" 
                << sensorId << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("CSI Source '" << name 
                << "' threw exception setting sensor-id");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceUsbNew(const char* name,
        uint width, uint height, uint fpsN, uint fpsD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_USB_SOURCE_NEW(name, width, height, fpsN, fpsD);

            LOG_INFO("New USB Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New USB Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceUsbDeviceLocationGet(const char* name, 
            const char** deviceLocation)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                UsbSourceBintr);


            DSL_USB_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<UsbSourceBintr>(m_components[name]);

            *deviceLocation = pSourceBintr->GetDeviceLocation();

            LOG_INFO("USB Source '" << name << "' returned device-location = '" 
                << *deviceLocation << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("USB Source '" << name 
                << "' threw exception getting device-location");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceUsbDeviceLocationSet(const char* name, 
            const char* deviceLocation)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                UsbSourceBintr);

            DSL_USB_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<UsbSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetDeviceLocation(deviceLocation))
            {
                LOG_ERROR("Failed to set device-location '" 
                    << deviceLocation << "' for USB Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("USB Source '" << name << "' set device-location = '" 
                << deviceLocation << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("USB Source '" << name 
                << "' threw exception setting device-location");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceUriNew(const char* name, const char* uri, 
        boolean isLive, uint intraDecode, uint dropFrameInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            std::string stringUri(uri);
            if (stringUri.find("http") == std::string::npos)
            {
                if (isLive)
                {
                    LOG_ERROR("Invalid URI '" << uri << "' for Live source '" << name << "'");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
                std::ifstream streamUriFile(uri);
                if (!streamUriFile.good())
                {
                    LOG_ERROR("URI Source'" << uri << "' Not found");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
            }
            m_components[name] = DSL_URI_SOURCE_NEW(
                name, uri, isLive, intraDecode, dropFrameInterval);

            LOG_INFO("New URI Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New URI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceFileNew(const char* name, const char* filePath, 
            boolean repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            std::string pathString(filePath);
            if (pathString.size())
            {
                std::ifstream streamUriFile(filePath);
                if (!streamUriFile.good())
                {
                    LOG_ERROR("File Source'" << filePath << "' Not found");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
            }
            m_components[name] = DSL_FILE_SOURCE_NEW(
                name, filePath, repeatEnabled);

            LOG_INFO("New File Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New File Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceFilePathGet(const char* name, const char** filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_FILE_SOURCE(m_components, name);

            DSL_RESOURCE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<ResourceSourceBintr>(m_components[name]);

            *filePath = pSourceBintr->GetUri();

            LOG_INFO("File Source '" << name << "' returned File Path = '" 
                << *filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Source '" << name << "' threw exception getting File Path");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceFilePathSet(const char* name, const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_FILE_SOURCE(m_components, name);

            DSL_RESOURCE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<ResourceSourceBintr>(m_components[name]);

            std::ifstream streamUriFile(filePath);
            if (!streamUriFile.good())
            {
                LOG_ERROR("File Source'" << filePath << "' Not found");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            if (!pSourceBintr->SetUri(filePath))
            {
                LOG_ERROR("Failed to Set FilePath '" << filePath << "' for File Source '" << name << "'");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            LOG_INFO("File Source '" << name << "' set File Path = '" 
                << filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Source '" << name << "' threw exception setting File path");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceFileRepeatEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, FileSourceBintr);

            DSL_FILE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<FileSourceBintr>(m_components[name]);
         
            *enabled = pSourceBintr->GetRepeatEnabled();

            LOG_INFO("File Source '" << name << "' returned Repeat Enabled = '" 
                << *enabled << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Source '" << name << "' threw exception getting Repeat Enabled");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceFileRepeatEnabledSet(const char* name, 
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                FileSourceBintr);

            DSL_FILE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<FileSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetRepeatEnabled(enabled))
            {
                LOG_ERROR("Failed to set Repeat Enabled for File Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("File Source '" << name << "' set Repeat Enabled = '" 
                << enabled << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Source '" << name 
                << "' threw exception setting Repeat Enabled");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceImageNew(const char* name, 
        const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            std::ifstream streamUriFile(filePath);
            if (!streamUriFile.good())
            {
                LOG_ERROR("Image Source'" << filePath << "' Not found");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            m_components[name] = DSL_SINGLE_IMAGE_SOURCE_NEW(name, 
                filePath);

            LOG_INFO("New Image Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Image Source '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceImageMultiNew(const char* name, 
        const char* filePath, uint fpsN, uint fpsD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_MULTI_IMAGE_SOURCE_NEW(name, 
                filePath, fpsN, fpsD);

            LOG_INFO("New Multi Image Source '" 
                << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Multi Image Source '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceImageMultiLoopEnabledGet(const char* name,
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                MultiImageSourceBintr);

            DSL_MULTI_IMAGE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<MultiImageSourceBintr>(m_components[name]);
         
            *enabled = pSourceBintr->GetLoopEnabled();

            LOG_INFO("Multi-Image Source '" << name << "' returned loop-enabled = '" 
                << *enabled << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Multi-Image Source '" << name 
                << "' threw exception getting the loop-enabled setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceImageMultiLoopEnabledSet(const char* name,
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                MultiImageSourceBintr);

            DSL_MULTI_IMAGE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<MultiImageSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetLoopEnabled(enabled))
            {
                LOG_ERROR("Failed to set the loop-enabled setting for \\\
                    Multi-Image Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("Multi-Image Source '" << name << "' set loop-enabled = '" 
                << enabled << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Multi-Image Source '" << name 
                << "' threw exception updating the loop-enabled setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceImageMultiIndicesGet(const char* name,
        int* startIndex, int* stopIndex)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                MultiImageSourceBintr);

            DSL_MULTI_IMAGE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<MultiImageSourceBintr>(m_components[name]);
         
            pSourceBintr->GetIndices(startIndex, stopIndex);

            LOG_INFO("Multi-Image Source '" << name << "' returned start-index = '" 
                << *startIndex << "' and stop-index = '" << *stopIndex
                << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Multi-Image Source '" << name 
                << "' threw exception getting the loop-enabled setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceImageMultiIndicesSet(const char* name,
        int startIndex, int stopIndex)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                MultiImageSourceBintr);

            DSL_MULTI_IMAGE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<MultiImageSourceBintr>(m_components[name]);
         
            if (startIndex < 0)
            {
                LOG_ERROR("Invalid start-index = '" << startIndex 
                    << "' for Multi-Image Source '" << name 
                    << "'. Index is zero-based");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
                
            if (!pSourceBintr->SetIndices(startIndex, stopIndex))
            {
                LOG_ERROR("Failed to set the indices for \\\
                    Multi-Image Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("Multi-Image Source '" << name << "' set start-index = '" 
                << startIndex << "' and stop-index = '" << stopIndex
                << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Multi-Image Source '" << name 
                << "' threw exception updating the loop-enabled setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceImageStreamNew(const char* name, const char* filePath, 
        boolean isLive, uint fpsN, uint fpsD, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            std::ifstream streamUriFile(filePath);
            if (!streamUriFile.good())
            {
                LOG_ERROR("Image Stream Source'" << filePath << "' Not found");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            m_components[name] = DSL_IMAGE_STREAM_SOURCE_NEW(
                name, filePath, isLive, fpsN, fpsD, timeout);

            LOG_INFO("New Image Stream Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Image Stream Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceImageStreamTimeoutGet(const char* name, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageStreamSourceBintr);

            DSL_IMAGE_STREAM_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<ImageStreamSourceBintr>(m_components[name]);
         
            *timeout = pSourceBintr->GetTimeout();

            LOG_INFO("Image Stream Source '" << name << "' returned Timeout = " 
                << *timeout << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Stream Source '" << name << "' threw exception getting Timeout");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceImageStreamTimeoutSet(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageStreamSourceBintr);

            DSL_IMAGE_STREAM_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<ImageStreamSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetTimeout(timeout))
            {
                LOG_ERROR("Failed to set Timeout for Image Stream Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("Image Stream Source '" << name << "' set Timeout = " 
                << timeout << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Stream Source '" << name 
                << "' threw exception setting Timeout");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceInterpipeNew(const char* name, 
        const char* listenTo, boolean isLive, 
        boolean acceptEos, boolean acceptEvents)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_INTERPIPE_SOURCE_NEW(
                name, listenTo, isLive, acceptEos, acceptEvents);

            LOG_INFO("New Inter-Pipe Source '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Inter-Pipe Source '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceInterpipeListenToGet(const char* name, 
        const char** listenTo)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                InterpipeSourceBintr);

            DSL_INTERPIPE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<InterpipeSourceBintr>(m_components[name]);
         
            *listenTo = pSourceBintr->GetListenTo();

            LOG_INFO("Inter-Pipe Source '" << name << "' returned listen_to = '" 
                << *listenTo << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Inter-Pipe Source '" << name 
                << "' threw exception getting listen-to setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceInterpipeListenToSet(const char* name, 
        const char* listenTo)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                InterpipeSourceBintr);

            DSL_INTERPIPE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<InterpipeSourceBintr>(m_components[name]);
         
            pSourceBintr->SetListenTo(listenTo);
            
            LOG_INFO("Inter-Pipe Source '" << name << "' set listen-to = '" 
                << listenTo << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Inter-Pipe Source '" << name 
                << "' threw exception setting listen-to");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceInterpipeAcceptSettingsGet(const char* name,
        boolean* acceptEos, boolean* acceptEvents)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                InterpipeSourceBintr);

            DSL_INTERPIPE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<InterpipeSourceBintr>(m_components[name]);
         
            bool bAcceptEos(false), bAcceptEvents(false);
            pSourceBintr->GetAcceptSettings(&bAcceptEos, &bAcceptEvents);
            *acceptEos = bAcceptEos;
            *acceptEvents = bAcceptEvents;

            LOG_INFO("Inter-Pipe Source '" << name << "' returned accept-eos = " 
                << *acceptEos << ", accept-events = " << *acceptEvents 
                << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Inter-Pipe Source '" << name 
                << "' threw exception getting accept setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceInterpipeAcceptSettingsSet(const char* name,
        boolean acceptEos, boolean acceptEvents)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                InterpipeSourceBintr);

            DSL_INTERPIPE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<InterpipeSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetAcceptSettings(acceptEos, acceptEvents))
            {
                LOG_ERROR("Inter-Pipe Source '" << name 
                    << "' failed to set Accept settings");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("Inter-Pipe Source '" << name << "' set accept-eos = " 
                << acceptEos << ", accept-events = " << acceptEvents 
                << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Inter-Pipe Source '" << name 
                << "' threw exception getting accept setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspNew(const char* name, const char* uri,  uint protocol, 
       uint intraDecode, uint dropFrameInterval, uint latency, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_RTSP_SOURCE_NEW(
                name, uri, protocol, intraDecode, dropFrameInterval, latency, timeout);

            LOG_INFO("New RTSP Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RTSP Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourcePphAdd(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to add itself to the Source as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], 
                DSL_PAD_SRC))
            {
                LOG_ERROR("Source '" << name 
                    << "' failed to add Pad Probe Handler '"
                    << handler << "'");
                return DSL_RESULT_SOURCE_HANDLER_ADD_FAILED;
            }
            LOG_INFO("Source '" << name << "' added Pad Probe Handler '"
                << handler << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw an exception adding Pad Probe Handler '" 
                << handler << "'");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::SourcePphRemove(const char* name, const char* handler) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to remove itself from the Tee
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], 
                DSL_PAD_SRC))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Source '" << name << "'");
                return DSL_RESULT_SOURCE_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("Source '" << name << "' removed Pad Probe Handler'" 
                << handler << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw an exception removing Pad Probe Handler '" 
                << handler << "'");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
         
            pSourceBintr->GetDimensions(width, height);

            LOG_INFO("Image Source '" << name << "' returned Width = " 
                << *width << " and Height = " << *height << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::SourceFrameRateGet(const char* name, uint* fpsN, uint* fpsD)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
         
            pSourceBintr->GetFrameRate(fpsN, fpsD);

            LOG_INFO("Image Source '" << name << "' returned FPS N = " 
                << *fpsN << " and FPS D = " << *fpsD << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceDecodeUriGet(const char* name, const char** uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);

            *uri = pSourceBintr->GetUri();

            LOG_INFO("Decode Source '" << name << "' returned URI = '" 
                << *uri << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting URI");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceDecodeUriSet(const char* name, const char* uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetUri(uri));
            {
                LOG_ERROR("Failed to Set URI '" << uri << "' for Decode Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("Decode Source '" << name << "' set URI = '" 
                << uri << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception setting URI");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceDecodeDewarperAdd(const char* name, const char* dewarper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, dewarper);
            DSL_RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, dewarper, DewarperBintr);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);
         
            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[dewarper]);
         
            if (!pSourceBintr->AddDewarperBintr(pDewarperBintr))
            {
                LOG_ERROR("Failed to add Dewarper '" << dewarper << "' to Decode Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED;
            }
            LOG_INFO("Decode Source '" << name << "' added Dewarper '" 
                << dewarper << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceDecodeDewarperRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->RemoveDewarperBintr())
            {
                LOG_ERROR("Failed to remove Dewarper from Decode Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED;
            }
            LOG_INFO("Decode Source '" << name << "' removed its Dewarper successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception removing Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTimeoutGet(const char* name, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            *timeout = pSourceBintr->GetBufferTimeout();

            LOG_INFO("RTSP Source '" << name << "' returned New Buffer Timeout = " 
                << *timeout << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting buffer timeout");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTimeoutSet(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->SetBufferTimeout(timeout);

            LOG_INFO("RTSP Source '" << name << "' set New Buffer Timeout = " 
                << timeout << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception setting buffer timeout");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspReconnectionParamsGet(const char* name, uint* sleep, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->GetReconnectionParams(sleep, timeout);
            
            LOG_INFO("RTSP Source '" << name << "' returned Sleep = " <<
                *sleep << " and Timeout = " << *timeout << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting reconnection params");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspReconnectionParamsSet(const char* name, uint sleep, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            if (!pSourceBintr->SetReconnectionParams(sleep, timeout))
            {
                LOG_ERROR("RTSP Source '" << name << "' failed to set reconnection params");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("RTSP Source '" << name << "' set Sleep = " << 
                sleep << " and Timeout = " << timeout << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception setting reconnection params");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspConnectionDataGet(const char* name, dsl_rtsp_connection_data* data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->GetConnectionData(data);

            LOG_INFO("RTSP Source '" << name << "' returned Connection Data successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting Connection Data");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspConnectionStatsClear(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->ClearConnectionStats();
            
            LOG_INFO("RTSP Source '" << name << "' cleared Connections Stats successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception clearing Connection Stats");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspStateChangeListenerAdd(const char* name, 
        dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->AddStateChangeListener(listener, clientData))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to add a State Change Listener");
                return DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED;
            }
            LOG_INFO("RTSP Source '" << name << "' added State Change Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw an exception adding a State Change Lister");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::SourceRtspStateChangeListenerRemove(const char* name, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->RemoveStateChangeListener(listener))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED;
            }
            LOG_INFO("RTSP Source '" << name << "' removed State Change Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw an exception removeing a State Change Lister");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTapAdd(const char* name, const char* tap)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tap);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);
            DSL_RETURN_IF_COMPONENT_IS_NOT_TAP(m_components, tap);

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            if (pSourceBintr->IsLinked())
            {
                LOG_ERROR("Can not add Tap '" << tap << "' to RTSP Source '" << name << 
                    "' as the Source is in a linked state");
                return DSL_RESULT_SOURCE_TAP_ADD_FAILED;
            }
         
            DSL_TAP_PTR pTapBintr = 
                std::dynamic_pointer_cast<TapBintr>(m_components[tap]);
         
            if (!pSourceBintr->AddTapBintr(pTapBintr))
            {
                LOG_ERROR("Failed to add Tap '" << tap << "' to RTSP Source '" << name << "'");
                return DSL_RESULT_SOURCE_TAP_ADD_FAILED;
            }
            LOG_INFO("RTSP Source '" << name << "' added Tap '" <<
                tap << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Tap");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTapRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
         
            if (pSourceBintr->IsLinked())
            {
                LOG_ERROR("Can not remove Tap from RTSP Source '" << name << 
                    "' as the Source is in a linked state");
                return DSL_RESULT_SOURCE_TAP_ADD_FAILED;
            }

            if (!pSourceBintr->RemoveTapBintr())
            {
                LOG_ERROR("Failed to remove Tap from RTSP Source '" << name << "'");
                return DSL_RESULT_SOURCE_TAP_REMOVE_FAILED;
            }
            LOG_INFO("RTSP Source '" << name << "' removed Tap successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception removing Tap");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceNameGet(int sourceId, const char** name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_sourceNamesById.find(sourceId) != m_sourceNamesById.end())
        {
            *name = m_sourceNamesById[sourceId].c_str();
            return DSL_RESULT_SUCCESS;
        }
        *name = NULL;
        return DSL_RESULT_SOURCE_NOT_FOUND;
    }

    DslReturnType Services::SourceIdGet(const char* name, int* sourceId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_sourceIdsByName.find(name) != m_sourceIdsByName.end())
        {
            *sourceId = m_sourceIdsByName[name];
            return DSL_RESULT_SUCCESS;
        }
        *sourceId = -1;
        return DSL_RESULT_SOURCE_NOT_FOUND;
    }

    uint Services::_sourceNameSet(const char* name)
    {
        LOG_FUNC();
        
        
       uint sourceId(0);
        
        // find the next available unused source-id
        auto ivec = find(m_usedSourceIds.begin(), m_usedSourceIds.end(), false);
        if (ivec != m_usedSourceIds.end())
        {
            sourceId = ivec - m_usedSourceIds.begin();
            m_usedSourceIds[sourceId] = true;
        }
        else
        {
            sourceId = m_usedSourceIds.size();
            m_usedSourceIds.push_back(true);
        }            
        
        m_sourceNamesById[sourceId] = name;
        m_sourceIdsByName[name] = sourceId;
        
        return sourceId;
    }

    bool Services::_sourceNameErase(const char* name)
    {
        LOG_FUNC();

        // called internally, do not lock mutex
        
        if (m_sourceIdsByName.find(name) == m_sourceIdsByName.end())
        {
            LOG_ERROR("Source '" << name << "' not found ");
            return false;
        }
        
        m_usedSourceIds[m_sourceIdsByName[name]] = false;
        m_sourceNamesById.erase(m_sourceIdsByName[name]);
        m_sourceIdsByName.erase(name);
        return true;
    }

    DslReturnType Services::SourcePause(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
                
            if (!pSourceBintr->IsInUse())
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in use");
                return DSL_RESULT_SOURCE_NOT_IN_USE;
            }
            GstState state;
            pSourceBintr->GetState(state, 0);
            if (state != GST_STATE_PLAYING)
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in play");
                return DSL_RESULT_SOURCE_NOT_IN_PLAY;
            }
            if (!pSourceBintr->SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Source '" << name << "' failed to change state to paused");
                return DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE;
            }
            LOG_INFO("Source '" << name << "' Paused successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on pause");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceResume(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
                
            if (!pSourceBintr->IsInUse())
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not in use");
                return DSL_RESULT_SOURCE_NOT_IN_USE;
            }
            GstState state;
            pSourceBintr->GetState(state, 0);
            if (state != GST_STATE_PAUSED)
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not Paused");
                return DSL_RESULT_SOURCE_NOT_IN_PAUSE;
            }

            if (!pSourceBintr->SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Source '" << name << "' failed to change state to Play");
                return DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE;
            }
            LOG_INFO("Source '" << name << "' set to Playing successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on Play");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
        
    boolean Services::SourceIsLive(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            boolean isLive = std::dynamic_pointer_cast<SourceBintr>(m_components[name])->IsLive();

            LOG_INFO("Source '" << name << "' returned Is-Live = " << isLive );
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    uint Services::SourceNumInUseGet()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        uint numInUse(0);
        
        for (auto const& imap: m_pipelines)
        {
            numInUse += imap.second->GetNumSourcesInUse();
        }
        return numInUse;
    }
    
    uint Services::SourceNumInUseMaxGet()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_sourceNumInUseMax;
    }
    
    boolean Services::SourceNumInUseMaxSet(uint max)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        uint numInUse(0);
        
        if (max < GetNumSourcesInUse())
        {
            LOG_ERROR("max setting = " << max << 
                " is less than the current number of Sources in use = " << numInUse);
            return false;
        }
        m_sourceNumInUseMax = max;
        return true;
    }

    DslReturnType Services::DewarperNew(const char* name, const char* configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Dewarper name '" << name << "' is not unique");
                return DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE;
            }
            
            LOG_INFO("Dewarper config file: " << configFile);
            
            std::ifstream ifsConfigFile(configFile);
            if (!ifsConfigFile.good())
            {
                LOG_ERROR("Dewarper Config File not found");
                return DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND;
            }

            m_components[name] = DSL_DEWARPER_NEW(name, configFile);

            LOG_INFO("New Dewarper '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Dewarper '" << name << "' threw exception on create");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordNew(const char* name, const char* outdir, uint container, 
        dsl_record_client_listener_cb clientListener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            struct stat info;

            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Tap name '" << name << "' is not unique");
                return DSL_RESULT_TAP_NAME_NOT_UNIQUE;
            }
            // ensure outdir exists
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir << "' for Record Tape '" << name << "'");
                return DSL_RESULT_TAP_FILE_PATH_NOT_FOUND;
            }

            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " << container << " for File Tap '" << name << "'");
                return DSL_RESULT_TAP_CONTAINER_VALUE_INVALID;
            }

            m_components[name] = DSL_RECORD_TAP_NEW(name, outdir, 
                container, clientListener);
            
            LOG_INFO("New Record Tap '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Record Tap '" << name << "' threw exception on create");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordSessionStart(const char* name, 
        uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->StartSession(start, duration, clientData))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to Start Session");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Session started successfully for Record Tap '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap'" << name << "' threw an exception Starting Session");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordSessionStop(const char* name, boolean sync)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->StopSession(sync))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to Stop Session");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Session stopped successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap'" << name << "' threw an exception setting Stoping Session");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordOutdirGet(const char* name, const char** outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            
            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *outdir = pRecordTapBintr->GetOutdir();
            
            LOG_INFO("Outdir = " << *outdir << " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap'" << name << "' threw an exception setting getting outdir");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordOutdirSet(const char* name, const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            
            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->SetOutdir(outdir))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set the outdir");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Outdir = " << outdir << " set successfully for Record Tap '" << name << "'");
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception setting getting outdir");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TapRecordContainerGet(const char* name, uint* container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *container = pRecordTapBintr->GetContainer();

            LOG_INFO("Container = " << *container 
                << " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception getting Cache Size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordContainerSet(const char* name, uint container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " 
                    << container << " for Record Tap '" << name << "'");
                return DSL_RESULT_TAP_CONTAINER_VALUE_INVALID;
            }

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->SetContainer(container))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set container");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Container = " << container 
                << " set successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception setting container type");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::TapRecordCacheSizeGet(const char* name, uint* cacheSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *cacheSize = pRecordTapBintr->GetCacheSize();

            LOG_INFO("Cashe size = " << *cacheSize << 
                " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception getting Cache Size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordCacheSizeSet(const char* name, uint cacheSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pRecordTapBintr->SetCacheSize(cacheSize))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set cache size");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Cashe size = " << cacheSize << 
                " set successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception setting cache size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::TapRecordDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            pRecordTapBintr->GetDimensions(width, height);

            LOG_INFO("Width = " << *width << " height = " << *height << 
                " returned successfully for Record Tap '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception getting dimensions");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);


            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pRecordTapBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set dimensions");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Width = " << width << " height = " << height << 
                " returned successfully for Record Tap '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception setting dimensions");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordIsOnGet(const char* name, boolean* isOn)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *isOn = pRecordTapBintr->IsOn();

            LOG_INFO("Is on = " << *isOn 
                << "returned successfully for Record Tap '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception getting is-recording-on flag");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordResetDoneGet(const char* name, boolean* resetDone)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *resetDone = pRecordTapBintr->ResetDone();

            LOG_INFO("Reset Done = " << *resetDone 
                << "returned successfully for Record Tap '" << name << "'");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception getting reset done flag");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordVideoPlayerAdd(const char* name, 
        const char* player)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, player);
            DSL_RETURN_IF_PLAYER_IS_NOT_VIDEO_PLAYER(m_players, player)

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->AddVideoPlayer(m_players[player]))
            {
                LOG_ERROR("Record Tap '" << name 
                    << "' failed to add Player '" << player << "'");
                return DSL_RESULT_TAP_PLAYER_ADD_FAILED;
            }
            LOG_INFO("Record Tap '" << name 
                << "added Video Player '" << player << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception adding Player '" << player << "'");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordVideoPlayerRemove(const char* name, 
        const char* player)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, player);
            DSL_RETURN_IF_PLAYER_IS_NOT_VIDEO_PLAYER(m_players, player)

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->RemoveVideoPlayer(m_players[player]))
            {
                LOG_ERROR("Record Tap '" << name 
                    << "' failed to remove Player '" << player << "'");
                return DSL_RESULT_TAP_PLAYER_REMOVE_FAILED;
            }
            LOG_INFO("Record Tap '" << name 
                << "removed Video Player '" << player << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception removing Player '" << player << "'");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordMailerAdd(const char* name, 
        const char* mailer, const char* subject)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->AddMailer(m_mailers[mailer], subject))
            {
                LOG_ERROR("Record Tap '" << name 
                    << "' failed to add Mailer '" << mailer << "'");
                return DSL_RESULT_TAP_MAILER_ADD_FAILED;
            }
            LOG_INFO("Record Tap '" << name 
                << "added Mailer '" << mailer << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception adding Mailer '" << mailer << "'");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordMailerRemove(const char* name, 
        const char* mailer)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->RemoveMailer(m_mailers[mailer]))
            {
                LOG_ERROR("Record Tap '" << name 
                    << "' failed to remove Mailer '" << mailer << "'");
                return DSL_RESULT_TAP_MAILER_REMOVE_FAILED;
            }
            LOG_INFO("Record Tap '" << name 
                << "removed Mailer '" << mailer << "' successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception removing Mailer '" << mailer << "'");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    bool Services::IsSourceComponent(const char* component)
    {
        LOG_FUNC();
     
        return (m_components[component]->IsType(typeid(CsiSourceBintr)) or 
            m_components[component]->IsType(typeid(UriSourceBintr)) or
            m_components[component]->IsType(typeid(RtspSourceBintr)));
    }
 
    uint Services::GetNumSourcesInUse()
    {
        LOG_FUNC();
        
        uint numInUse(0);
        
        for (auto const& imap: m_pipelines)
        {
            numInUse += imap.second->GetNumSourcesInUse();
        }
        return numInUse;
    }
    
}    