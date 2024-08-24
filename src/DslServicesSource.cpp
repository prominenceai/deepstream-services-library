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
#include "DslSourceBintr.h"

namespace DSL
{
    DslReturnType Services::SourceAppNew(const char* name, boolean isLive, 
        const char* bufferInFormat, uint width, uint height, uint fpsN, uint fpsD)
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
            m_components[name] = DSL_APP_SOURCE_NEW(name, isLive, 
                bufferInFormat, width, height, fpsN, fpsD);

            LOG_INFO("New App Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New App Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceAppDataHandlersAdd(const char* name,
        dsl_source_app_need_data_handler_cb needDataHandler, 
        dsl_source_app_enough_data_handler_cb enoughDataHandler, 
        void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->AddDataHandlers(needDataHandler,
                enoughDataHandler, clientData))
            {
                LOG_ERROR("Failed to add data-handlers to App Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("App Source '" << name 
                << "' added data-handlers successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception adding data-handlers");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceAppDataHandlersRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->RemoveDataHandlers())
            {
                LOG_ERROR("Failed to remove data-handlers from App Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("App Source '" << name 
                << "' removed data-handlers successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception removing data-handlers");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceAppBufferPush(const char* name, void* buffer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->PushBuffer(buffer))
            {
                LOG_ERROR("Failed to push buffer to App Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            // don't log successful case for performance reasons
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception on push buffer");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceAppSamplePush(const char* name, void* sample)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->PushSample(sample))
            {
                LOG_ERROR("Failed to push sample to App Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            // don't log successful case for performance reasons
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception on push sample");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceAppEos(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->Eos())
            {
                LOG_ERROR("Failed to end-of-stream App Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("End-of-stream event sent to  App Source '" 
                << name << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception on end-of-stream");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceAppStreamFormatGet(const char* name,
        uint* streamFormat)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            *streamFormat = pSourceBintr->GetStreamFormat();
            
            LOG_INFO("App Source '" << name << "' returned buffer-format = "
                << *streamFormat << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception getting buffer-format");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceAppStreamFormatSet(const char* name,
        uint streamFormat)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetStreamFormat(streamFormat))
            {
                LOG_ERROR("Failed to set stream-format to " 
                    << streamFormat << " for App Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("App Source '" << name 
                << "' set stream-format = " << streamFormat << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception setting buffer-format");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceAppDoTimestampGet(const char* name, 
        boolean* doTimestamp)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);
         
            *doTimestamp = pSourceBintr->GetDoTimestamp();

            LOG_INFO("Source '" << name << "' returned do-timestamp = " 
                << *doTimestamp << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting do-timestamp");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                
        
    DslReturnType Services::SourceAppDoTimestampSet(const char* name, 
        boolean doTimestamp)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetDoTimestamp(doTimestamp))
            {
                LOG_ERROR("Failed to set do-timestamp = " << doTimestamp 
                    << " for App Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("App Source '" << name << "' set do-timestamp = " 
                << doTimestamp << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception setting do-timestamp");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                
    DslReturnType Services::SourceAppBlockEnabledGet(const char* name,
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            *enabled = pSourceBintr->GetBlockEnabled();
            
            LOG_INFO("App Source '" << name << "' returned block-enabled = "
                << *enabled << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception getting block-enabled");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceAppBlockEnabledSet(const char* name,
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetBlockEnabled(enabled))
            {
                LOG_ERROR("Failed to set block-enabled to " 
                    << enabled << " for App Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("App Source '" << name 
                << "' set block-enabled = " << enabled << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception setting block-enabled");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceAppCurrentLevelBytesGet(const char* name,
        uint64_t* level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            *level = pSourceBintr->GetCurrentLevelBytes();
            
            LOG_INFO("App Source '" << name 
                << "' returned current-level-bytes = "
                << *level << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception getting current-level-bytes");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceAppMaxLevelBytesGet(const char* name,
        uint64_t* level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            *level = pSourceBintr->GetMaxLevelBytes();
            
            LOG_INFO("App Source '" << name 
                << "' returned max-level-bytes = "
                << *level << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception getting max-level-bytes");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceAppMaxLevelBytesSet(const char* name,
        uint64_t level)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                AppSourceBintr);

            DSL_APP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetMaxLevelBytes(level))
            {
                LOG_ERROR("App Source '" << name 
                    << "' failed to set max-level-bytes = " << level);
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("App Source '" << name << "' set max-level-bytes = "
                << level << " successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("App Source '" << name 
                << "' threw exception setting max-level-bytes");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

//    DslReturnType Services::SourceAppLeakyTypeGet(const char* name,
//        uint* leakyType)
//    {
//        LOG_FUNC();
//        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
//
//        try
//        {
//            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
//            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
//                AppSourceBintr);
//
//            DSL_APP_SOURCE_PTR pSourceBintr = 
//                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);
//
//            *leakyType = pSourceBintr->GetLeakyType();
//            
//            LOG_INFO("App Source '" << name << "' returned leaky-type = "
//                << *leakyType << " successfully");
//            
//            return DSL_RESULT_SUCCESS;
//        }
//        catch(...)
//        {
//            LOG_ERROR("App Source '" << name 
//                << "' threw exception getting leaky-type");
//            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
//        }
//    }
//    
//    DslReturnType Services::SourceAppLeakyTypeSet(const char* name,
//        uint leakyType)
//    {
//        LOG_FUNC();
//        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
//
//        try
//        {
//            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
//            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
//                AppSourceBintr);
//
//            if (leakyType > DSL_QUEUE_LEAKY_TYPE_DOWNSTREAM)
//            {
//                LOG_ERROR("Invalid leaky-type = " << leakyType 
//                    << " for App Source '" << name << "'");
//                return DSL_RESULT_SOURCE_SET_FAILED;
//            }
//            DSL_APP_SOURCE_PTR pSourceBintr = 
//                std::dynamic_pointer_cast<AppSourceBintr>(m_components[name]);
//
//            if (!pSourceBintr->SetLeakyType(leakyType))
//            {
//                LOG_ERROR("App Source '" << name 
//                    << "' failed to set leaky-type = " << leakyType);
//                return DSL_RESULT_SOURCE_SET_FAILED;
//            }
//
//            LOG_INFO("App Source '" << name << "' set leaky-type = "
//                << leakyType << " successfully");
//            
//            return DSL_RESULT_SUCCESS;
//        }
//        catch(...)
//        {
//            LOG_ERROR("App Source '" << name 
//                << "' threw exception setting leaky-type");
//            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
//        }
//    }
    
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

    DslReturnType Services::SourceV4l2New(const char* name, 
        const char* deviceLocation)
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
            m_components[name] = DSL_V4L2_SOURCE_NEW(name, deviceLocation);

            LOG_INFO("New V4L2 Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New V4L2 Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceV4l2DeviceLocationGet(const char* name, 
            const char** deviceLocation)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                V4l2SourceBintr);


            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            *deviceLocation = pSourceBintr->GetDeviceLocation();

            LOG_INFO("V4L2 Source '" << name << "' returned device-location = '" 
                << *deviceLocation << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception getting device-location");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceV4l2DeviceLocationSet(const char* name, 
            const char* deviceLocation)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                V4l2SourceBintr);

            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            if (!pSourceBintr->SetDeviceLocation(deviceLocation))
            {
                LOG_ERROR("Failed to set device-location '" 
                    << deviceLocation << "' for V4L2 Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("V4L2 Source '" << name << "' set device-location = '" 
                << deviceLocation << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception setting device-location");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceV4l2DimensionsSet(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Failed to set dimensions to width = " 
                    << width << " and height = " << height  
                    << " for V4L2 Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("V4L2 Source '" << name << "' set width = " 
                << width << " and height = " << height 
                << " for dimensions successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception setting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::SourceV4l2FrameRateSet(const char* name, 
        uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            if (!pSourceBintr->SetFrameRate(fps_n, fps_d))
            {
                LOG_ERROR("Failed to set frame-rate to fps_n = " 
                    << fps_n << " and fps_d = " << fps_d  
                    << " for V4L2 Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("V4L2 Source '" << name << "' set fps_n = " 
                << fps_n << " and fps_d = " << fps_d 
                << " for frame-rate successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception getting frame-rate");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceV4l2DeviceNameGet(const char* name, 
        const char** deviceName)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            *deviceName = pSourceBintr->GetDeviceName();

            LOG_INFO("V4L2 Source '" << name << "' returned device-name = '" 
                << *deviceName << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception getting device-name");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceV4l2DeviceFdGet(const char* name, 
        int* deviceFd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            *deviceFd = pSourceBintr->GetDeviceFd();

            LOG_INFO("V4L2 Source '" << name << "' returned device-fd = '" 
                << *deviceFd << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception getting device-fd");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceV4l2DeviceFlagsGet(const char* name, 
        uint* deviceFlags)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            *deviceFlags = pSourceBintr->GetDeviceFlags();

            LOG_INFO("V4L2 Source '" << name << "' returned device-flags = '" 
                << int_to_hex(*deviceFlags) << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception getting device-flags");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceV4l2PictureSettingsGet(const char* name, 
        int* brightness, int* contrast, int* hue)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            pSourceBintr->GetPictureSettings(brightness, contrast, hue);

            LOG_INFO("V4L2 Source '" << name 
                << "' returned picture-settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception getting picture-settings");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceV4l2PictureSettingsSet(const char* name, 
        int brightness, int contrast, int hue)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                V4l2SourceBintr);
            
            DSL_V4L2_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<V4l2SourceBintr>(m_components[name]);

            if (!pSourceBintr->SetPictureSettings(brightness, contrast, hue))
            {
                LOG_ERROR("Failed to set picture-settings for V4L2 Source '" 
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("V4L2 Source '" << name 
                << "' set picture-settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("V4L2 Source '" << name 
                << "' threw exception setting picture-settings");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::SourceUriNew(const char* name, const char* uri, 
        boolean isLive, uint skipFrames, uint dropFrameInterval)
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
                name, uri, isLive, skipFrames, dropFrameInterval);

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

    DslReturnType Services::SourceFileFilePathGet(const char* name, 
        const char** filePath)
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

            *filePath = pSourceBintr->GetUri();

            LOG_INFO("File Source '" << name << "' returned file-path = '" 
                << *filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Source '" << name 
                << "' threw exception getting file-path");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceFileFilePathSet(const char* name, 
        const char* filePath)
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

            if (!pSourceBintr->SetUri(filePath))
            {
                LOG_ERROR("Failed to Set file-path '" << filePath 
                    << "' for File Source '" << name << "'");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            LOG_INFO("File Source '" << name << "' set file-path = '" 
                << filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Source '" << name 
                << "' threw exception setting file-path");
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

    DslReturnType Services::SourceImageFilePathGet(const char* name, 
        const char** filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_IMAGE_SOURCE(m_components, name);

            DSL_RESOURCE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<ResourceSourceBintr>(m_components[name]);

            *filePath = pSourceBintr->GetUri();

            LOG_INFO("Image Source '" << name << "' returned file-path = '" 
                << *filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Source '" << name 
                << "' threw exception getting file-path");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceImageFilePathSet(const char* name, 
        const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_IMAGE_SOURCE(m_components, name);

            DSL_RESOURCE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<ResourceSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetUri(filePath))
            {
                LOG_ERROR("Failed to Set file-path '" << filePath 
                    << "' for Image Source '" << name << "'");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            LOG_INFO("Image Source '" << name << "' set file-path = '" 
                << filePath << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Source '" << name 
                << "' threw exception setting file-path");
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
       uint skipFrames, uint dropFrameInterval, uint latency, uint timeout)
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
                name, uri, protocol, skipFrames, dropFrameInterval, latency, timeout);

            LOG_INFO("New RTSP Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RTSP Source '" << name 
                << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceDuplicateNew(const char* name, 
        const char* original)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, original);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, original);

            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[original]);

            // create the new Duplicate Source with is-live set to same
            // value as the original source.
            DSL_DUPLICATE_SOURCE_PTR pDuplicateSource = DSL_DUPLICATE_SOURCE_NEW(
                name, original, pSourceBintr->IsLive());
         
            // add the Duplicate Source to the original Video Sour
            if (!pSourceBintr->AddDuplicateSource(pDuplicateSource))
            {
                LOG_ERROR("Failed to add Duplicate Source  '" << name
                    << " to Original Source '" << original << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            m_components[name] = pDuplicateSource;
                
            LOG_INFO("New Duplicate Source '" << name 
                << "' added to Original Source '"
                << original << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Duplicate Source '" << name 
                << "' threw an exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceDuplicateOriginalGet(const char* name, 
        const char** original)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                DuplicateSourceBintr);

            DSL_DUPLICATE_SOURCE_PTR pDuplicateSourceBintr =
                std::dynamic_pointer_cast<DuplicateSourceBintr>(
                    m_components[name]);
            *original = pDuplicateSourceBintr->GetOriginal();
            
            LOG_INFO("Duplicate Source '" << name 
                << "' returned Original Source '" << *original 
                << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Duplicate Source '" << name 
                << "' threw an exception getting Original Source");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceDuplicateOriginalSet(const char* name, 
        const char* original)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                DuplicateSourceBintr);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, original);

            DSL_DUPLICATE_SOURCE_PTR pDuplicateSourceBintr =
                std::dynamic_pointer_cast<DuplicateSourceBintr>(
                    m_components[name]);
                
            std::string oldOriginalSourceName = pDuplicateSourceBintr->
                GetOriginal();
                
            // The only reason that original source name would be blank
            // is if the original source was deleted. 
            if (oldOriginalSourceName.size())
            {
                // This will always be true, but no harm in checking.
                DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components,
                    oldOriginalSourceName.c_str());
                
                DSL_VIDEO_SOURCE_PTR pOldOriginalSourceBintr = 
                    std::dynamic_pointer_cast<VideoSourceBintr>(
                        m_components[oldOriginalSourceName.c_str()]);
                        
                // Need to first remove the duplicate from the old-original
                if (!pOldOriginalSourceBintr->RemoveDuplicateSource(
                    pDuplicateSourceBintr))
                {
                    LOG_ERROR("Failed to remove Duplicate Source  '" << name
                        << " from Original Source '" << oldOriginalSourceName << "'");
                    return DSL_RESULT_SOURCE_SET_FAILED;
                }
            }
            
            DSL_VIDEO_SOURCE_PTR pNewOriginalSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(
                    m_components[original]);
                
            // add the Duplicate Source to the new original Video Source
            if (!pNewOriginalSourceBintr->AddDuplicateSource(pDuplicateSourceBintr))
            {
                LOG_ERROR("Failed to add Duplicate Source  '" << name
                    << " to Original Source '" << original << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            pDuplicateSourceBintr->SetOriginal(original);
            
            LOG_INFO("Duplicate Source '" << name 
                << "' was updated with new Original Source '" << original 
                << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Duplicate Source '" << name 
                << "' threw an exception setting Original Source");
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

    DslReturnType Services::SourceMediaTypeGet(const char* name, 
        const char** mediaType)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
         
            *mediaType = pSourceBintr->GetMediaType();

            LOG_INFO("Source '" << name << "' returned media-type = " 
                << *mediaType << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting media-type");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutFormatGet(const char* name, 
        const char** format)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            *format = pSourceBintr->GetBufferOutFormat();

            LOG_INFO("Source '" << name << "' returned buffer-out-format = " 
                << *format << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-format");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutFormatSet(const char* name, 
        const char* format)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetBufferOutFormat(format))
            {
                LOG_ERROR("Failed to set buffer-out-format = " << format 
                    << " for Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("Source '" << name << "' set buffer-out-format = " 
                << format << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception setting buffer-out-format");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutDimensionsGet(const char* name, 
        uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            pSourceBintr->GetBufferOutDimensions(width, height);

            LOG_INFO("Source '" << name << "' returned width = " 
                << *width << " and height = " << *height 
                << " for buffer-out-dimensions successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutDimensionsSet(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->SetBufferOutDimensions(width, height))
            {
                LOG_ERROR("Failed to set buffer-out-dimensions to width = " 
                    << width << " and height = " << height  
                    << " for Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("Source '" << name << "' set width = " 
                << width << " and height = " << height 
                << " for buffer-out-dimensions successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::SourceVideoBufferOutFrameRateGet(const char* name, 
        uint* fps_n, uint* fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            pSourceBintr->GetBufferOutFrameRate(fps_n, fps_d);

            LOG_INFO("Source '" << name << "' returned fps_n = " 
                << *fps_n << " and fps_d = " << *fps_d 
                << " for buffer-out-frame-rate successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-frame-rate");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutFrameRateSet(const char* name, 
        uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetBufferOutFrameRate(fps_n, fps_d))
            {
                LOG_ERROR("Failed to set buffer-out-frame-rate to fps_n = " 
                    << fps_n << " and fps_d = " << fps_d  
                    << " for Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("Source '" << name << "' set fps_n = " 
                << fps_n << " and fps_d = " << fps_d 
                << " for buffer-out-frame-rate successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-frame-rate");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutCropRectangleGet(const char* name, 
        uint cropAt, uint* left, uint* top, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            pSourceBintr->GetBufferOutCropRectangle(cropAt, 
                left, top, width, height);

            LOG_INFO("Source '" << name << "' returned crop_at = "
                << cropAt << " left = " << *left 
                << ", top = " << *top << ", width = "
                << *width << ", and height = " << *height
                << " for buffer-out-crop-rectangle successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-crop-rectangle");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutCropRectangleSet(const char* name, 
        uint cropAt, uint left, uint top, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            if (cropAt > DSL_VIDEO_CROP_AT_DEST)
            {
                LOG_ERROR("Invalid 'crop_at' = " << cropAt 
                    << " setting buffer-out-crop rectangel for Source '"
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            if (!pSourceBintr->SetBufferOutCropRectangle(cropAt, 
                left, top, width, height))
            {
                LOG_ERROR("Failed to set buffer-out-crop-rectangle to crop_at = " 
                    << cropAt << ", left = " << left << ", top = " 
                    << top << ", width = " << width <<", and height = "
                    << height << " for Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("Source '" << name << "' set crop_at = "
                << cropAt << ", left = " << left << ", top = " 
                << top << ", width = " << width << ", and height = " 
                << height << " for buffer-out-crop-rectangle successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception setting buffer-out-crop-rectangle");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutOrientationGet(const char* name, 
        uint* orientation)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            *orientation = pSourceBintr->GetBufferOutOrientation();

            LOG_INFO("Source '" << name << "' returned buffer-out-orientation = " 
                << *orientation << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting buffer-out-orientation");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoBufferOutOrientationSet(const char* name, 
        uint orientation)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            if (orientation > DSL_VIDEO_ORIENTATION_FLIP_UPPER_LEFT_TO_LOWER_RIGHT)
            {
                LOG_ERROR("Invalid 'orientation' = " << orientation 
                    << " setting buffer-out-orientaton for Source '"
                    << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            if (!pSourceBintr->SetBufferOutOrientation(orientation))
            {
                LOG_ERROR("Failed to set buffer-out-orientation = " 
                    << orientation << " for Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("Source '" << name << "' set buffer-out-orientation = " 
                << orientation << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception setting buffer-out-orientation");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::SourceVideoDimensionsGet(const char* name, 
        uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            pSourceBintr->GetDimensions(width, height);

            LOG_INFO("Source '" << name << "' returned Width = " 
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
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            pSourceBintr->GetFrameRate(fpsN, fpsD);

            LOG_INFO("Source '" << name << "' returned FPS N = " 
                << *fpsN << " and FPS D = " << *fpsD << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceUriUriGet(const char* name, const char** uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                UriSourceBintr);

            DSL_URI_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<UriSourceBintr>(m_components[name]);

            *uri = pSourceBintr->GetUri();

            LOG_INFO("URI Source '" << name << "' returned URI = '" 
                << *uri << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("URI Source '" << name << "' threw exception getting URI");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceUriUriSet(const char* name, const char* uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                UriSourceBintr);

            DSL_URI_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<UriSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetUri(uri))
            {
                LOG_ERROR("Failed to Set URI '" << uri 
                    << "' for URI Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("URI Source '" << name << "' set URI = '" 
                << uri << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception setting URI");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspUriGet(const char* name, const char** uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                RtspSourceBintr);

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            *uri = pSourceBintr->GetUri();

            LOG_INFO("RTSP Source '" << name << "' returned URI = '" 
                << *uri << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting URI");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::SourceRtspUriSet(const char* name, const char* uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name,
                RtspSourceBintr);

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetUri(uri));
            {
                LOG_ERROR("Failed to Set URI '" << uri 
                    << "' for RTSP Source '" << name << "'");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            LOG_INFO("RTSP Source '" << name << "' set URI = '" 
                << uri << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception setting URI");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceVideoDewarperAdd(const char* name, const char* dewarper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, dewarper);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                dewarper, DewarperBintr);

            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[dewarper]);
         
            if (!pSourceBintr->AddDewarperBintr(pDewarperBintr))
            {
                LOG_ERROR("Failed to add Dewarper '" << dewarper 
                    << "' to Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED;
            }
            LOG_INFO("Uri Source '" << name << "' added Dewarper '" 
                << dewarper << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceVideoDewarperRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_VIDEO_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->RemoveDewarperBintr())
            {
                LOG_ERROR("Failed to remove Dewarper from Uri Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED;
            }
            LOG_INFO("Uri Source '" << name << "' removed its Dewarper successfully");

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

    DslReturnType Services::SourceRtspConnectionParamsGet(const char* name, uint* sleep, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->GetConnectionParams(sleep, timeout);
            
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
    
    DslReturnType Services::SourceRtspConnectionParamsSet(const char* name, 
        uint sleep, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            if (!pSourceBintr->SetConnectionParams(sleep, timeout))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to set reconnection params");
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
    
    DslReturnType Services::SourceRtspConnectionDataGet(const char* name, 
        dsl_rtsp_connection_data* data)
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

    DslReturnType Services::SourceRtspLatencyGet(const char* name, 
        uint* latency)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            *latency = pSourceBintr->GetLatency();

            LOG_INFO("RTSP Source '" << name 
                << "' returned latency = " << *latency 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw exception getting latency");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspLatencySet(const char* name, 
        uint latency)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetLatency(latency))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to set latency = " << latency);
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("RTSP Source '" << name 
                << "' set latency = " << latency 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw exception setting latency");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::SourceRtspDropOnLatencyEnabledGet(const char* name, 
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            *enabled = pSourceBintr->GetDropOnLatencyEnabled();

            LOG_INFO("RTSP Source '" << name 
                << "' returned drop-on-latency enabled = " << *enabled 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw exception getting drop-on-latnecy enabled setting");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspDropOnLatencyEnabledSet(const char* name, 
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetDropOnLatencyEnabled(enabled))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to set drop-on-latency enabled = " << enabled);
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("RTSP Source '" << name 
                << "' set drop-on-latency enabled = " << enabled
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw exception setting latency");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::SourceRtspTlsValidationFlagsGet(const char* name, 
        uint* flags)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            *flags = pSourceBintr->GetTlsValidationFlags();

            LOG_INFO("RTSP Source '" << name 
                << "' returned tls-validation-flags = " << std::hex << *flags 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw exception getting tls-validation-flags");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspTlsValidationFlagsSet(const char* name, 
        uint flags)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RtspSourceBintr);   

            if (flags > DSL_TLS_CERTIFICATE_VALIDATE_ALL)
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to set tls-validation-flags -- invalid flags = "
                    << std::hex << flags);
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetTlsValidationFlags(flags))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to set tls-validation-flags");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }

            LOG_INFO("RTSP Source '" << name 
                << "' set tls-validation-flags = " << std::hex << flags 
                << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw exception setting tls-validation-flags");
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
    
    DslReturnType Services::SourceUniqueIdGet(const char* name, int* uniqueId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);

            *uniqueId = pSourceBintr->GetUniqueId();
            
            LOG_INFO("Source '" << name 
                << "' returned source-id = " << int_to_hex(*uniqueId));
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting unique source-id");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceStreamIdGet(const char* name, int* streamId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);

            // streammux source pad-id == stream-id for all sources
            *streamId = pSourceBintr->GetRequestPadId();
            
            LOG_INFO("Source '" << name 
                << "' returned stream-id = " << *streamId);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name 
                << "' threw exception getting unique source-id");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceNameGet(int uniqueId, const char** name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_sourceNamesById.find(uniqueId) != m_sourceNamesById.end())
        {
            *name = m_sourceNamesById[uniqueId].c_str();
            return DSL_RESULT_SUCCESS;
        }
        *name = NULL;
        return DSL_RESULT_SOURCE_NOT_FOUND;
    }

    void Services::_sourceNameSet(const char* name, uint uniqueId)
    {
        LOG_FUNC();

        // called internally, do not lock mutex
        
        LOG_INFO("Setting Source name = " << name << " with id = " << uniqueId);
        
        m_sourceNamesById[uniqueId] = name;
        m_sourceIdsByName[name] = uniqueId;
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
                std::dynamic_pointer_cast<VideoSourceBintr>(m_components[name]);
                
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

            if (!pSourceBintr->SetState(GST_STATE_PLAYING, 
                DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
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

            boolean isLive = std::dynamic_pointer_cast<SourceBintr>
                (m_components[name])->IsLive();

            LOG_INFO("Source '" << name << "' returned Is-Live = " << isLive );
            return isLive;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DewarperNew(const char* name, 
        const char* configFile, uint sourceId)
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
            
            std::ifstream ifsConfigFile(configFile);
            if (!ifsConfigFile.good())
            {
                LOG_ERROR("Dewarper Config File not found");
                return DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND;
            }

            m_components[name] = DSL_DEWARPER_NEW(name, 
                configFile, sourceId);

            LOG_INFO("New Dewarper '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Dewarper '" << name << "' threw exception on create");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DewarperConfigFileGet(const char* name, 
        const char** configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, DewarperBintr);

            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[name]);

            *configFile = pDewarperBintr->GetConfigFile();

            LOG_INFO("Dewarper '" << name << "' returned config-file = '"
                << *configFile << "' successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Dewarper '" << name 
                << "' threw exception getting the config-file pathspec");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DewarperConfigFileSet(const char* name, 
        const char* configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, DewarperBintr);

            std::ifstream streamConfigFile(configFile);
            if (!streamConfigFile.good())
            {
                LOG_ERROR("Dewarper config file not found");
                return DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND;
            }
            
            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[name]);

            if (!pDewarperBintr->SetConfigFile(configFile))
            {
                LOG_ERROR("Dewarper '" << name 
                    << "' failed to set the config file");
                return DSL_RESULT_DEWARPER_SET_FAILED;
            }
            LOG_INFO("Dewarper '" << name << "' set config-file = '"
                << configFile << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Dewarper '" << name 
                << "' threw exception setting config-file");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DewarperCameraIdGet(const char* name, 
        uint* cameraId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, DewarperBintr);

            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[name]);

            *cameraId = pDewarperBintr->GetCameraId();

            LOG_INFO("camera-id = " << *cameraId 
                << " returned successfully for Dewarper '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Dewarper '" << name 
                << "' threw an exception getting camera-id");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DewarperCameraIdSet(const char* name, 
        uint cameraId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, DewarperBintr);

            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[name]);

            if (!pDewarperBintr->SetCameraId(cameraId))
            {
                LOG_ERROR("Dewarper '" << name 
                    << "' failed to set camera-id = " << cameraId);
                return DSL_RESULT_DEWARPER_SET_FAILED;
            }
            LOG_INFO("camera-id = " << cameraId 
                << " set successfully for Dewarper '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Dewarper '" << name 
                << "' threw an exception setting camera-id");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DewarperNumBatchBuffersGet(const char* name, 
        uint* num)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, DewarperBintr);

            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[name]);

            *num = pDewarperBintr->GetNumBatchBuffers();

            LOG_INFO("num-batch-buffers = " << *num 
                << " returned successfully for Dewarper '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Dewarper '" << name 
                << "' threw an exception getting num-batch-buffers");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DewarperNumBatchBuffersSet(const char* name, 
        uint num)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, DewarperBintr);

            if (num == 0 or num > 4)
            {
                LOG_ERROR("Invalid num-batch-buffers = " << num 
                    << " for Dewarper '" << name << "'");
                return false;
            }
            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[name]);

            if (!pDewarperBintr->SetNumBatchBuffers(num))
            {
                LOG_ERROR("Dewarper '" << name 
                    << "' failed to set num-batch-buffers = " << num);
                return DSL_RESULT_DEWARPER_SET_FAILED;
            }
            LOG_INFO("num-batch-buffers = " << num 
                << " set successfully for Dewarper '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Dewarper '" << name 
                << "' threw an exception setting num-batch-buffers");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordNew(const char* name, 
        const char* outdir, uint container, 
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
                LOG_ERROR("Unable to access outdir '" << outdir 
                    << "' for Record Tape '" << name << "'");
                return DSL_RESULT_TAP_FILE_PATH_NOT_FOUND;
            }

            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " << container 
                    << " for File Tap '" << name << "'");
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
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RecordTapBintr);

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
            LOG_ERROR("Record Tap'" << name 
                << "' threw an exception Starting Session");
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
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, 
                name, RecordTapBintr);

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
            LOG_ERROR("Record Tap'" << name 
                << "' threw an exception setting Stoping Session");
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
        
    DslReturnType Services::TapRecordMaxSizeGet(const char* name, uint* maxSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *maxSize = pRecordTapBintr->GetMaxSize();

            LOG_INFO("Max size = " << *maxSize << 
                " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception getting Max Size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordMaxSizeSet(const char* name, uint maxSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pRecordTapBintr->SetMaxSize(maxSize))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set max size");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Cashe size = " << maxSize << 
                " set successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception setting max size");
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
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *cacheSize = pRecordTapBintr->GetCacheSize();

            LOG_INFO("Cashe size = " << *cacheSize << 
                " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception getting Cache Size");
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
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, 
                RecordTapBintr);

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

}    