/*
The MIT License

Copyright (c)   2022, Prominence AI, Inc.

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
#include "DslMsgSinkBintr.h"

namespace DSL
{
    DslReturnType Services::SinkMsgAzureNew(const char* name, 
        const char* converterConfigFile, uint payloadType, 
        const char* brokerConfigFile, const char* connectionString, 
        const char* topic)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Sink name '" << name << "' is not unique");
                return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
            }

            LOG_INFO("Message Converter config file: " << converterConfigFile);

            std::ifstream configFile(converterConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Message Converter config file not found");
                return DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND;
            }
            std::string testPath(brokerConfigFile);
            if (testPath.size())
            {
                LOG_INFO("Message Broker config file: " << brokerConfigFile);
                
                std::ifstream configFile(brokerConfigFile);
                if (!configFile.good())
                {
                    LOG_ERROR("Message Broker config file not found");
                    return DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND;
                }
            }

            m_components[name] = DSL_AZURE_MSG_SINK_NEW(name,
                converterConfigFile, payloadType, brokerConfigFile, 
                connectionString, topic);

            LOG_INFO("New Message Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Message Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkMsgConverterSettingsGet(const char* name, 
        const char** converterConfigFile, uint* payloadType)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_MESSAGE_SINK(m_components, name);

            DSL_MSG_SINK_PTR pMsgSinkBintr = 
                std::dynamic_pointer_cast<MsgSinkBintr>(m_components[name]);

            pMsgSinkBintr->GetConverterSettings(converterConfigFile,
                payloadType);

            LOG_INFO("Message Sink '" << name 
                << "' returned Message Converter Settings successfully");
            LOG_INFO("Converter config file = '" << *converterConfigFile
                << "' Payload schema type = '" << *payloadType << "'");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Message Sink'" << name 
                << "' threw an exception getting Message Converter Settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

            
    DslReturnType Services::SinkMsgConverterSettingsSet(const char* name, 
        const char* converterConfigFile, uint payloadType)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_MESSAGE_SINK(m_components, name);

            LOG_INFO("Message Converter config file: " << converterConfigFile);

            std::ifstream configFile(converterConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Message Converter config file not found");
                return DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND;
            }
            DSL_MSG_SINK_PTR pMsgSinkBintr = 
                std::dynamic_pointer_cast<MsgSinkBintr>(m_components[name]);

            if (!pMsgSinkBintr->SetConverterSettings(converterConfigFile,
                payloadType))
            {
                LOG_ERROR("Message Sink '" << name 
                    << "' failed to Set Message Converter Settings");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            LOG_INFO("Message Sink '" << name 
                << "' set Message Converter Settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Message Sink'" << name 
                << "' threw an exception setting Message Converter Settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkMsgBrokerSettingsGet(const char* name, 
        const char** brokerConfigFile, const char** connectionString, 
        const char** topic)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_MESSAGE_SINK(m_components, name);

            DSL_MSG_SINK_PTR pMsgSinkBintr = 
                std::dynamic_pointer_cast<MsgSinkBintr>(m_components[name]);

            pMsgSinkBintr->GetBrokerSettings(brokerConfigFile,
                connectionString, topic);
            LOG_INFO("Message Sink '" << name 
                << "' returned Message Broker Settings successfully");
            LOG_INFO("Broker config file = '" << *brokerConfigFile  
                << "' Connection string = '" << *connectionString 
                << "' Topic = '" << *topic);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Message Sink'" << name 
                << "' threw an exception setting Message Broker Settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

            
    DslReturnType Services::SinkMsgBrokerSettingsSet(const char* name, 
        const char* brokerConfigFile, const char* connectionString, const char* topic)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_MESSAGE_SINK(m_components, name);

            LOG_INFO("Message Broker config file: " << brokerConfigFile);

            std::ifstream configFile(brokerConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Message Broker config file not found");
                return DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND;
            }
            DSL_MSG_SINK_PTR pMsgSinkBintr = 
                std::dynamic_pointer_cast<MsgSinkBintr>(m_components[name]);

            if (!pMsgSinkBintr->SetBrokerSettings(brokerConfigFile,
                connectionString, topic))
            {
                LOG_ERROR("Message Sink '" << name 
                    << "' failed to Set Message Broker Settings");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            LOG_INFO("Message Sink '" << name 
                << "' set Message Broker Settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Message Sink'" << name 
                << "' threw an exception setting Message Broker Settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
}
