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
#include "DslMessageBroker.h"

namespace DSL
{
    DslReturnType Services::MessageBrokerNew(const char* name,
        const char* brokerConfigFile, const char* protocolLib, 
        const char* connectionString)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            LOG_INFO("Message Broker config file: " << brokerConfigFile);

            std::ifstream configFile(brokerConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Message Broker config file not found");
                return DSL_RESULT_BROKER_CONFIG_FILE_NOT_FOUND;
            }

            LOG_INFO("Message Broker protocol lib: " << protocolLib);

            std::ifstream protocolLibFile(protocolLib);
            if (!protocolLibFile.good())
            {
                LOG_ERROR("Message Broker protocol lib not found");
                return DSL_RESULT_BROKER_PROTOCOL_LIB_NOT_FOUND;
            }
            
            // ensure broker name uniqueness 
            if (m_messageBrokers.find(name) != m_messageBrokers.end())
            {   
                LOG_ERROR("Message Broker name '" << name << "' is not unique");
                return DSL_RESULT_BROKER_NAME_NOT_UNIQUE;
            }

            m_messageBrokers[name] = DSL_MESSAGE_BROKER_NEW(name, 
                brokerConfigFile, protocolLib, connectionString);

            LOG_INFO("New Message Broker '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Message Broker '" << name 
                << "' threw exception on create");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MessageBrokerSettingsGet(const char* name, 
        const char** brokerConfigFile, const char** protocolLib, 
        const char** connectionString)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_messageBrokers, name);

            m_messageBrokers[name]->GetSettings(brokerConfigFile,
                protocolLib, connectionString);
            LOG_INFO("Message Broker '" << name 
                << "' returned Settings successfully");
            LOG_INFO("Broker config file = '" << *brokerConfigFile  
                << "' Connection string = '" << *connectionString);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Message Broker'" << name 
                << "' threw an exception getting Settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

            
    DslReturnType Services::MessageBrokerSettingsSet(const char* name, 
        const char* brokerConfigFile, const char* protocolLib,
        const char* connectionString)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_messageBrokers, name);

            LOG_INFO("Message Broker config file: " << brokerConfigFile);

            std::ifstream configFile(brokerConfigFile);
            if (!configFile.good())
            {
                LOG_ERROR("Message Broker config file not found");
                return DSL_RESULT_BROKER_CONFIG_FILE_NOT_FOUND;
            }

            LOG_INFO("Message Broker protocol lib: " << protocolLib);

            std::ifstream protocolLibFile(protocolLib);
            if (!protocolLibFile.good())
            {
                LOG_ERROR("Message Broker protocol lib not found");
                return DSL_RESULT_BROKER_PROTOCOL_LIB_NOT_FOUND;
            }

            if (!m_messageBrokers[name]->SetSettings(brokerConfigFile,
                protocolLib, connectionString))
            {
                LOG_ERROR("Message Broker '" << name 
                    << "' failed to set Settings");
                return DSL_RESULT_BROKER_SET_FAILED;
            }
            LOG_INFO("Message Broker '" << name 
                << "' set Settings successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Message Broker'" << name 
                << "' threw an exception setting Message Broker Settings");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MessageBrokerConnect(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->Connect())
            {
                LOG_ERROR("MessageBroker '" << name << "' failed to connect");
                return DSL_RESULT_BROKER_CONNECT_FAILED;
            }

            LOG_INFO("MessageBroker '" << name << "' connected successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name << "' threw an exception on Delete");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MessageBrokerDisconnect(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->Disconnect())
            {
                LOG_ERROR("MessageBroker '" << name << "' failed to disconnect");
                return DSL_RESULT_BROKER_DISCONNECT_FAILED;
            }

            LOG_INFO("MessageBroker '" << name << "' disconnected successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name << "' threw an exception on Delete");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MessageBrokerIsConnected(const char* name,
        boolean* connected)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            *connected = m_messageBrokers[name]->IsConnected();

            LOG_INFO("MessageBroker '" << name << "' returned is-connected = "
                << *connected << " successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name 
                << "' threw an exception getting Is Connected");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MessageBrokerMessageSendAsync(const char* name,
        const char* topic, void* message, size_t size, 
        dsl_message_broker_send_result_listener_cb result_listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->SendMessageAsync(topic, message, 
                size, result_listener, clientData))
            {
                LOG_ERROR("MessageBroker '" << name 
                    << "' failed to send a Message asynchoronously");
                return DSL_RESULT_BROKER_MESSAGE_SEND_FAILED;
            }

            LOG_INFO("MessageBroker '" << name 
                << "' queued the Message to be sent asynchronously successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name 
                << "' threw an exception sending a Message asynchronously");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MessageBrokerSubscriberAdd(const char* name,
        dsl_message_broker_subscriber_cb subscriber, const char** topics,
        uint numTopics, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->AddSubscriber(subscriber, topics,
                numTopics, userData))
            {
                LOG_ERROR("MessageBroker '" << name << "' failed to add Subscriber");
                return DSL_RESULT_BROKER_SUBSCRIBER_ADD_FAILED;
            }

            LOG_INFO("MessageBroker '" << name << "' added Subscriber successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name 
                << "' threw an exception adding Subscriber");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MessageBrokerSubscriberRemove(const char* name,
        dsl_message_broker_subscriber_cb subscriber)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->RemoveSubscriber(subscriber))
            {
                LOG_ERROR("MessageBroker '" << name 
                    << "' failed to remove Subscriber");
                return DSL_RESULT_BROKER_SUBSCRIBER_REMOVE_FAILED;
            }

            LOG_INFO("MessageBroker '" << name << "' removed Subscriber successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name 
                << "' threw an exception adding Subscriber");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MessageBrokerConnectionListenerAdd(const char* name,
        dsl_message_broker_connection_listener_cb handler, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->AddConnectionListener(handler, userData))
            {
                LOG_ERROR("MessageBroker '" << name 
                    << "' failed to add Error Handler");
                return DSL_RESULT_BROKER_LISTENER_ADD_FAILED;
            }

            LOG_INFO("MessageBroker '" << name 
                << "' added a Connection Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name 
                << "' threw an exception adding a Connection Listener");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MessageBrokerConnectionListenerRemove(const char* name,
        dsl_message_broker_connection_listener_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            if (!m_messageBrokers[name]->RemoveConnectionListener(handler))
            {
                LOG_ERROR("MessageBroker '" << name 
                    << "' failed to remove a Connection Listener");
                return DSL_RESULT_BROKER_LISTENER_REMOVE_FAILED;
            }

            LOG_INFO("MessageBroker '" << name 
                << "' removed a Connection Listener successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name 
                << "' threw an exception removing a Connection Listener");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MessageBrokerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_BROKER_NAME_NOT_FOUND(m_messageBrokers, name);

            m_messageBrokers.erase(name);

            LOG_INFO("MessageBroker '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("MessageBroker '" << name << "' threw an exception on Delete");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
            
    }

    DslReturnType Services::MessageBrokerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            m_messageBrokers.clear();

            LOG_INFO("All Message Brokers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on MessageBrokerDeleteAll");
            return DSL_RESULT_BROKER_THREW_EXCEPTION;
        }
    }

    uint Services::MessageBrokerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_messageBrokers.size();
    }
    
}