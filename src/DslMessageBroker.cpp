/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

#include "DslMessageBroker.h"
#include <nvmsgbroker.h>
//#include "../test/unit/DslMessageBrokerStubs.h"

namespace DSL
{
 
    MessageBroker::MessageBrokerMap MessageBroker::g_messageBrokers;
    
    MessageBroker::MessageBroker(const char* name,
        const char* brokerConfigFile, const char* protocolLib, 
        const char* connectionString)
        : Base(name)
        , m_brokerConfigFile(brokerConfigFile)
        , m_connectionString(connectionString)
        , m_protocolLib(protocolLib)
        , m_isConnected(false)
        , m_connectionHandle(NULL)
    {
        LOG_FUNC();
        
    }
    
    MessageBroker::~MessageBroker()
    {
        LOG_FUNC();
        
        if (IsConnected())
        {
            Disconnect();
        }
    }
    
    void MessageBroker::GetSettings(const char** brokerConfigFile,
        const char** protocolLib, const char** connectionString)
    {
        LOG_FUNC();
        
        *brokerConfigFile = m_brokerConfigFile.c_str();
        *protocolLib = m_protocolLib.c_str();
        *connectionString = m_connectionString.c_str();
    }

    bool MessageBroker::SetSettings(const char* brokerConfigFile,
        const char* protocolLib, const char* connectionString)
    {
        LOG_FUNC();
        
        if (IsConnected())
        {
            LOG_ERROR("Unable to set Settings for MessageBroker '" << GetName() 
                << "' as it's currently connected");
            return false;
        }

        m_brokerConfigFile.assign(brokerConfigFile);
        m_protocolLib.assign(protocolLib);
        m_connectionString.assign(connectionString);
        
        return true;
    }
    
    bool MessageBroker::Connect()
    {
        LOG_FUNC();
        
        if (IsConnected())
        {
            LOG_ERROR("MessageBroker '" << GetName() 
                << "' is all ready connected");
            return false;
        }
        
        m_connectionHandle = nv_msgbroker_connect(
            const_cast<char*>(m_connectionString.c_str()), 
            const_cast<char*>(m_protocolLib.c_str()),
            broker_connection_listener_cb, 
            const_cast<char*>(m_brokerConfigFile.c_str()));
            
        if (!m_connectionHandle)
        {
            LOG_ERROR("MessageBroker '" << GetName() << "' failed to connect");
            return false;
        }
        LOG_INFO("MessageBroker '" << GetName() 
            << "' connected successfully - handle = " 
            << std::to_string(((uint64_t)m_connectionHandle)));
            
        // Map this MessageBroker to the connection handle.    
        g_messageBrokers[m_connectionHandle] = this;
        m_isConnected = true;
        return true;
    }
    
    bool MessageBroker::Disconnect()
    {
        LOG_FUNC();
        
        if (!IsConnected())
        {
            LOG_ERROR("MessageBroker '" << GetName() 
                << "' is not in a connected state");
            return false;
        }
        if (nv_msgbroker_disconnect(m_connectionHandle) != NV_MSGBROKER_API_OK)
        {
            LOG_ERROR("MessageBroker '" << GetName() << "' failed to disconnect");
            return false;
        }

        LOG_INFO("MessageBroker '" << GetName() 
            << "' disconnected successfully - handle = " 
            << std::to_string(((uint64_t)m_connectionHandle)));

        // unmap the connection handle and reset flags.
        g_messageBrokers.erase(m_connectionHandle);
        m_connectionHandle = NULL;
        m_isConnected = false;
        return true;
    }
    
    bool MessageBroker::IsConnected()
    {
        LOG_FUNC();
        
        return m_isConnected;
    }

    bool MessageBroker::SendMessageAsync(const char* topic, void* message, 
        size_t size, dsl_message_broker_send_result_listener_cb result_listener, 
        void* clientData)
    {
        LOG_FUNC();
        
        if (!IsConnected())
        {
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' is not connected - unable to send message");
            return false;
        }
        
        NvMsgBrokerClientMsg messagePacket = {const_cast<char*>(topic), message, size};
        
        NvMsgBrokerErrorType retcode = nv_msgbroker_send_async(m_connectionHandle,
            messagePacket, (nv_msgbroker_send_cb_t)result_listener, 
            clientData);
            
        if (retcode != NV_MSGBROKER_API_OK)
        {
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' failed to send message with return code = " << retcode);
            return false;
        }
        return true;
    }
        
    bool MessageBroker::AddSubscriber(dsl_message_broker_subscriber_cb subscriber, 
        const char** topics, uint numTopics, void* clientData)
    {
        LOG_FUNC();
        
        if (!IsConnected())
        {
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' is not connected - unable to add subscriber");
            return false;
        }
        if (m_messageSubscribers.find(subscriber) != m_messageSubscribers.end())
        {   
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' - Subscriber is not unique");
            return false;
        }
        for (const char** topic = topics; *topic; topic++)
        {
            std::shared_ptr<std::string> pTopic = std::shared_ptr<std::string>(
                new std::string(*topic));
            m_messageTopics[pTopic] = subscriber;
        }
        m_messageSubscribers[subscriber] = clientData;
        
        nv_msgbroker_subscribe(m_connectionHandle, 
            const_cast<char**>(topics), numTopics, broker_message_subscriber_cb, this);
        
        return true;
    }
            
    bool MessageBroker::RemoveSubscriber(dsl_message_broker_subscriber_cb subscriber)
    {
        LOG_FUNC();
        
        if (m_messageSubscribers.find(subscriber) == m_messageSubscribers.end())
        {   
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' - Subscriber was not found");
            return false;
        }
        m_messageSubscribers.erase(subscriber);
        
        return true;
    }
    
    void MessageBroker::HandleIncomingMessage(NvMsgBrokerErrorType status, 
        void* message, int length, char* topic)
    {
        LOG_FUNC();
        
        if (!IsConnected())
        {
            LOG_ERROR("MessageBroker '" << GetName() 
                << "' is not connected - unsolicited message");
            return;
        }
        if (topic)
        {
            std::shared_ptr<std::string> pTopic = std::shared_ptr<std::string>(
                new std::string(topic));
            
            for(auto const& imap: m_messageTopics)
            {
                if (*imap.first == *pTopic)
                {
                    try
                    {
                        std::wstring wstrTopic(pTopic->begin(), pTopic->end());
                        imap.second(m_messageSubscribers[m_messageTopics[pTopic]],
                            status, message, length, wstrTopic.c_str());
                    }
                    catch(...)
                    {
                        LOG_ERROR("Exceptions occurred for MessageBroker '" << GetName() 
                            << "' calling Subscriber with an incoming message");
                    }
                    return;
                }
            }
            LOG_WARN("MessageBroker '" << GetName() 
                << "' received a message for topic '" << topic 
                << "', however no client has subscribed for this topic");
            }
    }
            
    bool MessageBroker::AddConnectionListener(
        dsl_message_broker_connection_listener_cb handler, 
        void* clientData)
    {
        LOG_FUNC();
        
        if (m_connectionListeners.find(handler) != m_connectionListeners.end())
        {   
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' - Connection Listener is not unique");
            return false;
        }
        m_connectionListeners[handler] = clientData;
        
        return true;
    }
            
    bool MessageBroker::RemoveConnectionListener(
        dsl_message_broker_connection_listener_cb handler)
    {
        LOG_FUNC();
        
        if (m_connectionListeners.find(handler) == m_connectionListeners.end())
        {   
            LOG_ERROR("MessageBroker  '" << GetName() 
                << "' - Connection Listener was not found");
            return false;
        }
        m_connectionListeners.erase(handler);
        
        return true;
    }

    void MessageBroker::HandleConnectionEvent(NvMsgBrokerErrorType status)
    {
        LOG_FUNC();
        
        for (auto const& imap: m_connectionListeners)
        {
            
            try
            {
                imap.first(imap.second, status);
            }
            catch(...)
            {
                LOG_ERROR("Exception occurred for MessageBroker '" << GetName() 
                    << "' calling Connection Listener");
            }
        }
    }
            
    static void broker_connection_listener_cb(NvMsgBrokerClientHandle h_ptr, 
        NvMsgBrokerErrorType status)
    {
        if (MessageBroker::g_messageBrokers.find(h_ptr) == 
            MessageBroker::g_messageBrokers.end())
        {
            LOG_ERROR("Invalid MessageBroker connection handle received");
            return;
        }
        static_cast<MessageBroker*>(MessageBroker::g_messageBrokers[h_ptr])
            ->HandleConnectionEvent(status);
    }
    
    static void broker_message_subscriber_cb(NvMsgBrokerErrorType status, 
        void *msg, int msglen, char *topic, void *user_ptr)
    {
        static_cast<MessageBroker*>(user_ptr)->HandleIncomingMessage(
            status, msg, msglen, topic);        
    }
    
}