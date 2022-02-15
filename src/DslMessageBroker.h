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

#ifndef _DSL_MESSAGE_BROKER_H
#define _DSL_MESSAGE_BROKER_H

#include "Dsl.h"
#include "DslBase.h"
#include <nvmsgbroker.h>

namespace DSL {

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_MESSAGE_BROKER_PTR std::shared_ptr<MessageBroker>
    #define DSL_MESSAGE_BROKER_NEW(name, \
            brokerConfigFile, protocolLib, connectionString) \
        std::shared_ptr<MessageBroker>(new MessageBroker(name, \
            brokerConfigFile, protocolLib, connectionString))

    /**
     * @class MessageBroker
     * @brief Implements an MessageBroker class.
     */
    class MessageBroker : public Base
    {
    public:
    
        /**
         * @brief ctor for the MessageBroker class.
         * @param name unique name for the new MessageBroker.
         */
        MessageBroker(const char* name,
            const char* brokerConfigFile, const char* protocolLib, 
            const char* connectionString);
        
        /**
         * @brief dtor for the MessageBroker class.
         */
        ~MessageBroker();
        
        /**
         * @brief Gets the current settings for the MessageBroker.
         * @param[out] brokerConfigFile absolute file-path to the current message
         * borker config file in use.
         * @param[out] protocolLib current protocol adapter library in use
         * @param[out] connectionString current connection string in use.
         */
        void GetSettings(const char** brokerConfigFile, const char** protocolLib,
            const char** connectionString);

        /**
         * @brief Sets the message broker settings for the MessageBroker.
         * @param[in] brokerConfigFile absolute or relative file-path to 
         * a new message borker config file to use.
         * @param[in] protocolLib new protocol adapter library to use.
         * @param[in] connectionString new connection string to use.
         * @return true if successful, false otherwise.
         */
        bool SetSettings(const char* brokerConfigFile, const char* protocolLib, 
            const char* connectionString);
            
        /**
         * @brief Connects the MessageBroker to a remote entitiy.
         * @return true if successful, false otherwise. 
         */
        bool Connect();
        
        /**
         * @brief Disconnects the MessageBroker from a remote entitiy.
         * @return true if successful, false otherwise. 
         */
        bool Disconnect();
        
        /**
         * @brief Returns the current connected state for the MessageBroker.
         * @return true if connected, false otherwise. 
         */
        bool IsConnected();
        
        /**
         * @brief Sends a message synchronously with a specific topic
         * @param topic topic for the message
         * @param message message buffer to send
         * @param size size of the message buffer.
         */
        bool SendMessageSync(const char* topic, uint8_t* message, size_t size);

        /**
         * @brief Sends a message asynchronously with a specific topic
         * @param topic topic for the message
         * @param message message buffer to send
         * @param size size of the message buffer.
         * @param result asynchronous result callback
         * @param clientData client-data to return on callback.
         * @return true on success, false otherwise.
         */
        bool SendMessageAsync(const char* topic, void* message, 
            size_t size, dsl_message_send_result_cb result, void* clientData);

        /**
         * @brief adds a callback to be notified on incoming messages filtered by topic.
         * @param[in] subscriber pointer to the client's function to call on incoming message.
         * @param[in] topics null terminated list of topics to filter on.
         * @param[in] numTopics the number of topics in the filter list. 
         * @param[in] clientData opaque pointer to client data passed back to the subscriber.
         * @return true if successful, false otherwise.
         */
        bool AddSubscriber(dsl_message_subscriber_cb subscriber, 
            const char** topics, uint numTopics, void* clientData);

        /**
         * @brief removes a previously added subscriber callback
         * @param[in] subscriber function function to remove
         * @return true if successful, false otherwise.
         */
        bool RemoveSubscriber(dsl_message_subscriber_cb subscriber);
        
        /**
         * @brief handles an incoming message by directing it to the correct
         * subscriber by topic.
         * @param status one of the NvMsgBrokerErrorType enum values (nvmsgbroker.h)
         * @param message the incoming message payload
         * @param length the length of the payload in bytes
         * @param topic the topic for this message if one is assigned
         */
        void HandleIncomingMessage(NvMsgBrokerErrorType status, 
            void* message, int length, char* topic);

        /**
         * @brief adds a callback to be notified on incoming message errors.
         * @param[in] handler pointer to the client's function to call on error.
         * @param[in] clientData opaque pointer to client data passed back to the handler.
         * @return true if successful, false otherwise.
         */
        bool AddErrorHandler(dsl_message_error_handler_cb handler, 
            void* clientData);

        /**
         * @brief removes a previously added error handler callback
         * @param[in] handler handler function to remove.
         * @return true if successful, false otherwise.
         */
        bool RemoveErrorHandler(dsl_message_error_handler_cb handler);

    private:

        /**
         * @brief absolute path to the message broker config file in use.
         */
        std::string m_brokerConfigFile; 
        
        /**
         * @brief connection string used as end-point for communication with server.
         */
        std::string m_connectionString;
        
        /**
         * @brief Absolute pathname to the library that contains the protocol adapter.
         */
        std::string m_protocolLib; 

        /**
         * @brief handle to the open protocol adapter
         */
        void* m_libHandle;
        
        /**
         * @brief connected state, true while the broker is connected, false otherwise.
         */
        bool m_isConnected;
        
        /**
         * @brief handle returned on successful connection. 
         */
        NvMsgBrokerClientHandle m_connectionHandle;

        /**
         * @brief map of all currently subscribed to message topics mapped
         * by client callback function. Single subscriber per topic only.
         */
        std::map<std::shared_ptr<std::string>, dsl_message_subscriber_cb> m_messageTopics;

        /**
         * @brief map of all currently registered IoT Message Handler
         * callback functions mapped with the user provided data.
         */
        std::map<dsl_message_subscriber_cb, void*> m_messageSubscribers;

        /**
         * @brief map of all currently registered IoT Message Error Handlers
         * callback functions mapped with the user provided data.
         */
        std::map<dsl_message_error_handler_cb, void*> m_errorHandlers;

    };
    
    /**
     * @brief 
     * @param connectionHandle
     * @param status
     */
    static void broker_connection_listener_cb(NvMsgBrokerClientHandle h_ptr, 
        NvMsgBrokerErrorType status);

    /**
     * @brief Broker callback function to receive incoming messages
     * @param flag status of the call, NV_MSGBROKER_API_OK if message receivced
     * @param message address of the message payload
     * @param length length of the message payload
     * @param topic topic for the incoming message
     * @param client_data the instance of Message Broker that subscribed for the message.
     */
    static void broker_message_subscriber_cb(NvMsgBrokerErrorType status, 
        void *msg, int msglen, char *topic, void *user_ptr);    
}


#endif // _DSL_MESSAGE_BROKER_H    
    