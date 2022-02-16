/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#include "catch.hpp"
#include "DslServices.h"
#include "DslMessageBroker.h"

using namespace DSL;

static std::string brokerName("iot-message-broker");
static std::string protocolLib("/opt/nvidia/deepstream/deepstream/lib/libnvds_redis_proto.so");
//static std::string protocolLib("/opt/nvidia/deepstream/deepstream/lib/libnvds_azure_edge_proto.so");

//static std::string connectionString("HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>"); 
static std::string connectionString(""); 
static std::string brokerConfigFile("./test/configs/cfg_redis.txt");
static std::string topic1("subscribed/topic/1");
static std::string topic2("subscribed/topic/2");
static std::string topic3("subscribed/topic/3");

SCENARIO( "A new MessageBroker is created correctly", "[MessageBroker]" )
{
    GIVEN( "Attributes for a new MessageBroker" )
    {
        WHEN( "When the MessageBroker is created" )
        {
            DSL_MESSAGE_BROKER_PTR pMessageBroker = 
                DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                    protocolLib.c_str(), connectionString.c_str());
                    
            THEN( "All members are setup correctly" )
            {
                const char* cRetBrokerConfigFile;
                const char* cRetProtocolLib;
                const char* cRetConnectionString;
                pMessageBroker->GetSettings(&cRetBrokerConfigFile,
                    &cRetProtocolLib, &cRetConnectionString);

                std::string retBrokerConfigFile(cRetBrokerConfigFile);
                std::string retProtocolLib(cRetProtocolLib);
                std::string retConnectionString(cRetConnectionString);
                
                REQUIRE (retBrokerConfigFile == brokerConfigFile);
                REQUIRE (retProtocolLib == protocolLib);
                REQUIRE (retConnectionString == connectionString);
            }
        }
        
    }
}

SCENARIO( "A new MessageBroker can connect and disconnect correctly", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        WHEN( "When the MessageBroker is connected" )
        {
            REQUIRE( pMessageBroker->Connect() == true );
            
            THEN( "the MessageBroker can then be disconnected" )
            {
                REQUIRE( pMessageBroker->Disconnect() == true );
            }
        }
        
    }
}

static void message_subscriber_cb_1(uint status, void *message, 
    uint length, const wchar_t* topic, void* client_data)
{    
    std::wcout << L"subscriber-1 called with topic = '" <<
        topic << L"\n";
}
    
static void message_subscriber_cb_2(uint status, void *message, 
    uint length, const wchar_t* topic, void* client_data)
{    
    std::wcout << L"subscriber-2 called with topic = '" <<
        topic << L"\n";
}
    
static void message_subscriber_cb_3(uint status, void *message, 
    uint length, const wchar_t* topic, void* client_data)
{    
    std::wcout << L"subscriber-3 called with topic = '" <<
        topic << L"\n";
}
    
static void connection_listener_cb_1(void* client_data, uint status)
{    
    std::wcout << L"connection-listener-1 called with status = '" <<
        status << L"\n";
}

static void connection_listener_cb_2(void* client_data, uint status)
{    
    std::wcout << L"connection-listener-2 called with status = '" <<
        status << L"\n";
}
    
static void connection_listener_cb_3(void* client_data, uint status)
{    
    std::wcout << L"connection-listener-3 called with status = '" <<
        status << L"\n";
}
    
    
SCENARIO( "An unconnected MessageBroker fails subscriber add", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        WHEN( "the broker is unconnected" )
        {
            const char* topics[] = {topic1.c_str(), NULL};
            
            THEN( "the MessageBroker fails to add a Subscriber" )
            {
                REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_1,
                    topics, 1, NULL) == false );
            }
        }
        
    }
}

SCENARIO( "A connected MessageBroker can add and remove a subscriber", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        REQUIRE( pMessageBroker->Connect() == true );

        WHEN( "When a subscriber is added when the broker is connected" )
        {
            const char* topics[] = {topic1.c_str(), NULL};

            REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_1,
                topics, 1, NULL) == true );

            // second call must fail
            REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_1,
                topics, 1, NULL) == false );
            
            THEN( "the MessageBroker can then be disconnected" )
            {
                REQUIRE( pMessageBroker->RemoveSubscriber(message_subscriber_cb_1) == true );

                // second call must fail
                REQUIRE( pMessageBroker->RemoveSubscriber(message_subscriber_cb_1) == false );
            }
        }
        
    }
}

SCENARIO( "A MessageBroker routes message to three subscribers correctly", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        REQUIRE( pMessageBroker->Connect() == true );

        const char* topics1[] = {topic1.c_str(), NULL};
        const char* topics2[] = {topic2.c_str(), NULL};
        const char* topics3[] = {topic3.c_str(), NULL};

        WHEN( "When the broker receives a messages for an non-subscribed topic" )
        {
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic2.c_str()));
                
            THEN( "the MessageBroker logs the correct warning response" )
            {
                // NOTE: requires manual/visual verification

            }
        }
        WHEN( "When the broker receives three messages from the server" )
        {
            REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_1,
                topics1, 1, NULL) == true );

            REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_2,
                topics2, 1, NULL) == true );
            
            REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_3,
                topics3, 1, NULL) == true );
            
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic1.c_str()));
                
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic2.c_str()));
                
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic3.c_str()));
                
            THEN( "the correct Subscriber is called for each" )
            {
                // NOTE: requires manual/visual verification
                
                REQUIRE( pMessageBroker->RemoveSubscriber(message_subscriber_cb_1) == true );
                REQUIRE( pMessageBroker->RemoveSubscriber(message_subscriber_cb_2) == true );
                REQUIRE( pMessageBroker->RemoveSubscriber(message_subscriber_cb_3) == true );

            }
        }
    }
}

SCENARIO( "A MessageBroker routes messages for multiple topics to a single Subscriber", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        REQUIRE( pMessageBroker->Connect() == true );

        const char* topics[] = {topic1.c_str(), topic2.c_str(), topic3.c_str(), NULL};

        REQUIRE( pMessageBroker->AddSubscriber(message_subscriber_cb_1,
            topics, 3, NULL) == true );

        WHEN( "When the broker receives three messages from the server" )
        {
            
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic1.c_str()));
                
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic2.c_str()));
                
            pMessageBroker->HandleIncomingMessage(NV_MSGBROKER_API_OK, (void*)0x1234567812345678,
                123, const_cast<char*>(topic3.c_str()));
                
            THEN( "the single Subscriber is called for all" )
            {
                // NOTE: requires manual/visual verification
                
                REQUIRE( pMessageBroker->RemoveSubscriber(message_subscriber_cb_1) == true );

            }
        }
    }
}

void async_response(void *user_ptr,  uint status)
{
    std::cout << "response callback called \n";
}

SCENARIO( "A connected MessageBroker can send a message asynchronously", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        REQUIRE( pMessageBroker->Connect() == true );
        
        std::string topic("/topics/mytopic");
        std::string message("this is the message to send");

        WHEN( "When a client sends a message asynchronously" )
        {
            REQUIRE( pMessageBroker->SendMessageAsync(topic.c_str(),
                (void*)message.c_str(), message.size(), async_response, NULL) == true );
            
            THEN( "the asynchronous response callback is called" )
            {
                // NOTE: requires manual/visual verification
            }
        }
        
    }
}

SCENARIO( "A MessageBroker calls all Connection Listeners correctly", "[MessageBroker]" )
{
    GIVEN( "A new MessageBroker" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        REQUIRE( pMessageBroker->Connect() == true );

        WHEN( "When the broker receives three messages from the server" )
        {
            REQUIRE( pMessageBroker->AddConnectionListener(connection_listener_cb_1,
                NULL) == true );

            REQUIRE( pMessageBroker->AddConnectionListener(connection_listener_cb_2,
                NULL) == true );
            
            REQUIRE( pMessageBroker->AddConnectionListener(connection_listener_cb_3,
                NULL) == true );
            
            pMessageBroker->HandleConnectionEvent(NV_MSGBROKER_API_OK);
                
            THEN( "Each listener is called in turn" )
            {
                // NOTE: requires manual/visual verification
                
                REQUIRE( pMessageBroker->RemoveConnectionListener(connection_listener_cb_1) == true );
                REQUIRE( pMessageBroker->RemoveConnectionListener(connection_listener_cb_2) == true );
                REQUIRE( pMessageBroker->RemoveConnectionListener(connection_listener_cb_3) == true );

            }
        }
    }
}

