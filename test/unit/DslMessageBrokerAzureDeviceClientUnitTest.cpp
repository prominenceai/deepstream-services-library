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
static std::string protocolLib("/opt/nvidia/deepstream/deepstream/lib/libnvds_azure_proto.so");

static std::string connectionString;
static std::string brokerConfigFile(
    "/opt/nvidia/deepstream/deepstream-6.0/sources/libs/azure_protocol_adaptor/device_client/cfg_azure.txt");

SCENARIO( "A new MessageBrokerDeviceClient is created correctly", "[MessageBrokerDeviceClient]" )
{
    GIVEN( "Attributes for a new MessageBrokerDeviceClient" )
    {
        WHEN( "When the MessageBrokerDeviceClient is created" )
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

SCENARIO( "A new MessageBrokerDeviceClient can connect and disconnect correctly", 
	"[MessageBrokerDeviceClient]" )
{
    GIVEN( "A new MessageBrokerDeviceClient" )
    {
        DSL_MESSAGE_BROKER_PTR pMessageBroker = 
            DSL_MESSAGE_BROKER_NEW(brokerName.c_str(), brokerConfigFile.c_str(), 
                protocolLib.c_str(), connectionString.c_str());

        WHEN( "When the MessageBrokerDeviceClient is connected" )
        {
            REQUIRE( pMessageBroker->Connect() == true );
            
            THEN( "the MessageBrokerDeviceClient can then be disconnected" )
            {
                REQUIRE( pMessageBroker->Disconnect() == true );
            }
        }
    }
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
    

static void async_response(void *user_ptr,  uint status)
{
    std::cout << "response callback called \n";
}

SCENARIO( "A connected MessageBrokerDeviceClient can send a message asynchronously", 
	"[MessageBrokerDeviceClient]" )
{
    GIVEN( "A new MessageBrokerDeviceClient" )
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
                REQUIRE( pMessageBroker->Disconnect() == true );
            }
        }
        
    }
}

SCENARIO( "A MessageBrokerDeviceClient calls all Connection Listeners correctly", 
	"[MessageBrokerDeviceClient]" )
{
    GIVEN( "A new MessageBrokerDeviceClient" )
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
                
                REQUIRE( pMessageBroker->RemoveConnectionListener(
					connection_listener_cb_1) == true );
                REQUIRE( pMessageBroker->RemoveConnectionListener(
					connection_listener_cb_2) == true );
                REQUIRE( pMessageBroker->RemoveConnectionListener(
					connection_listener_cb_3) == true );

                REQUIRE( pMessageBroker->Disconnect() == true );
            }
        }
    }
}

