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

#include "catch.hpp"
#include "Dsl.h"
#include "DslApi.h"

static std::wstring broker_name(L"message-brocker");
static std::wstring broker_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/device_client/cfg_azure.txt");
static std::wstring protocol_lib(NVDS_AZURE_PROTO_LIB);

static std::wstring topic(L"DSL_MESSAGE_TOPIC");

static std::string message("Hello remote server - edge device calling");


SCENARIO( "The Message Brokers container is updated correctly on new Message Broker", "[message-broker-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_message_broker_list_size() == 0 );

        WHEN( "A new Message Broker is created" ) 
        {
            REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
                protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_message_broker_list_size() == 1 );

                REQUIRE( dsl_message_broker_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Message Brokers container is updated correctly on Message Broker delete", "[message-broker-api]" )
{
    GIVEN( "A Message Broker Component" ) 
    {
        REQUIRE( dsl_message_broker_list_size() == 0 );
        
        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_message_broker_list_size() == 1 );

        WHEN( "A new Message Sink is deleted" ) 
        {
            REQUIRE( dsl_message_broker_delete(broker_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_message_broker_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Message Broker Constructor verifies all input file pathspecs correctly", "[message-broker-api]" )
{
    GIVEN( "An empty list of components " ) 
    {
        REQUIRE( dsl_message_broker_list_size() == 0 );
        
        WHEN( "A new Message Broker is created with an invalid Message Broker config pathsepc" ) 
        {
            std::wstring invalid_broker_config_file(L"./invalide/path/spec");
            
            REQUIRE( dsl_message_broker_new(broker_name.c_str(), 
                invalid_broker_config_file.c_str(), protocol_lib.c_str(),
                NULL) == DSL_RESULT_BROKER_CONFIG_FILE_NOT_FOUND );
            
            THEN( "The list size remains unchanged" )
            {
                REQUIRE( dsl_message_broker_list_size() == 0 );
            }
        }
        WHEN( "A new Message Sink is created with an invalid Protocol Synctax Lib pathsepc" ) 
        {
            std::wstring invalid_protocol_lib(L"./invalide/path/spec");
            
            REQUIRE( dsl_message_broker_new(broker_name.c_str(), 
                broker_config_file.c_str(), invalid_protocol_lib.c_str(),
                NULL) == DSL_RESULT_BROKER_PROTOCOL_LIB_NOT_FOUND );
            
            THEN( "The list size remains unchanged" )
            {
                REQUIRE( dsl_message_broker_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Message Broker's settings can be updated", "[message-broker-api]" )
{
    GIVEN( "A Message Broker in memeory" ) 
    {
        REQUIRE( dsl_message_broker_list_size() == 0 );
        
        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

        const wchar_t* c_ret_broker_config_file;
        const wchar_t* c_ret_protocol_lib;
        const wchar_t* c_ret_connection_string;
        
        // confirm the original settings first
        REQUIRE( dsl_message_broker_settings_get(broker_name.c_str(),
            &c_ret_broker_config_file, &c_ret_protocol_lib,
            &c_ret_connection_string) == DSL_RESULT_SUCCESS );
            
        std::wstring ret_broker_config_file(c_ret_broker_config_file);
        std::wstring ret_protocol_lib(c_ret_protocol_lib);
        std::wstring ret_connection_string(c_ret_connection_string);
        
        std::wstring null_connection_string;

        REQUIRE( ret_broker_config_file == broker_config_file );
        REQUIRE( ret_protocol_lib == protocol_lib );
        REQUIRE( ret_connection_string == null_connection_string );
        
        WHEN( "When new Message Broker settings are set" ) 
        {
            std::wstring new_broker_config_file(
                L"/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt");
            std::wstring new_protocol_lib(L"/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so");
            std::wstring new_connection_string(
                L"HostName=my-hub.azure-devices.net;DeviceId=6789;SharedAccessKey=efghi");
            std::wstring new_topic;
            
            REQUIRE( dsl_message_broker_settings_set(broker_name.c_str(),
                new_broker_config_file.c_str(), new_protocol_lib.c_str(),
                new_connection_string.c_str()) == DSL_RESULT_SUCCESS ); 
            
            THEN( "The correct settings are returned on get" )
            {
                REQUIRE( dsl_message_broker_settings_get(broker_name.c_str(),
                    &c_ret_broker_config_file, &c_ret_protocol_lib, 
                    &c_ret_connection_string) == DSL_RESULT_SUCCESS );
                    
                ret_broker_config_file.assign(c_ret_broker_config_file);
                ret_protocol_lib.assign(c_ret_protocol_lib);
                ret_connection_string.assign(c_ret_connection_string);

                REQUIRE( ret_broker_config_file == new_broker_config_file );
                REQUIRE( ret_protocol_lib == new_protocol_lib );
                REQUIRE( ret_connection_string == new_connection_string );
                
                REQUIRE( dsl_message_broker_delete(broker_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_message_broker_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Message Broker can connect and disconnect", "[message-broker-api]" )
{
    GIVEN( "A new Message Broker in memory" ) 
    {
        boolean retConnected(true);
        
        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
            &retConnected) == DSL_RESULT_SUCCESS );
        REQUIRE( retConnected == false );
        
        WHEN( "The Message Broker is connected" ) 
        {
            REQUIRE( dsl_message_broker_connect(broker_name.c_str()) 
                == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
                &retConnected) == DSL_RESULT_SUCCESS );
            REQUIRE( retConnected == true );

            THEN( "The Message Broker then be disconnected" ) 
            {
                REQUIRE( dsl_message_broker_disconnect(broker_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
                    &retConnected) == DSL_RESULT_SUCCESS );
                REQUIRE( retConnected == false );
                
                REQUIRE( dsl_message_broker_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Message Broker API checks for null pointers", "[message-broker-api]" )
{
    GIVEN( "A set of test Attributes" ) 
    {
        boolean retConnected(true);
        
        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
            &retConnected) == DSL_RESULT_SUCCESS );
        REQUIRE( retConnected == false );
        
        WHEN( "The Message Broker is connected" ) 
        {
            REQUIRE( dsl_message_broker_connect(broker_name.c_str()) 
                == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
                &retConnected) == DSL_RESULT_SUCCESS );
            REQUIRE( retConnected == true );

            THEN( "The Message Broker than be disconnected" ) 
            {
                REQUIRE( dsl_message_broker_disconnect(broker_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
                    &retConnected) == DSL_RESULT_SUCCESS );
                REQUIRE( retConnected == false );
                
                REQUIRE( dsl_message_broker_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
} 

static void connection_listener_cb(void* client_data, uint status)
{    
}

SCENARIO( "A Connection Listener can be added to and removed from a Message Brocker", "[message-broker-api]" )
{
    GIVEN( "A new Message Broker" )
    {
        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Connection Listener is added" )
        {
            REQUIRE( dsl_message_broker_connection_listener_add(broker_name.c_str(),
                connection_listener_cb, NULL) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_message_broker_connection_listener_add(broker_name.c_str(),
                connection_listener_cb, NULL) == DSL_RESULT_BROKER_LISTENER_ADD_FAILED );

            THEN( "The same Connection Listener can be removed" ) 
            {
                REQUIRE( dsl_message_broker_connection_listener_remove(
                    broker_name.c_str(), connection_listener_cb) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_message_broker_connection_listener_remove(
                    broker_name.c_str(), connection_listener_cb) == 
                        DSL_RESULT_BROKER_LISTENER_REMOVE_FAILED );
                    
                REQUIRE( dsl_message_broker_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

static void message_subscriber_cb(void* client_data, uint status, void *message, 
    uint length, const wchar_t* topic)
{    
}

SCENARIO( "A Message Subscriber can be added to and removed from a Message Brocker", "[message-broker-api]" )
{
    GIVEN( "A new Message Broker" )
    {
        const wchar_t* topics[] = {topic.c_str(), NULL};

        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_message_broker_connect(broker_name.c_str()) 
            == DSL_RESULT_SUCCESS );

        WHEN( "A Image Player is added" )
        {
            REQUIRE( dsl_message_broker_subscriber_add(broker_name.c_str(),
                message_subscriber_cb, topics, NULL) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_message_broker_subscriber_add(broker_name.c_str(),
                message_subscriber_cb, topics, NULL) == DSL_RESULT_BROKER_SUBSCRIBER_ADD_FAILED );

            THEN( "The same Image Player can be remove" ) 
            {
                REQUIRE( dsl_message_broker_subscriber_remove(
                    broker_name.c_str(), message_subscriber_cb) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_message_broker_subscriber_remove(
                    broker_name.c_str(), message_subscriber_cb) == 
                        DSL_RESULT_BROKER_SUBSCRIBER_REMOVE_FAILED );
                    
                REQUIRE( dsl_message_broker_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

static void send_message_result_listener_cb(void* client_data, uint status)
{  
    std::cout << "result listener called with status = " << status;
}

SCENARIO( "The Message Broker can send an async message", "[new]" )
{
    GIVEN( "A new Message Broker in a connected state" ) 
    {
        boolean retConnected(true);
        
        REQUIRE( dsl_message_broker_new(broker_name.c_str(), broker_config_file.c_str(), 
            protocol_lib.c_str(), NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_message_broker_connect(broker_name.c_str()) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
            &retConnected) == DSL_RESULT_SUCCESS );
        REQUIRE( retConnected == true );
            
        WHEN( "A message is sent asynchronously " ) 
        {
            REQUIRE( dsl_message_broker_message_send_async(broker_name.c_str(), 
                topic.c_str(), const_cast<char*>(message.c_str()), message.size(), 
                send_message_result_listener_cb, NULL) == DSL_RESULT_SUCCESS );

            THEN( "The client result_listener_cb is called" ) 
            {
                // requires manual/visual verification
                REQUIRE( dsl_message_broker_disconnect(broker_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_message_broker_is_connected(broker_name.c_str(),
                    &retConnected) == DSL_RESULT_SUCCESS );
                REQUIRE( retConnected == false );
                
                REQUIRE( dsl_message_broker_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    
