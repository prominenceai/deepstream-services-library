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

static std::wstring sink_name(L"msg-sink");
static const std::wstring converter_config_file(
	L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.txt");
static const std::wstring protocol_lib(NVDS_AZURE_PROTO_LIB);
static const uint payload_type(DSL_MSG_PAYLOAD_DEEPSTREAM);
static const std::wstring broker_config_file(
	L"/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/device_client/cfg_azure.txt");


static std::wstring connection_string(
    L"HostName=my-hub.azure-devices.net;DeviceId=1234;SharedAccessKey=abcd"); 
static std::wstring topic(L"DSL_MESSAGE_TOP");


SCENARIO( "The Components container is updated correctly on new Message Sink", "[message-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Message Sink is created" ) 
        {
            REQUIRE( dsl_sink_message_new(sink_name.c_str(), converter_config_file.c_str(),
                payload_type, broker_config_file.c_str(), protocol_lib.c_str(),
                NULL, topic.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Message Sink delete", "[message-sink-api]" )
{
    GIVEN( "A Message Sink Component" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        
        REQUIRE( dsl_sink_message_new(sink_name.c_str(), converter_config_file.c_str(),
            payload_type, broker_config_file.c_str(), protocol_lib.c_str(),
            NULL, topic.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new Message Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Message Sink API verifies all input file pathspecs correctly", "[message-sink-api]" )
{
    GIVEN( "An empty list of components " ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        
        WHEN( "A new Message Sink is created with an invalid Message Converter config pathsepc" ) 
        {
            std::wstring invalid_converter_config_file(L"./invalide/path/spec");
            
            REQUIRE( dsl_sink_message_new(sink_name.c_str(), invalid_converter_config_file.c_str(),
                payload_type, broker_config_file.c_str(), protocol_lib.c_str(),
                NULL, topic.c_str()) == 
                    DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND );
            
            THEN( "The list size remains unchanged" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Message Sink is created with an invalid Message Broker config pathsepc" ) 
        {
            std::wstring invalid_broker_config_file(L"./invalide/path/spec");
            
            REQUIRE( dsl_sink_message_new(sink_name.c_str(), converter_config_file.c_str(),
                payload_type, invalid_broker_config_file.c_str(), protocol_lib.c_str(),
                NULL, topic.c_str()) == 
                    DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND );
            
            THEN( "The list size remains unchanged" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Message Sink is created with an invalid Protocol Synctax Lib pathsepc" ) 
        {
            std::wstring invalid_protocol_lib(L"./invalide/path/spec");
            
            REQUIRE( dsl_sink_message_new(sink_name.c_str(), converter_config_file.c_str(),
                payload_type, invalid_protocol_lib.c_str(), protocol_lib.c_str(),
                NULL, topic.c_str()) == 
                    DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND );
            
            THEN( "The list size remains unchanged" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Message Sink's Broker Settings can be updated", "[message-sink-api]" )
{
    GIVEN( "A Message Sink Component" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        
        REQUIRE( dsl_sink_message_new(sink_name.c_str(), converter_config_file.c_str(),
            payload_type, broker_config_file.c_str(), protocol_lib.c_str(),
            NULL, topic.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* c_ret_broker_config_file;
        const wchar_t* c_ret_protocol_lib;
        const wchar_t* c_ret_connection_string;
        const wchar_t* c_ret_topic;
        
        // confirm the original settings first
        REQUIRE( dsl_sink_message_broker_settings_get(sink_name.c_str(),
            &c_ret_broker_config_file, &c_ret_protocol_lib,
            &c_ret_connection_string, &c_ret_topic) == DSL_RESULT_SUCCESS );
            
        std::wstring ret_broker_config_file(c_ret_broker_config_file);
        std::wstring ret_protocol_lib(c_ret_protocol_lib);
        std::wstring ret_connection_string(c_ret_connection_string);
        std::wstring ret_topic(c_ret_topic);

        REQUIRE( ret_broker_config_file == broker_config_file );
        REQUIRE( ret_protocol_lib == protocol_lib );
        REQUIRE( ret_connection_string == connection_string );
        REQUIRE( ret_topic == topic );
        
        WHEN( "When new Message Converter settings are set" ) 
        {
            std::wstring new_broker_config_file(L"./test/configs/cfg_kafka.txt");
            std::wstring new_protocol_lib(NVDS_AZURE_EDGE_PROTO_LIB);
            std::wstring new_connection_string(
                L"HostName=my-hub.azure-devices.net;DeviceId=6789;SharedAccessKey=efghi");
            std::wstring new_topic;
            
            REQUIRE( dsl_sink_message_broker_settings_set(sink_name.c_str(),
                new_broker_config_file.c_str(), new_protocol_lib.c_str(),
                new_connection_string.c_str(), NULL) == DSL_RESULT_SUCCESS ); // NULL should return empty string
            
            THEN( "The correct settings are returned on get" )
            {
                REQUIRE( dsl_sink_message_broker_settings_get(sink_name.c_str(),
                    &c_ret_broker_config_file, &c_ret_protocol_lib, 
                    &c_ret_connection_string, &c_ret_topic) == DSL_RESULT_SUCCESS );
                    
                ret_broker_config_file.assign(c_ret_broker_config_file);
                ret_protocol_lib.assign(c_ret_protocol_lib);
                ret_connection_string.assign(c_ret_connection_string);
                ret_topic.assign(c_ret_topic);

                REQUIRE( ret_broker_config_file == new_broker_config_file );
                REQUIRE( ret_protocol_lib == new_protocol_lib );
                REQUIRE( ret_connection_string == new_connection_string );
                REQUIRE( ret_topic == new_topic );
                
                REQUIRE( dsl_component_delete(sink_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

