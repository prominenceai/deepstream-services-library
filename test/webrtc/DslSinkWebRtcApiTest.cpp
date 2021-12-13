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
#include "Dsl.h"
#include "DslApi.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 


static const std::wstring webrtc_sink_name(L"webrtc-sink");
static const std::wstring stun_server(L"stun.l.google.com:19302");
static const std::wstring turn_server;
static uint codec(DSL_CODEC_H264);
static uint bitrate(4000000);
static uint interval(0);

// ---------------------------------------------------------------------------

SCENARIO( "A new WebRTC Sink can be created and deleted successfully", "[webrtc-sink-api]" )
{

    GIVEN( "An empty list of components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When a new WebRTC Sink is created" ) 
        {
            REQUIRE( dsl_sink_webrtc_new(webrtc_sink_name.c_str(),
                stun_server.c_str(), NULL, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

            // Second call with the same name must fail
            REQUIRE( dsl_sink_webrtc_new(webrtc_sink_name.c_str(),
                stun_server.c_str(), NULL, codec, bitrate, interval) == DSL_RESULT_SINK_NAME_NOT_UNIQUE );

            THEN( "The list of components is updated correctly and the same componenct can be deleted" )
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete(webrtc_sink_name.c_str()) == DSL_RESULT_SUCCESS );

                // Second call must fail
                REQUIRE( dsl_component_delete(webrtc_sink_name.c_str()) == DSL_RESULT_COMPONENT_NAME_NOT_FOUND );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new WebRTC Sink can set and get its server properties successfully", "[webrtc-sink-api]" )
{

    GIVEN( "A new WebRTC Sink" ) 
    {
        REQUIRE( dsl_sink_webrtc_new(webrtc_sink_name.c_str(),
            stun_server.c_str(), NULL, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        const wchar_t* c_ret_stun_server;
        const wchar_t* c_ret_turn_server;
        REQUIRE( dsl_sink_webrtc_servers_get(webrtc_sink_name.c_str(),
            &c_ret_stun_server, &c_ret_turn_server) == DSL_RESULT_SUCCESS );

        std::wstring ret_stun_server(c_ret_stun_server);
        std::wstring ret_turn_server(c_ret_turn_server);

        REQUIRE( ret_stun_server == stun_server );
        REQUIRE( ret_turn_server == L"" );


        WHEN( "When the server settings are updated" ) 
        {
            static const std::wstring new_stun_server;
            static const std::wstring new_turn_server(L"turn1.mycompany.net:19302");

            REQUIRE( dsl_sink_webrtc_servers_set(webrtc_sink_name.c_str(),
                new_stun_server.c_str(), new_turn_server.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct settings are returned on get" )
            {
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_sink_webrtc_servers_get(webrtc_sink_name.c_str(),
                    &c_ret_stun_server, &c_ret_turn_server) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_stun_server == new_stun_server );
                REQUIRE( ret_turn_server == new_turn_server );

                REQUIRE( dsl_component_delete(webrtc_sink_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

static void webrtc_sink_client_listener(dsl_webrtc_connection_data* info, 
    void* client_data)
{
    
}

SCENARIO( "A Client Listener can be added and removed from a WebRTC Sink", "[webrtc-sink-api]" )
{
    GIVEN( "A WebRTC Sink in memory" ) 
    {
        REQUIRE( dsl_sink_webrtc_new(webrtc_sink_name.c_str(),
            stun_server.c_str(), NULL, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A client-listener is added" )
        {
            REQUIRE( dsl_sink_webrtc_client_listener_add(webrtc_sink_name.c_str(),
                webrtc_sink_client_listener, NULL) == DSL_RESULT_SUCCESS );

            // Adding the same listener twice must fail.
            REQUIRE( dsl_sink_webrtc_client_listener_add(webrtc_sink_name.c_str(),
                webrtc_sink_client_listener, NULL) == DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_ADD_FAILED );

            THEN( "The same listener can't be added again" ) 
            {
                REQUIRE( dsl_sink_webrtc_client_listener_remove(webrtc_sink_name.c_str(),
                    webrtc_sink_client_listener) == DSL_RESULT_SUCCESS );

                // Second call to remove must fiail
                REQUIRE( dsl_sink_webrtc_client_listener_remove(webrtc_sink_name.c_str(),
                    webrtc_sink_client_listener) == DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
