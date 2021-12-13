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

SCENARIO( "The Websocket Server can listen on and disconnect from a specific port number", "[websocket-server-api]" )
{
    GIVEN( "The Websocket Server singlton" )
    {
        uint initial_port_number(99);
        boolean is_listening(true);

        // test the default state first
        REQUIRE( dsl_websocket_server_listening_state_get(&is_listening, 
            &initial_port_number) == DSL_RESULT_SUCCESS );
        REQUIRE( is_listening == false );
        REQUIRE( initial_port_number == 0 );

        WHEN( "The Websocket Server starts listening on a specified port" )
        {
            uint new_port_number(DSL_WEBSOCKET_SERVER_DEFAULT_WEBSOCKET_PORT);
            uint ret_port_number(0);
            REQUIRE( dsl_websocket_server_listening_start(new_port_number) 
                == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_websocket_server_listening_state_get(&is_listening, 
                &ret_port_number) == DSL_RESULT_SUCCESS );
            REQUIRE( is_listening == true );
            REQUIRE( ret_port_number == new_port_number );

            // Second call to start when already listening must fail
            REQUIRE( dsl_websocket_server_listening_start(new_port_number) 
                == DSL_RESULT_WEBSOCKET_SERVER_SET_FAILED );

            THEN( "The Websocket Server can correctly disconnect." )
            {
                REQUIRE( dsl_websocket_server_listening_stop() == DSL_RESULT_SUCCESS );

                // Second call with the same Signaling Transceiver must fail
                REQUIRE( dsl_websocket_server_listening_stop() == 
                    DSL_RESULT_WEBSOCKET_SERVER_SET_FAILED );

                // Listening state and port must be set back to default.    
                REQUIRE( dsl_websocket_server_listening_state_get(&is_listening, 
                    &ret_port_number) == DSL_RESULT_SUCCESS );
                REQUIRE( is_listening == false );
                REQUIRE( ret_port_number == 0 );
            }
        }
    }
}

SCENARIO( "The Websocket Server can add a handler for a new a Path", "[websocket-server-api]" )
{
    GIVEN( "The singleton Websocet server" )
    {
        uint initial_port_number(99);
        boolean is_listening(true);

        // test the default state first
        REQUIRE( dsl_websocket_server_listening_state_get(&is_listening, 
            &initial_port_number) == DSL_RESULT_SUCCESS );
        REQUIRE( is_listening == false );
        REQUIRE( initial_port_number == 0 );

        WHEN( "A new Path is added to the Soup Server Manager" )
        {
            std::wstring new_path(L"/ws/new_path");
            REQUIRE( dsl_websocket_server_path_add(new_path.c_str()) == true );

            THEN( "The Soup Server Manager starts and stops listening correctly." )
            {
                uint new_port_number(DSL_WEBSOCKET_SERVER_DEFAULT_WEBSOCKET_PORT);
                uint ret_port_number(0);
                REQUIRE( dsl_websocket_server_listening_start(new_port_number) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_websocket_server_listening_state_get(&is_listening, 
                    &ret_port_number) == DSL_RESULT_SUCCESS );
                REQUIRE( is_listening == true );
                REQUIRE( ret_port_number == new_port_number );

                REQUIRE( dsl_websocket_server_listening_stop() == DSL_RESULT_SUCCESS );

                // Listening state and port must be set back to default.    
                REQUIRE( dsl_websocket_server_listening_state_get(&is_listening, 
                    &ret_port_number) == DSL_RESULT_SUCCESS );
                REQUIRE( is_listening == false );
                REQUIRE( ret_port_number == 0 );
            }
        }
    }
}

