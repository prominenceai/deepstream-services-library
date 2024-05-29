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


static std::wstring sink_name(L"livekit-webrtc-sink");
static std::wstring url(L"ws://127.0.0.1:7880");
static std::wstring api_key(L"ZDUzNnZrZDgxYWNhc3BpcDJwMmE2NWNuMG46MjJkMTBiNGYtODA4Zi00OWRhLWEzMTQtOGM4MDhkOTdiYTU3");
static std::wstring secret_key(L"secret");
static std::wstring room(L"testroom");
static std::wstring identity(L"dsl_pipeline");
static std::wstring participant(L"dsl_pipeline");

// ---------------------------------------------------------------------------

SCENARIO( "A new Live Kit WebRTC Sink can be created and deleted successfully", "[livekit-webrtc-sink-api]" )
{

    GIVEN( "An empty list of components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When a new Live Kit WebRTC Sink is created" ) 
        {
            REQUIRE( dsl_sink_webrtc_livekit_new(sink_name.c_str(),
                url.c_str(), api_key.c_str(), secret_key.c_str(), 
                room.c_str(), identity.c_str(), participant.c_str())
                    == DSL_RESULT_SUCCESS );

            // Second call with the same name must fail
            REQUIRE( dsl_sink_webrtc_livekit_new(sink_name.c_str(),
                url.c_str(), api_key.c_str(), secret_key.c_str(), 
                room.c_str(), identity.c_str(), participant.c_str())
                    == DSL_RESULT_SINK_NAME_NOT_UNIQUE );

            THEN( "The list of components is updated correctly and the same component can be deleted" )
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete(sink_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                // Second call must fail
                REQUIRE( dsl_component_delete(sink_name.c_str()) 
                    == DSL_RESULT_COMPONENT_NAME_NOT_FOUND );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
