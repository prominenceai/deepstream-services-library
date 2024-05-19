/*
The MIT License

Copyright (c) 2021-2024, Prominence AI, Inc.

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
#include "DslPipelineBintr.h"
#include "DslSinkBintr.h"
using namespace DSL;

static std::string sinkName("livekit-webrtc-sink");
static std::string url("ws://127.0.0.1:7880");
static std::string apiKey("ZDUzNnZrZDgxYWNhc3BpcDJwMmE2NWNuMG46MjJkMTBiNGYtODA4Zi00OWRhLWEzMTQtOGM4MDhkOTdiYTU3");
static std::string secretKey("secret");
static std::string room("testroom");
static std::string identity("dsl_pipeline");
static std::string participant("dsl_pipeline");

static std::string pipelineName("pipeline");

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(30000)


SCENARIO( "A new LiveKitWebRtcSinkBintr is created correctly",  
    "[LiveKitWebRtcSinkBintr]" )
{
    GIVEN( "Attributes for a new LiveKitWebRtcSinkBintr" ) 
    {
        WHEN( "The LiveKitWebRtcSinkBintr is created " )
        {
            DSL_LIVEKIT_WEBRTC_SINK_PTR pSinkBintr = 
                DSL_LIVEKIT_WEBRTC_SINK_NEW(sinkName.c_str(), url.c_str(), apiKey.c_str(),
                    secretKey.c_str(), room.c_str(), identity.c_str(), participant.c_str());
            
            THEN( "The correct attribute values are returned" )
            {
            }
        }
    }
}

SCENARIO( "A new LiveKitWebRtcSinkBintr can LinkAll and UnlinkAll Child Elementrs successfully", 
    "[LiveKitWebRtcSinkBintr]" )
{
    GIVEN( "A new LiveKitWebRtcSinkBintr in an Unlinked state" ) 
    {
        DSL_LIVEKIT_WEBRTC_SINK_PTR pSinkBintr = 
            DSL_LIVEKIT_WEBRTC_SINK_NEW(sinkName.c_str(), url.c_str(), apiKey.c_str(),
                secretKey.c_str(), room.c_str(), identity.c_str(), participant.c_str());

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new LiveKitWebRtcSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The LiveKitWebRtcSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );

                // Unlink must set linked state
                pSinkBintr->UnlinkAll();
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

