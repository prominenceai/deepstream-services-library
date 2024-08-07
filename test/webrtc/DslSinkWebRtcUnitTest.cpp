/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslSinkWebRtcBintr.h"
using namespace DSL;

static std::string sinkName("webrtc-sink");
static std::string stunServer("stun.1.google.com:19302");
static std::string turnServer;
static uint codec(DSL_CODEC_H264);
static uint bitrate(4000000);
static uint interval(0);

static std::string pipelineName("pipeline");

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(30000)

SCENARIO( "A new WebRtcSinkBintr simple test",  "[WebRtcSinkBintr]" )
{
    GIVEN( "Attributes for a new WebRtcSinkBintr" ) 
    {
        WHEN( "The WebRtcSinkBintr is created " )
        {
            DSL_WEBRTC_SINK_PTR pSinkBintr = 
                DSL_WEBRTC_SINK_NEW(sinkName.c_str(), stunServer.c_str(), turnServer.c_str(),
                    codec, 4000000, 0);
            
            THEN( "Run the mail lop indefinetly" )
            {
                GMainLoop *mainloop = g_main_loop_new(NULL, FALSE);
                g_main_loop_run(mainloop);
            }
        }
    }
}      

SCENARIO( "A new WebRtcSinkBintr is created correctly",  "[WebRtcSinkBintr]" )
{
    GIVEN( "Attributes for a new WebRtcSinkBintr" ) 
    {
        WHEN( "The WebRtcSinkBintr is created " )
        {
            DSL_WEBRTC_SINK_PTR pSinkBintr = 
                DSL_WEBRTC_SINK_NEW(sinkName.c_str(), stunServer.c_str(), turnServer.c_str(),
                    codec, 4000000, 0);
            
            THEN( "The correct attribute values are returned" )
            {
                const char* cRetStunServer(NULL); 
                const char* cRetTurnServer(NULL);
                pSinkBintr->GetServers(&cRetStunServer, &cRetTurnServer);
                std::string retStunServer(cRetStunServer);
                std::string retTurnServer(cRetTurnServer);
                REQUIRE( retStunServer == stunServer);
                REQUIRE( retTurnServer == turnServer);

                boolean retEnabled;
                REQUIRE( pSinkBintr->GetSyncEnabled(&retEnabled) == true );
                REQUIRE( retEnabled == true );

                uint retCodec(99), retBitrate(99), retInterval(99);
                pSinkBintr->GetEncoderSettings(&retCodec, &retBitrate, &retInterval);
                REQUIRE( retCodec == codec );
                REQUIRE( retBitrate == bitrate );
                REQUIRE( retInterval == interval );
            }
        }
    }
}

SCENARIO( "A new WebRtcSinkBintr can LinkAll and UnlinkAll Child Elementrs successfully", "[WebRtcSinkBintr]" )
{
    GIVEN( "A new WebRtcSinkBintr in an Unlinked state" ) 
    {
        DSL_WEBRTC_SINK_PTR pSinkBintr = 
            DSL_WEBRTC_SINK_NEW(sinkName.c_str(), stunServer.c_str(), turnServer.c_str(),
                DSL_CODEC_H264, 4000000, 0);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new WebRtcSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The WebRtcSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );

                // Unlink must set linked state
                pSinkBintr->UnlinkAll();
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new WebRtcSinkBintr can be added to and removed from a parent Pipeline successfully", "[WebRtcSinkBintr]" )
{
    GIVEN( "A new WebRtcSinkBintr in an Unlinked state" ) 
    {
        DSL_WEBRTC_SINK_PTR pSinkBintr = 
            DSL_WEBRTC_SINK_NEW(sinkName.c_str(), stunServer.c_str(), turnServer.c_str(),
                DSL_CODEC_H264, 4000000, 0);

        DSL_PIPELINE_PTR pPipeline = DSL_PIPELINE_NEW(pipelineName.c_str());

        WHEN( "The WebRtcSinkBintr is added to a Pipeline" )
        {
            REQUIRE( pSinkBintr->AddToParent(pPipeline) == true );

            THEN( "Only the WebRtcSinkBintr's Fake Sink is added to the Pipeline and not the actual WebRtcSinkBintr" )
            {
                REQUIRE( pSinkBintr->IsParent(pPipeline) == false );
                REQUIRE( pSinkBintr->RemoveFromParent(pPipeline) == true );
            }
        }
    }
}
