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

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name(L"file-source");
static const std::wstring file_path(L"./test/streams/sample_1080p_h264.mp4");

static const std::wstring window_sink_name(L"window-sink");
static const uint offset_x(100);
static const uint offset_y(140);
static const uint sink_width(320);
static const uint sink_height(180);

static const std::wstring webrtc_sink_name(L"webrtc-sink");
static const std::wstring stun_server(L"stun.l.google.com:19302");
static uint codec(DSL_CODEC_H264);
static uint bitrate(4000000);
static uint interval(0);

// ---------------------------------------------------------------------------

SCENARIO( "A new Pipeline with a File Source and WebRTC Sink can play without a connection", "[webrtc-sink-play]" )
{

    GIVEN( "A Pipeline, File source and WebRTC Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_webrtc_new(webrtc_sink_name.c_str(),
            stun_server.c_str(), NULL, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", L"webrtc-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


