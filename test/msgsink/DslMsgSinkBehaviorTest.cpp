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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"uri-source");
static const std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
static const uint intr_decode(false);
static const uint drop_frame_interval(0); 

static const std::wstring primary_gie_name(L"primary-gie");
static const std::wstring infer_config_file(L"./test/configs/config_infer_primary_nano.txt");
static const std::wstring model_engine_file(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static const std::wstring tracker_name(L"ktl-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring tiler_name1(L"tiler");
static const uint tiler_width(1920);
static const uint tiler_height(1080);

static const std::wstring osd_name(L"on-screen-display");
static const boolean text_enabled(false);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const uint offest_x(100);
static const uint offest_y(140);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"window-sink");

static const std::wstring message_sink_name(L"message-sink");
static const std::wstring converter_config_file(L"./test/configs/dstest4_msgconv_config.txt");
static const uint payload_type(DSL_MSG_PAYLOAD_DEEPSTREAM);
static const std::wstring broker_config_file(L"./test/configs/cfg_azure.txt");
//static std::string connectionString(L"HostName=<my-hub>.azure-devices.net;DeviceId=<device_id>;SharedAccessKey=<my-policy-key>"); 
static std::wstring connection_string(
    L"HostName=my-hub.azure-devices.net;DeviceId=1234;SharedAccessKey=abcd"); 
static std::wstring topic(L"DSL_MESSAGE_TOP");
 

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Tiled Display, Window Sink, and Message Sink can play", "[message-sink-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_message_azure_new(message_sink_name.c_str(), converter_config_file.c_str(),
            payload_type, broker_config_file.c_str(), connection_string.c_str(),
            topic.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source",L"primary-gie", L"tiler", 
            L"window-sink", L"message-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
