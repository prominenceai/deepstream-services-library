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
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static const uint intr_decode(false);
static const uint drop_frame_interval(0); 

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static const std::wstring tracker_name(L"iou-tracker");
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
static const std::wstring converter_config_file(
	L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.txt");
static const std::wstring protocol_lib(NVDS_AZURE_PROTO_LIB);
static const uint payload_type(DSL_MSG_PAYLOAD_DEEPSTREAM);
static const std::wstring broker_config_file(
	L"/opt/nvidia/deepstream/deepstream/sources/libs/azure_protocol_adaptor/device_client/cfg_azure.txt");

static std::wstring topic(L"DSL_MESSAGE_TOPIC");

static const std::wstring ode_handler_name(L"ode-handler");
static const std::wstring occurrence_trigger_name(L"occurrence-trigger");
static const std::wstring message_action_name(L"message-action");
static const std::wstring print_action_name(L"print-action");

// ---------------------------------------------------------------------------

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Tiled Display, Window Sink, and Message Sink can play", "[message-sink-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Tiled Display, Window Sink, Message Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_message_new(message_sink_name.c_str(), converter_config_file.c_str(),
            payload_type, broker_config_file.c_str(), protocol_lib.c_str(),
            NULL, topic.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false)  == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_action_message_meta_add_new(message_action_name.c_str()) == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_occurrence_new(occurrence_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_ONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            message_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            occurrence_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
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

                dsl_delete_all();
            }
        }
    }
}

