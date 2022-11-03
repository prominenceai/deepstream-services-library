/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(500)

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"source-1");
static const std::wstring source_name2(L"source-2");
static const std::wstring source_name3(L"source-3");
static const std::wstring source_name4(L"source-4");

static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

static const std::wstring mov_uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov");
static const uint intr_decode(false);
static const uint drop_frame_interval(0); 

static const std::wstring jpeg_file_path(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static const std::wstring tracker_name(L"ktl-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring osd_name(L"on-screen-display");
static const boolean text_enabled(false);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const std::wstring tiler_name(L"tiler");

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"window-sink");

static std::wstring custom_ppm_name1(L"custom-ppm-1");
static std::wstring custom_ppm_name2(L"custom-ppm-2");
static std::wstring custom_ppm_name3(L"custom-ppm-3");
static std::wstring custom_ppm_name4(L"custom-ppm-4");

static std::wstring buffer_timeout_name_1(L"buffer-timeout-ppm-1");
static std::wstring buffer_timeout_name_2(L"buffer-timeout-ppm-2");
static std::wstring buffer_timeout_name_3(L"buffer-timeout-ppm-3");
static std::wstring buffer_timeout_name_4(L"buffer-timeout-ppm-4");

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    std::cout << "Custom Pad Probe Handler callback #1 called " << std::endl;
    return DSL_PAD_PROBE_OK;
}
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
    std::cout << "Custom Pad Probe Handler callback #2 called " << std::endl;
    return DSL_PAD_PROBE_OK;
}
static boolean pad_probe_handler_cb3(void* buffer, void* user_data)
{
    std::cout << "Custom Pad Probe Handler callback #3 called " << std::endl;
    return DSL_PAD_PROBE_OK;
}


static boolean pad_probe_handler_cb4(void* buffer, void* user_data)
{
    std::cout << "Custom Pad Probe Handler callback #4 called " << std::endl;
    return DSL_PAD_PROBE_REMOVE;
}

SCENARIO( "Multiple Custom PPHs are called in the correct add order", "[pph-behavior]" )
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

        const wchar_t* components[] = {L"source-1",L"primary-gie", L"window-sink", NULL};
        
        REQUIRE( dsl_pph_custom_new(custom_ppm_name1.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_custom_new(custom_ppm_name2.c_str(), 
            pad_probe_handler_cb2, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_custom_new(custom_ppm_name3.c_str(), 
            pad_probe_handler_cb3, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_primary_pph_add(primary_gie_name.c_str(), 
            custom_ppm_name3.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_infer_primary_pph_add(primary_gie_name.c_str(), 
            custom_ppm_name1.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_infer_primary_pph_add(primary_gie_name.c_str(), 
            custom_ppm_name2.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                // Note: requires visual verification for execution order
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

SCENARIO( "A Custom PPH can remove be removed on return", "[pph-behavior]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"source-1",L"primary-gie", L"window-sink", NULL};
        
        REQUIRE( dsl_pph_custom_new(custom_ppm_name4.c_str(), 
            pad_probe_handler_cb4, NULL) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_primary_pph_add(primary_gie_name.c_str(), 
            custom_ppm_name4.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                // Note: requires visual verification for single call and removal.
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

static void buffer_timeout_handler_cb(uint timeout, void* client_data)
{
    static uint count(0);
    
    std::cout << "Buffer Timeout Handeler called with timeout = " 
        << timeout << std::endl;
        
    if (++count >=3)
    {
        dsl_main_loop_quit();
    }
}

SCENARIO( "A Buffer Timeout PPH calls its handler function correctly ", "[temp]" )
{
    GIVEN( "A Pipeline, four images source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_new(source_name1.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_buffer_timeout_new(buffer_timeout_name_1.c_str(), 
            1, buffer_timeout_handler_cb, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_pph_add(source_name1.c_str(), 
            buffer_timeout_name_1.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_image_new(source_name2.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_buffer_timeout_new(buffer_timeout_name_2.c_str(), 
            1, buffer_timeout_handler_cb, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_pph_add(source_name2.c_str(), 
            buffer_timeout_name_2.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_image_new(source_name3.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_buffer_timeout_new(buffer_timeout_name_3.c_str(), 
            1, buffer_timeout_handler_cb, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_pph_add(source_name3.c_str(), 
            buffer_timeout_name_3.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_image_new(source_name4.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_buffer_timeout_new(buffer_timeout_name_4.c_str(), 
            1, buffer_timeout_handler_cb, NULL) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_pph_add(source_name4.c_str(), 
            buffer_timeout_name_4.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(),
            DSL_STREAMMUX_DEFAULT_WIDTH, DSL_STREAMMUX_DEFAULT_HEIGHT) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"source-1", L"source-2", L"source-3",
            L"source-4", L"tiler", L"window-sink", NULL};
        
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                // Note: requires visual verification for single call and removal.
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_main_loop_run();
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                boolean enabled(true);
                REQUIRE( dsl_pph_enabled_get(buffer_timeout_name_1.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == false);
                enabled = true;
                REQUIRE( dsl_pph_enabled_get(buffer_timeout_name_2.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == false);
                enabled = true;
                REQUIRE( dsl_pph_enabled_get(buffer_timeout_name_3.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == false);
                enabled = true;
                REQUIRE( dsl_pph_enabled_get(buffer_timeout_name_4.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == false);

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


