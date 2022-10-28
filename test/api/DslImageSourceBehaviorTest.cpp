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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(3000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name(L"test-pipeline");
static const std::wstring player_name(L"test-player");

static const std::wstring source_name1(L"image-source-1");
static const std::wstring source_name2(L"image-source-2");
static const std::wstring source_name3(L"image-source-3");
static const std::wstring source_name4(L"image-source-4");
static const std::wstring jpeg_file_path(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");
static const std::wstring jpeg_file_path_multi(L"./test/streams/sample_720p.%d.jpg");
static const std::wstring mjpeg_file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mjpeg");
static const std::wstring mjpeg_file_path_multi(
    L"./test/streams/sample_720p.%d.mjpeg");

static const std::wstring png_file_path(L"./test/streams/sample_720p.png");

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

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


static const std::wstring ode_handler_name(L"ode-handler");
static const std::wstring occurrence_trigger_name(L"occurrence-trigger");
static const std::wstring summation_trigger_name(L"summation-trigger");

static const std::wstring print_action_name(L"print-action");

static const std::wstring pipeline_graph_name(L"pipeline-playing");

// ---------------------------------------------------------------------------

// 
// Function to be called on End-of-Stream (EOS) event
// 
static void eos_event_listener(void* client_data)
{
    std::wcout << L"EOS event for Pipeline " << std::endl;
    dsl_main_loop_quit();
}    

// ---------------------------------------------------------------------------

SCENARIO( "A new Pipeline with a JPEG Image Source, Primary GIE, Tiled Display, \
    Window Sink, ODE Trigger and Action can play",
    "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Tiled Display, Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_new(source_name1.c_str(),
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(),
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false) == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_occurrence_new(occurrence_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_ONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            occurrence_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source-1",L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
                eos_event_listener, NULL) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_main_loop_run();
                
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A new Pipeline with an MJPEG Image Frame Source, Primary GIE, Tiled Display, \
    Window Sink, ODE Trigger and Action can play",
    "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Tiled Display, Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_new(source_name1.c_str(),
            mjpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(),
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false) == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_occurrence_new(occurrence_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_ONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            occurrence_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source-1",L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

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

SCENARIO( "A new Pipeline with 4 JPEG Image Sources, Primary GIE, \
    Tiled Display, Window Sink can play", "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Tiled Display, Window Sink" ) 
    {
        uint fps_n(10), fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_new(source_name1.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_image_new(source_name2.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_image_new(source_name3.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_image_new(source_name4.c_str(), 
            jpeg_file_path.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false)  == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_occurrence_new(occurrence_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            occurrence_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"image-source-1", L"image-source-2", 
            L"image-source-3", L"image-source-4", L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
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

// ---------------------------------------------------------------------------

SCENARIO( "A new Pipeline with a Image Stream Source, Primary GIE, Tiled Display, \
    Window Sink, ODE Trigger and Action can play",
    "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Tiled Display, Window Sink" ) 
    {
        uint fps_n(10), fps_d(1);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_stream_new(source_name1.c_str(),
            jpeg_file_path.c_str(), false, fps_n, fps_d, 1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(),
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false) == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_occurrence_new(occurrence_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_ONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            occurrence_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source-1",L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
                eos_event_listener, NULL) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                    const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                dsl_main_loop_run();
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A new Pipeline with a Multi Image Source, Primary GIE, Tiled Display, \
    Window Sink, ODE Trigger and Action can play",
    "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Window Sink" ) 
    {
        uint fps_n(1), fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_multi_new(source_name1.c_str(), 
            jpeg_file_path_multi.c_str(), fps_n, fps_d) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false)  == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_summation_new(summation_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(summation_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            summation_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source-1",L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
                eos_event_listener, NULL) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                    const_cast<wchar_t*>(pipeline_graph_name.c_str()));
                dsl_main_loop_run();
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A new Pipeline with a Multi Image Source with start and stop indices can play",
    "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Window Sink" ) 
    {
        uint fps_n(1), fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_multi_new(source_name1.c_str(), 
            jpeg_file_path_multi.c_str(), fps_n, fps_d) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_source_image_multi_indices_set(source_name1.c_str(), 
            1, 4) == DSL_RESULT_SUCCESS );
            

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false)  == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_summation_new(summation_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(summation_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            summation_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source-1",L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
                eos_event_listener, NULL) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                    const_cast<wchar_t*>(pipeline_graph_name.c_str()));
                dsl_main_loop_run();
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A new Pipeline with a Multi MJPEG Image Frame Source, Primary GIE, Tiled Display, \
    Window Sink, ODE Trigger and Action can play",
    "[image-source-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Tiled Display, Window Sink" ) 
    {
        uint fps_n(1), fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_multi_new(source_name1.c_str(), 
            mjpeg_file_path_multi.c_str(), fps_n, fps_d) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(print_action_name.c_str(), false)  == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_occurrence_new(occurrence_trigger_name.c_str(),
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_ONE) ==
            DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(occurrence_trigger_name.c_str(), 
            print_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pph_ode_new(ode_handler_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_handler_name.c_str(), 
            occurrence_trigger_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), ode_handler_name.c_str(), 
            DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source-1",L"primary-gie", L"tiler", 
            L"window-sink", NULL};
        
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*3);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

//SCENARIO( "A new Player with a Multi JPEG Image Source and Window Sink can play",
//    "[temp]" )
//{
//    GIVEN( "A Player with a Multi JPEG Source, Window Sink" ) 
//    {
//        uint fps_n(1), fps_d(1);
//
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_image_multi_new(source_name1.c_str(), 
//            jpeg_file_path_multi.c_str(), fps_n, fps_d) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
//            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
//
//        
//        const wchar_t* components[] = {L"image-source-1",
//            L"window-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_player_new(player_name.c_str(), source_name1.c_str(),
//                window_sink_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_player_termination_event_listener_add(player_name.c_str(), 
//                eos_event_listener, NULL) == DSL_RESULT_SUCCESS );
//                
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
//                dsl_main_loop_run();
//                REQUIRE( dsl_player_stop(player_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                dsl_delete_all();
//            }
//        }
//    }
//}

