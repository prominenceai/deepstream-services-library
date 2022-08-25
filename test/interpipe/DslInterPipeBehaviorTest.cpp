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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name1(L"test-pipeline-1");
static const std::wstring pipeline_name2(L"test-pipeline-2");
static const std::wstring pipeline_name3(L"test-pipeline-3");
static const std::wstring pipeline_name4(L"test-pipeline-4");

static const std::wstring source_name1(L"uri-source-1");
static const std::wstring source_name2(L"uri-source-2");
static const std::wstring source_name3(L"uri-source-3");
static const std::wstring source_name4(L"uri-source-4"); 
static const std::wstring uri1(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov");
static const std::wstring uri2(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_push.mov");
static const std::wstring uri3(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov");
static const std::wstring uri4(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov");
static const uint intr_decode(false);
static const uint drop_frame_interval(0); 

static const std::wstring primary_gie_name1(L"primary-gie-1");
static const std::wstring primary_gie_name2(L"primary-gie-2");

static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
        
static const std::wstring tracker_name1(L"ktl-tracker-1");
static const std::wstring tracker_name2(L"ktl-tracker-2");

static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring tiler_name(L"tiler");
static const uint width(1280);
static const uint height(720);
        
static const std::wstring osd_name1(L"osd-1");
static const std::wstring osd_name2(L"osd-2");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const uint offest_x(100);
static const uint offest_y(140);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name1(L"window-sink-1");
static const std::wstring window_sink_name2(L"window-sink-2");

static const std::wstring inter_pipe_source_name1(L"inter-pipe-source-1");
static const std::wstring inter_pipe_source_name2(L"inter-pipe-source-2");
static const std::wstring inter_pipe_source_name3(L"inter-pipe-source-3");
static const std::wstring inter_pipe_source_name4(L"inter-pipe-source-4");

static const std::wstring inter_pipe_sink_name1(L"inter-pipe-sink-1");
static const std::wstring inter_pipe_sink_name2(L"inter-pipe-sink-2");
static const std::wstring inter_pipe_sink_name3(L"inter-pipe-sink-3");
static const std::wstring inter_pipe_sink_name4(L"inter-pipe-sink-4");

static const std::wstring player_name1(L"player-1");
static const std::wstring player_name2(L"player-2");
static const std::wstring player_name3(L"player-3");
static const std::wstring player_name4(L"player-4");

static const boolean forward_eos(false);
static const boolean forward_events(false);
static const boolean accept_eos(false);
static const boolean accept_events(false);


SCENARIO( "Two Pipelines, one with Inter-Pipe Sink, the other with Inter-Pipe Src, can play ", 
    "[inter-pipe-play]" )
{
    GIVEN( "A two simple pipelines" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri1.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name1.c_str(),
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );

        const wchar_t* components1[] = {source_name1.c_str(), 
            inter_pipe_sink_name1.c_str(), NULL};

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name1.c_str(), 
            inter_pipe_sink_name1.c_str(), false, accept_eos, accept_events) == 
            DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_window_new(window_sink_name1.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components2[] = {inter_pipe_source_name1.c_str(), 
            window_sink_name1.c_str(), NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name1.c_str(), 
                components1) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name2.c_str(), 
                components2) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_play(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );
                
                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_pipeline_state_get(pipeline_name1.c_str(), &currentState) == DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                REQUIRE( dsl_pipeline_state_get(pipeline_name2.c_str(), &currentState) == DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*10);
                
                REQUIRE( dsl_pipeline_stop(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_stop(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Pipeline with and Inter-Pipe Source can dynamically switch between four Inter-Pipe Sinks", 
    "[inter-pipe-play]" )
{
    GIVEN( "Four file sources, four inter-pipe sinks, one inter-pipe source, and one window sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri1.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri2.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name3.c_str(), uri3.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name4.c_str(), uri4.c_str(), 
            true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name1.c_str(),
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name2.c_str(),
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name3.c_str(),
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name4.c_str(),
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name1.c_str(), 
            inter_pipe_sink_name1.c_str(), false, accept_eos, accept_events) == 
            DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name1.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components1[] = {inter_pipe_source_name1.c_str(), 
            window_sink_name1.c_str(), NULL};
        
        WHEN( "When the Players and Pipeline are Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name1.c_str(), 
                components1) == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_player_new(player_name1.c_str(),
                source_name1.c_str(), inter_pipe_sink_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_player_new(player_name2.c_str(),
                source_name2.c_str(), inter_pipe_sink_name2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_player_new(player_name3.c_str(),
                source_name3.c_str(), inter_pipe_sink_name3.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_player_new(player_name4.c_str(),
                source_name4.c_str(), inter_pipe_sink_name4.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Players and Pipeline are Able to LinkAll and Play" )
            {
                REQUIRE( dsl_player_play(player_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_play(player_name2.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_play(player_name3.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_play(player_name4.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_play(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
                
                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_player_state_get(player_name1.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                REQUIRE( dsl_player_state_get(player_name2.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                REQUIRE( dsl_player_state_get(player_name3.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                REQUIRE( dsl_player_state_get(player_name4.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                REQUIRE( dsl_pipeline_state_get(pipeline_name1.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);

                REQUIRE( dsl_source_interpipe_listen_to_set(inter_pipe_source_name1.c_str(),
                    inter_pipe_sink_name2.c_str()) == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);

                REQUIRE( dsl_source_interpipe_listen_to_set(inter_pipe_source_name1.c_str(),
                    inter_pipe_sink_name3.c_str()) == DSL_RESULT_SUCCESS );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);

                REQUIRE( dsl_source_interpipe_listen_to_set(inter_pipe_source_name1.c_str(),
                    inter_pipe_sink_name4.c_str()) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);
                
                REQUIRE( dsl_player_stop(player_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_stop(player_name2.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_stop(player_name3.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_stop(player_name4.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_stop(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Two Pipelines with and Inter-Pipe Sources can listen to a single Inter-Pipe Sinks", 
    "[inter-pipe-play]" )
{
    GIVEN( "One file source, inter-pipe sink, two inter-pipe sources and two window sinks" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri1.c_str(), 
            true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name1.c_str(),
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name1.c_str(), 
            inter_pipe_sink_name1.c_str(), false, accept_eos, accept_events) == 
            DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name2.c_str(), 
            inter_pipe_sink_name1.c_str(), false, accept_eos, accept_events) == 
            DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name1.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 4) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(tracker_name1.c_str(), 
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_osd_new(osd_name1.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name1.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name2.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components1[] = {inter_pipe_source_name1.c_str(), 
            primary_gie_name1.c_str(), tracker_name1.c_str(), osd_name1.c_str(), 
            window_sink_name1.c_str(), NULL};
        
        const wchar_t* components2[] = {inter_pipe_source_name2.c_str(), 
            window_sink_name2.c_str(), NULL};
        
        WHEN( "When the Pipelines and Player are Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name1.c_str(), 
                components1) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name2.c_str(), 
                components2) == DSL_RESULT_SUCCESS );
                
            REQUIRE( dsl_player_new(player_name1.c_str(),
                source_name1.c_str(), inter_pipe_sink_name1.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Pipelines and Player are able to LinkAll and Play" )
            {
                REQUIRE( dsl_player_play(player_name1.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_play(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_play(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );
                
                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_player_state_get(player_name1.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                REQUIRE( dsl_pipeline_state_get(pipeline_name1.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                REQUIRE( dsl_pipeline_state_get(pipeline_name2.c_str(), &currentState) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);

                REQUIRE( dsl_player_stop(player_name1.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_stop(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_stop(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
