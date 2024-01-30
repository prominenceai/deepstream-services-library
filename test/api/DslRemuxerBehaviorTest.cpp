/*
The MIT License

Copyright (c) 2023, Prominence AI, Inc.

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

static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4");

static const std::wstring source_name1(L"file-source-1");
static const std::wstring source_name2(L"file-source-2");
static const std::wstring source_name3(L"file-source-3");
static const std::wstring source_name4(L"file-source-4");

static const std::wstring remuxer_name(L"remuxer");

static const std::wstring branch_name1(L"branch-1");
static const std::wstring branch_name2(L"branch-2");

static const std::wstring primary_gie_name1(L"primary-gie-1");
static const std::wstring primary_gie_name2(L"primary-gie-2");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        
static const std::wstring tracker_name1(L"iou-tracker-1");
static const std::wstring tracker_name2(L"iou-tracker-2");
static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring tiler_name1(L"tiler-1");
static const std::wstring tiler_name2(L"tiler-2");
static const std::wstring tiler_name3(L"tiler-3");
static const std::wstring tiler_name4(L"tiler-4");

static const std::wstring osd_name1(L"osd-1");
static const std::wstring osd_name2(L"osd-2");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const std::wstring sink_name1(L"sink-1");
static const std::wstring sink_name2(L"sink-2");
static const std::wstring sink_name3(L"sink-3");
static const std::wstring sink_name4(L"sink-4");

static const std::wstring ode_pph_name1(L"ode-handler1");
static const std::wstring ode_pph_name2(L"ode-handler2");

static const std::wstring ode_trigger_name1(L"occurrence-1");
static const std::wstring ode_trigger_name2(L"occurrence-2");
static const uint class_id(0);

static const std::wstring ode_action_name1(L"print-1");
static const std::wstring ode_action_name2(L"print-2");

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(640);
static const uint sink_height(360);

// -----------------------------------------------------------------------------------

SCENARIO( "Two File Sources, Remuxer, and two branches with Tilers and Window Sinks can play", 
    "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Dewarper, two Tilers and two Window-Sinks" ) 
    {
        static const uint width(1280);
        static const uint height(360);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name2.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_egl_new(sink_name2.c_str(),
            offest_x+300, offest_y+300, width, height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        const wchar_t* branch1_components[] = {
            tiler_name1.c_str(), sink_name1.c_str(), NULL};

        const wchar_t* branch2_components[] = {
            tiler_name2.c_str(), sink_name2.c_str(), NULL};

        REQUIRE( dsl_branch_new_component_add_many(branch_name1.c_str(), 
            branch1_components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_branch_new_component_add_many(branch_name2.c_str(), 
            branch2_components) == DSL_RESULT_SUCCESS );
        
        const wchar_t* remuxer_branches[] = {
            branch_name1.c_str(), branch_name2.c_str(), NULL};

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tee_branch_add(remuxer_name.c_str(), 
            branch_name1.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_branch_add(remuxer_name.c_str(), 
            branch_name2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), NULL};
//            const wchar_t* components[] = {
//                source_name1.c_str(), 
//                remuxer_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

// -----------------------------------------------------------------------------------

SCENARIO( "Two File Sources, Remuxer, and two branches with Window Sinks, \
each added to a single stream can play", 
    "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Dewarper, and two Window-Sinks" ) 
    {
        static const uint width(640);
        static const uint height(360);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_egl_new(sink_name2.c_str(),
            offest_x+300, offest_y+300, width, height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        uint streamIds1[] = {0};
        uint streamIds2[] = {1};
        
        REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
            sink_name1.c_str(), streamIds1, 1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
            sink_name2.c_str(), streamIds2, 1) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "Two File Sources, Remuxer, and two branches with PGIE, Tracker OSD, and Window Sinks, \
each added to a single stream can play", 
    "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Dewarper, and two Window-Sinks" ) 
    {
        static const uint width(640);
        static const uint height(360);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name1.c_str(), 
            infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name2.c_str(), 
            infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name1.c_str(), NULL,
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name2.c_str(), NULL,
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );


        REQUIRE( dsl_osd_new(osd_name1.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name2.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_ode_new(ode_pph_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_new(ode_pph_name2.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_pph_add(osd_name1.c_str(), ode_pph_name1.c_str(), 
            DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_osd_pph_add(osd_name2.c_str(), ode_pph_name2.c_str(), 
            DSL_PAD_SRC) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(ode_trigger_name1.c_str(), 
            NULL, class_id, 2) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_print_new(ode_action_name1.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(ode_trigger_name1.c_str(), 
            ode_action_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name1.c_str(), 
            ode_trigger_name1.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(ode_trigger_name2.c_str(), 
            NULL, class_id, 2) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_print_new(ode_action_name2.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(ode_trigger_name2.c_str(), 
            ode_action_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name2.c_str(), 
            ode_trigger_name2.c_str()) == DSL_RESULT_SUCCESS );


        REQUIRE( dsl_sink_window_egl_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_egl_new(sink_name2.c_str(),
            offest_x+300, offest_y+300, width, height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        const wchar_t* branch1_components[] = {
            primary_gie_name1.c_str(), tracker_name1.c_str(), 
            osd_name1.c_str(), sink_name1.c_str(), NULL};

        const wchar_t* branch2_components[] = {
            primary_gie_name2.c_str(), tracker_name2.c_str(), 
            osd_name2.c_str(), sink_name2.c_str(), NULL};

        REQUIRE( dsl_branch_new_component_add_many(branch_name1.c_str(), 
            branch1_components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_branch_new_component_add_many(branch_name2.c_str(), 
            branch2_components) == DSL_RESULT_SUCCESS );
        
        const wchar_t* remuxer_branches[] = {
            branch_name1.c_str(), branch_name2.c_str(), NULL};

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        uint streamIds1[] = {0};
        uint streamIds2[] = {1};
        
        REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
            branch_name1.c_str(), streamIds1, 1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
            branch_name2.c_str(), streamIds2, 1) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}
