/*
The MIT License

Copyright (c) 2023-2024, Prominence AI, Inc.

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

static const std::wstring osd_name1(L"osd-1");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const std::wstring sink_name1(L"sink-1");

static const std::wstring ode_pph_name(L"ode-handler");

static const std::wstring ode_action_name(L"print");

static const std::wstring ode_trigger_name(L"occurrence");
static const uint class_id(0);
static const uint limit_10(10);

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(640);
static const uint sink_height(360);

// -----------------------------------------------------------------------------------

SCENARIO( "Two File Sources, Remuxer with and two PGIE branches, a Tiler and Window Sink can play", 
    "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Remuxer, two PGIEs, Tiler, and Window-Sink" ) 
    {
        uint width(1280);
        uint height(360);
        
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

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name1.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        const wchar_t* remuxer_branches[] = {
            primary_gie_name1.c_str(), primary_gie_name2.c_str(), NULL};

        REQUIRE( dsl_remuxer_new_branch_add_many(remuxer_name.c_str(),
            remuxer_branches) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), tiler_name1.c_str(), osd_name1.c_str(),
                sink_name1.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_link_method_set(pipeline_name.c_str(),
                    DSL_PIPELINE_LINK_METHOD_BY_POSITION) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), tiler_name1.c_str(), osd_name1.c_str(),
                sink_name1.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_link_method_set(pipeline_name.c_str(),
                    DSL_PIPELINE_LINK_METHOD_BY_ADD_ORDER) == DSL_RESULT_SUCCESS );
                
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

SCENARIO( "Two File Sources, Remuxer with and two PGIE branches each added to a single \
stream, Tiler, and Window Sink can play", "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Remuxer, two PGIEs, Tiler, and Window-Sink" ) 
    {
        uint width(1280);
        uint height(360);
        
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

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name1.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
            
        std::vector<uint> streamIds1 = {1};
        std::vector<uint> streamIds2 = {0};
        
        REQUIRE( dsl_remuxer_branch_add_to(remuxer_name.c_str(), 
            primary_gie_name1.c_str(), &streamIds1[0], streamIds1.size()) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_remuxer_branch_add_to(remuxer_name.c_str(), 
            primary_gie_name2.c_str(), &streamIds2[0], streamIds2.size()) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_remuxer_pph_add(remuxer_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(ode_trigger_name.c_str(), 
            source_name2.c_str(), class_id, DSL_ODE_TRIGGER_LIMIT_NONE) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_limit_frame_set(ode_trigger_name.c_str(), 
            2) == DSL_RESULT_SUCCESS );

            
        REQUIRE( dsl_ode_action_print_new(ode_action_name.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(ode_trigger_name.c_str(), 
            ode_action_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_ode_trigger_add(ode_pph_name.c_str(), 
            ode_trigger_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), tiler_name1.c_str(), osd_name1.c_str(),
                sink_name1.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_link_method_set(pipeline_name.c_str(),
                    DSL_PIPELINE_LINK_METHOD_BY_POSITION) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), L"state-playing");
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), tiler_name1.c_str(), osd_name1.c_str(),
                sink_name1.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_link_method_set(pipeline_name.c_str(),
                    DSL_PIPELINE_LINK_METHOD_BY_ADD_ORDER) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), L"state-playing");
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

// -----------------------------------------------------------------------------------

SCENARIO( "Four File Sources, Remuxer with and two PGIE branches each added to a single \
stream, Tiler, and Window Sink can play", "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Remuxer, two PGIEs, Tiler, and Window-Sink" ) 
    {
        uint width(1280);
        uint height(720);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name3.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name4.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name1.c_str(), 
            infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name2.c_str(), 
            infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name1.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
            
        std::vector<uint> streamIds1 = {1};
        std::vector<uint> streamIds2 = {1,2,3};
        
        REQUIRE( dsl_remuxer_branch_add_to(remuxer_name.c_str(), 
            primary_gie_name1.c_str(), &streamIds1[0], streamIds1.size()) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_remuxer_branch_add_to(remuxer_name.c_str(), 
            primary_gie_name2.c_str(), &streamIds2[0], streamIds2.size()) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_ode_new(ode_pph_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_remuxer_pph_add(remuxer_name.c_str(), 
            ode_pph_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(ode_trigger_name.c_str(), 
            source_name2.c_str(), class_id, DSL_ODE_TRIGGER_LIMIT_NONE) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_limit_frame_set(ode_trigger_name.c_str(), 
            2) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_action_print_new(ode_action_name.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(ode_trigger_name.c_str(), 
            ode_action_name.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_color_custom_new(L"full-white", 
            1.0, 1.0, 1.0, 1.0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_font_new(L"arial-14-white", 
            L"arial", 14, L"full-white") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_source_stream_id_new(L"source-stream-id", 
            15, 20, L"arial-14-white", False, 
            L"full-white") == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_display_meta_add_new(L"add-source-info", 
            L"source-stream-id") == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_ode_trigger_always_new(L"always-trigger", 
            DSL_ODE_ANY_SOURCE, DSL_ODE_PRE_OCCURRENCE_CHECK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_action_add(L"always-trigger", 
            L"add-source-info") == DSL_RESULT_SUCCESS );
            
        const wchar_t* triggers[] = {
            L"always-trigger", ode_trigger_name.c_str(), NULL};
            
        REQUIRE( dsl_pph_ode_trigger_add_many(ode_pph_name.c_str(), 
            triggers) == DSL_RESULT_SUCCESS );


        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                source_name3.c_str(), source_name4.c_str(), 
                remuxer_name.c_str(), tiler_name1.c_str(), osd_name1.c_str(),
                sink_name1.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_link_method_set(pipeline_name.c_str(),
                    DSL_PIPELINE_LINK_METHOD_BY_POSITION) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), L"state-playing");
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                source_name3.c_str(), source_name4.c_str(), 
                remuxer_name.c_str(), tiler_name1.c_str(), osd_name1.c_str(),
                sink_name1.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_link_method_set(pipeline_name.c_str(),
                    DSL_PIPELINE_LINK_METHOD_BY_POSITION) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), L"state-playing");
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

