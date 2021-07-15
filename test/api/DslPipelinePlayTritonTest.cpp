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

static const std::wstring primary_tis_name(L"primary-tis");
static const std::wstring ptis_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt");

static const std::wstring secondary_tis_name(L"secondary-tis");
static const std::wstring stis_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_secondary_plan_engine_carcolor.txt");

static const std::wstring file_path(L"./test/streams/sample_1080p_h264.mp4");

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name(L"file-source");

static const std::wstring ktl_tracker_name(L"ktl-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring tilerName(L"tiler");
static const uint tiler_width(1280);
static const uint tiler_height(720);

static const std::wstring osd_name(L"on-screen-display");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
        
static const std::wstring sink_name(L"overlay-sink");
static const uint display_id(0);
static const uint depth(0);
static const uint offset_x(100);
static const uint offset_y(140);
static const uint sink_width(1280);
static const uint sink_height(720);


SCENARIO( "A new Pipeline with a File Source, Primary TIS, Overlay Sink can play", "[triton-play]" )
{
    GIVEN( "A Pipeline, File source, Primary TIS, Overlay Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_tis_primary_new(primary_tis_name.c_str(), ptis_infer_config_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name.c_str(), display_id, depth,
            offset_x, offset_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", L"primary-tis", L"overlay-sink", NULL};
        
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

SCENARIO( "A new Pipeline with a File Source, Primary TIS, KTL Tracker, OSD, and Overlay Sink can play", "[triton-play]" )
{
    GIVEN( "A Pipeline, File source, Primary TIS, KTL Tracker, OSD, and Overlay Sink" ) 
    {
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_tis_primary_new(primary_tis_name.c_str(), ptis_infer_config_file.c_str(), 0) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_ktl_new(ktl_tracker_name.c_str(), tracker_width, tracker_height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name.c_str(), display_id, depth,
            offset_x, offset_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", L"primary-tis", 
            L"ktl-tracker", L"on-screen-display", L"overlay-sink", NULL};
        
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

SCENARIO( "A new Pipeline with a File Source, Primary TIS, KTL Tracker, Secondary TIS, OSD, and Overlay Sink can play", "[triton-play]" )
{
    GIVEN( "A Pipeline, File source, Primary TIS, KTL Tracker, Secondary TIS, OSD, and Overlay Sink" ) 
    {
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_tis_primary_new(primary_tis_name.c_str(), ptis_infer_config_file.c_str(), 0) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_ktl_new(ktl_tracker_name.c_str(), tracker_width, tracker_height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_tis_secondary_new(secondary_tis_name.c_str(), stis_infer_config_file.c_str(), 
            primary_tis_name.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(sink_name.c_str(), display_id, depth,
            offset_x, offset_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", L"primary-tis", 
            L"ktl-tracker", L"secondary-tis", L"on-screen-display", L"overlay-sink", NULL};
        
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


