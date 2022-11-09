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

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static const std::wstring secondary_gie_name(L"secondary-gie");
static const std::wstring sgie_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carcolor.txt");
static const std::wstring sgie_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine");


static const std::wstring primary_tis_name(L"primary-tis");
static const std::wstring ptis_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt");

static const std::wstring secondary_tis_name(L"secondary-tis");
static const std::wstring stis_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_secondary_plan_engine_carcolor.txt");

    
static const std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

static const std::wstring dcf_perf_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml");

static const std::wstring file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"file-source-1");
static const std::wstring source_name2(L"file-source-2");
static const std::wstring source_name3(L"file-source-3");
static const std::wstring source_name4(L"file-source-4");

static const std::wstring dcf_tracker_name(L"dcf-tracker");
static const uint tracker_width(640);
static const uint tracker_height(384);

static const std::wstring tiler_name(L"tiler");
static const uint tiler_width(1280);
static const uint tiler_height(720);

static const std::wstring osd_name(L"on-screen-display");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);
        
static const std::wstring sink_name(L"window-sink");
static const uint offset_x(100);
static const uint offset_y(140);
static const uint sink_width(1280);
static const uint sink_height(720);


SCENARIO( "A new Pipeline with a Primary GIE, DCF Tracker with its Batch Processing \
    and Past Frame Reporting enabled", "[now]" )
{
    GIVEN( "A Pipeline, File source, Primary GIE, KTL Tracker, OSD, and Overlay Sink" ) 
    {
        
        boolean inference_interval(4);
        boolean batch_processing_enabled(true);
        boolean past_frame_reporting_enabled(true);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name2.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name3.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name4.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(dcf_tracker_name.c_str(), 
            dcf_perf_tracker_config_file.c_str(), tracker_width, tracker_height)
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_batch_processing_enabled_set(dcf_tracker_name.c_str(), 
            batch_processing_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_past_frame_reporting_enabled_set(dcf_tracker_name.c_str(), 
            past_frame_reporting_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(sink_name.c_str(),
            offset_x, offset_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source-1", L"file-source-2", 
            L"file-source-3", L"file-source-4", L"primary-gie", 
            L"dcf-tracker", L"tiler", L"on-screen-display", L"window-sink", NULL};
        
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

SCENARIO( "A new Pipeline with a Primary GIE, DCF Tracker with its Batch Processing and \
    Past Frame Reporting disabled", "[tracker-play]" )
{
    GIVEN( "A Pipeline, File source, Primary GIE, KTL Tracker, OSD, and Overlay Sink" ) 
    {
        boolean inference_interval(4);
        boolean batch_processing_enabled(false);
        boolean past_frame_reporting_enabled(false);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(dcf_tracker_name.c_str(), 
            dcf_perf_tracker_config_file.c_str(), tracker_width, tracker_height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_batch_processing_enabled_set(dcf_tracker_name.c_str(), 
            batch_processing_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_past_frame_reporting_enabled_set(dcf_tracker_name.c_str(), 
            past_frame_reporting_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(sink_name.c_str(),
            offset_x, offset_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source-1", L"primary-gie", 
            L"dcf-tracker", L"on-screen-display", L"window-sink", NULL};
        
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

SCENARIO( "A new Pipeline with a Primary GIE, IOU Tracker and optional \
    config file", "[tracker-play]" )
{
    GIVEN( "A Pipeline, File source, Primary GIE, IOU Tracker, OSD, and Overlay Sink" ) 
    {
        boolean inference_interval(4);
        boolean batch_processing_enabled(true);
        boolean past_frame_reporting_enabled(true);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name2.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name3.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_file_new(source_name4.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(dcf_tracker_name.c_str(), 
            iou_tracker_config_file.c_str(), tracker_width, tracker_height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(sink_name.c_str(),
            offset_x, offset_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source-1", L"file-source-2", 
            L"file-source-3", L"file-source-4", L"primary-gie", 
            L"dcf-tracker", L"tiler", L"on-screen-display", L"window-sink", NULL};
        
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

