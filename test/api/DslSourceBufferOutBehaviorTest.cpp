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

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"uri-source-1");
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static const uint skip_frames(0);
static const uint drop_frame_interval(0); 

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

static const std::wstring tracker_name(L"iou-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

static const std::wstring secondary_gie_name1(L"secondary-gie-1");
static const std::wstring sgie_infer_config_file1(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
static const std::wstring sgie_model_engine_file1(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet.etlt_b8_gpu0_int8.engine");

static const std::wstring tiler_name1(L"tiler-1");

static const uint offest_x(0);
static const uint offest_y(0);
static const uint tiler_width(1280);
static const uint tiler_height(720);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring tiler_name(L"tiler");

static const std::wstring osd_name(L"on-screen-display");
static const boolean text_enabled(true);
static const boolean clock_enabled(true);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const std::wstring window_sink_name(L"egl-sink");

SCENARIO( "A URI File Source can play with buffer-out-format = RGBA]",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, PGIE, Tracker, SGIE, OSD, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            0) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name1.c_str(), 
            sgie_infer_config_file1.c_str(), sgie_model_engine_file1.c_str(), 
                primary_gie_name.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(),
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1", L"primary-gie", L"iou-tracker", 
            L"secondary-gie-1", L"tiler-1", L"on-screen-display", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-format is set to RGBA" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_format_set(source_name1.c_str(),
                DSL_VIDEO_FORMAT_RGBA) == DSL_RESULT_SUCCESS );
                
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

SCENARIO( "A URI File Source can play with buffer-out-format = NV12",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-format is set to NV12" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_format_set(source_name1.c_str(),
                DSL_VIDEO_FORMAT_NV12) == DSL_RESULT_SUCCESS );
                
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

SCENARIO( "A URI File Source can play with scaled-down frame-rate",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-frame-rate is set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_frame_rate_set(source_name1.c_str(),
                2, 1) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A URI File Source can play with buffer-out-orientation = \
DSL_VIDEO_ORIENTATION_FLIP_UPPER_LEFT_TO_LOWER_RIGHT]", "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-orientation is set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_orientation_set(source_name1.c_str(),
                DSL_VIDEO_ORIENTATION_FLIP_UPPER_LEFT_TO_LOWER_RIGHT) == DSL_RESULT_SUCCESS );
                
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

SCENARIO( "A URI File Source can play with buffer-out-crop-pre set]", 
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-crop settings are set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_crop_rectangle_set(source_name1.c_str(),
                DSL_VIDEO_CROP_AT_SRC, 10, 10, 200, 200) == DSL_RESULT_SUCCESS );
                
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

SCENARIO( "A URI File Source can play with buffer-out-crop-post set]", 
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-crop settings are set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_crop_rectangle_set(source_name1.c_str(),
                DSL_VIDEO_CROP_AT_DEST, 1000, 1000, 200, 200) == DSL_RESULT_SUCCESS );
                
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

SCENARIO( "A URI File Source with three Duplicate Sources can play",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, 3 Duplicate Sources, Tiler, and Window Sink" ) 
    {
        std::wstring duplicate_source_1(L"duplicate-source-1");
        std::wstring duplicate_source_2(L"duplicate-source-2");
        std::wstring duplicate_source_3(L"duplicate-source-3");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_duplicate_new(duplicate_source_1.c_str(),
            source_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_video_buffer_out_crop_rectangle_set(
            duplicate_source_1.c_str(), DSL_VIDEO_CROP_AT_SRC, 
            480, 270, 960, 540) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_duplicate_new(duplicate_source_2.c_str(),
            source_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_video_buffer_out_frame_rate_set(duplicate_source_2.c_str(),
            2, 1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_duplicate_new(duplicate_source_3.c_str(),
            source_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_video_buffer_out_orientation_set(duplicate_source_3.c_str(),
            DSL_VIDEO_ORIENTATION_FLIP_HORIZONTALLY) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        
        const wchar_t* components[] = {L"uri-source-1", 
            L"duplicate-source-1", L"duplicate-source-2", L"duplicate-source-3",
            L"tiler", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-format is set to RGBA" ) 
        {
                
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

SCENARIO( "A V4L2 Camera Source with three Duplicate Sources can play",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, 3 Duplicate Sources, Tiler, and Window Sink" ) 
    {
        std::wstring duplicate_source_1(L"duplicate-source-1");
        std::wstring duplicate_source_2(L"duplicate-source-2");
        std::wstring duplicate_source_3(L"duplicate-source-3");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_duplicate_new(duplicate_source_1.c_str(),
            source_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_video_buffer_out_crop_rectangle_set(
            duplicate_source_1.c_str(), DSL_VIDEO_CROP_AT_SRC, 
            480, 270, 960, 540) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_duplicate_new(duplicate_source_2.c_str(),
            source_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_video_buffer_out_frame_rate_set(duplicate_source_2.c_str(),
            2, 1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_duplicate_new(duplicate_source_3.c_str(),
            source_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_video_buffer_out_orientation_set(duplicate_source_3.c_str(),
            DSL_VIDEO_ORIENTATION_FLIP_HORIZONTALLY) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        
        const wchar_t* components[] = {L"uri-source-1", 
            L"duplicate-source-1", L"duplicate-source-2", L"duplicate-source-3",
            L"tiler", L"egl-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-format is set to RGBA" ) 
        {
                
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
