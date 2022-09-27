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
static const std::wstring source_name2(L"uri-source-2");
static const std::wstring source_name3(L"uri-source-3");
static const std::wstring source_name4(L"uri-source-4");
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static const uint intr_decode(false);
static const uint drop_frame_interval(0); 

static const std::wstring image_source(L"image-source");
static const std::wstring image_path1(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");
static const uint fps_n(15), fps_d(1);

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

// Note: Creating segmantation model engine file with the below config files builds the
// engine under the trtis_model_repo, even though we are not using the triton inference server
static const std::wstring sem_seg_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-segmentation-test/dstest_segmentation_config_semantic.txt");
static const std::wstring sem_seg_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/trtis_model_repo/Segmentation_Semantic/1/unetres18_v4_pruned0.65_800_data.uff_b1_gpu0_fp32.engine");

static const std::wstring image_path2(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_industrial.jpg");
static const std::wstring ind_seg_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-segmentation-test/dstest_segmentation_config_industrial.txt");
static const std::wstring ind_seg_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/triton_model_repo/Segmentation_Industrial/1/unet_output_graph.uff_b1_gpu0_fp32.engine");

static const std::wstring seg_visual_name(L"segvisual");
static const uint seg_visual_width(512);
static const uint seg_visual_height(512);

static const std::wstring tracker_name(L"ktl-tracker");
static const uint tracker_width(480);
static const uint tracker_height(272);

static const std::wstring secondary_gie_name1(L"secondary-gie-1");
static const std::wstring sgie_infer_config_file1(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carcolor.txt");
static const std::wstring sgie_model_engine_file1(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine");
        
static const std::wstring secondary_gie_name2(L"secondary-gie-2");
static const std::wstring sgie_infer_config_file2(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carmake.txt");
static const std::wstring sgie_model_engine_file2(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarMake/resnet18.caffemodel_b8_gpu0_fp16.engine");
        
static const std::wstring secondary_gie_name3(L"secondary-gie-3");
static const std::wstring sgie_infer_config_file3(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
static const std::wstring sgie_model_engine_file3(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18.caffemodel_b8_gpu0_fp16.engine");

static const std::wstring tiler_name1(L"tiler-1");
static const std::wstring tiler_name2(L"tiler-2");
static const uint tiler_width(1920);
static const uint tiler_height(1080);

static const std::wstring demuxer_name(L"demuxer");
static const std::wstring branch_name1(L"branch-1");
static const std::wstring branch_name2(L"branch-2");

static const std::wstring osd_name(L"on-screen-display");
static const boolean text_enabled(false);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const std::wstring fake_sink_name(L"fake-sink");
static const std::wstring overlay_sink_name1(L"overlay-sink-1");
static const std::wstring overlay_sink_name2(L"overlay-sink-2");
static const uint display_id(0);
static const uint depth(0);
static const uint offest_x(100);
static const uint offest_y(140);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"window-sink");

static const std::wstring rtsp_sink_name(L"rtsp-sink");
static const std::wstring host(L"rjhowell-desktop.local");
static const uint udp_port(5400);
static const uint rtsp_port(8554);
static const uint codec(DSL_CODEC_H264);
static const uint bitrate(4000000);
static const uint interval(0);

SCENARIO( "A new Pipeline with a URI File Source, FakeSink", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"fake-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_pipeline_state_get(pipeline_name.c_str(), &currentState) == DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, GIE, FakeSink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1", L"primary-gie", L"tiler-1", L"fake-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_pipeline_state_get(pipeline_name.c_str(), &currentState) == DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
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

SCENARIO( "A new Pipeline with a URI File Source, Overlay Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
                false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name1.c_str(), display_id, depth,
                offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
            
            const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"overlay-sink-1", NULL};
            
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
}
//SCENARIO( "A new Pipeline with a URI https Source, OverlaySink, and Tiled Display can play", "[temp]" )
//{
//    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
//    {
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            true, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_sink_overlay_new(overlay_sink_name1.c_str(), display_id, depth,
//            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"overlay-sink-1", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

SCENARIO( "A new Pipeline with a URI File Source, Window Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
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


SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Window Sink, and Tiled Display can play", "[pipeline-play]" )
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
        
        const wchar_t* components[] = {L"uri-source-1",L"primary-gie", L"tiler-1", L"window-sink", NULL};
        
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


SCENARIO( "A new Pipeline with a URI Source, Primary GIE, KTL Tracker, Window Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, KTL Tracker, Primary GIE, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(tracker_name.c_str(), tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1",L"primary-gie", L"ktl-tracker", L"tiler-1", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
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

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, KTL Tracker, Window Sink, \
    On-Screen Display, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, KTL Tracker, Primary GIE, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(tracker_name.c_str(), tracker_width, tracker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1", L"primary-gie", L"ktl-tracker", 
            L"tiler-1", L"on-screen-display", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
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

//SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and DSL_CODEC_H264 FileSink can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
//    {
//
//        std::wstring fileSinkName(L"file-sink");
//        std::wstring filePath(L"./output.mp4");
//        uint codec(DSL_CODEC_H264);
//        uint muxer(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}
//
//SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and DSL_CODEC_H265 FileSink can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
//    {
//        std::wstring fileSinkName(L"file-sink");
//        std::wstring filePath(L"./output.mp4");
//        uint codec(DSL_CODEC_H265);
//        uint muxer(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

//SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and DSL_CODEC_MPEG4 FileSink can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
//    {
//        std::wstring fileSinkName(L"file-sink");
//        std::wstring filePath(L"./output.mp4");
//        uint codec(DSL_CODEC_MPEG4);
//        uint muxer(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

SCENARIO( "A new Pipeline with a URI File Source, DSL_CODEC_H264 RTSP Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, DSL_CODEC_H264 RTSP Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_rtsp_new(rtsp_sink_name.c_str(), host.c_str(),
            udp_port, rtsp_port, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"rtsp-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

//SCENARIO( "A new Pipeline with a URI File Source, DSL_CODEC_H265 RTSP Sink, and Tiled Display can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, DSL_CODEC_H265 RTSP Sink, and Tiled Display" ) 
//    {
//        std::wstring rtsp_sink_name(L"rtsp-sink");
//        std::wstring host(L"prominence-desktop1.local");
//        uint udp_port(5400);
//        uint rtsp_port(8554);
//        uint codec(DSL_CODEC_H265);
//        uint bitrate(4000000);
//        uint interval(0);
//
//        std::wstring pipeline_name(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_sink_rtsp_new(rtsp_sink_name.c_str(), host.c_str(),
//            udp_port, rtsp_port, codec, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"rtsp-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Secondary GIE, \
    Window Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Overlay Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_ktl_new(tracker_name.c_str(), tracker_width, tracker_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name1.c_str(), sgie_infer_config_file1.c_str(), 
            sgie_model_engine_file1.c_str(), primary_gie_name.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1",L"primary-gie", L"ktl-tracker", 
            L"secondary-gie-1", L"tiler-1", L"on-screen-display", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
            
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
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

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Three Secondary GIEs, \
    Window Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_ktl_new(tracker_name.c_str(), 
            tracker_width, tracker_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name1.c_str(), sgie_infer_config_file1.c_str(), 
            sgie_model_engine_file1.c_str(), primary_gie_name.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name2.c_str(), sgie_infer_config_file2.c_str(), 
            sgie_model_engine_file2.c_str(), primary_gie_name.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name3.c_str(), sgie_infer_config_file3.c_str(), 
            sgie_model_engine_file3.c_str(), primary_gie_name.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1",L"primary-gie", L"ktl-tracker", 
            L"secondary-gie-1", L"secondary-gie-2", L"secondary-gie-3", L"tiler-1", L"on-screen-display", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
            
            dsl_pipeline_streammux_batch_properties_set(pipeline_name.c_str(), 8, 40000 );

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

SCENARIO( "A new Pipeline with a URI File Source, FakeSink, and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branch_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        
        const wchar_t* components[] = {L"uri-source-1", L"demuxer", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_branch_component_add(branch_name1.c_str(), fake_sink_name.c_str())== DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), branch_name1.c_str())== DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );

                REQUIRE( dsl_component_delete(demuxer_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, FakeSink, WindowSink and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, Window Sink and Demuxer" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branch_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* pipelineComps[] = {L"uri-source-1", L"demuxer", NULL};
        const wchar_t* branchComps[] = {L"fake-sink", L"window-sink", NULL};

        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_branch_component_add_many(branch_name1.c_str(), branchComps) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxer_name.c_str(), branch_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), pipelineComps) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );

                REQUIRE( dsl_component_delete(demuxer_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with two URI File Sources, two overlaySinks and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            uint offest_x1(100);
            uint offest_y1(140);
            uint offest_x2(400);
            uint offest_y2(440);
            uint sink_width1(720);
            uint sink_height1(360);
            uint sink_width2(720);
            uint sink_height2(360);

            std::wstring pipeline_name(L"test-pipeline");
            
            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
                false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_source_uri_new(source_name2.c_str(), uri.c_str(), 
                false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name1.c_str(), display_id, depth,
                offest_x1, offest_y1, sink_width1, sink_height1) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name2.c_str(), display_id, depth,
                offest_x2, offest_y2, sink_width2, sink_height2) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            
            const wchar_t* components[] = {L"uri-source-1", L"uri-source-2", L"demuxer", NULL};
            const wchar_t* branches[] = {L"overlay-sink-1", L"overlay-sink-2", NULL};
            
            WHEN( "When the Sinks are added to the Demuxer and the Pipeline is Assembled" ) 
            {
                REQUIRE( dsl_tee_branch_add_many(demuxer_name.c_str(), branches) == DSL_RESULT_SUCCESS );
            
                REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

                THEN( "Pipeline is Able to LinkAll and Play" )
                {
                    REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_pipeline_list_size() == 0 );

                    REQUIRE( dsl_tee_branch_remove_many(demuxer_name.c_str(), branches) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "A new Pipeline with two URI File Sources, PGIE, Demuxer two Overlay Sinks, \
    one OSD, and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            uint offest_x1(160);
            uint offest_y1(240);
            uint offest_x2(750);
            uint offest_y2(340);
            uint sink_width1(720);
            uint sink_height1(360);
            uint sink_width2(1080);
            uint sink_height2(540);
            
            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
                false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_source_uri_new(source_name2.c_str(), uri.c_str(), 
                false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
                bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name1.c_str(), display_id, depth,
                offest_x1, offest_y1, sink_width1, sink_height1) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name2.c_str(), display_id, depth,
                offest_x2, offest_y2, sink_width2, sink_height2) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
                model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_tee_demuxer_new(demuxer_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_branch_new(branch_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            
            const wchar_t* components[] = {L"uri-source-1", L"uri-source-2", L"primary-gie", L"demuxer", NULL};
            const wchar_t* branchComps[] = {L"on-screen-display", L"overlay-sink-1", NULL};
            const wchar_t* branches[] = {L"branch-1", L"overlay-sink-2", NULL};
            
            WHEN( "When the Sinks are added to Sources the Pipeline is Assembled" ) 
            {
                REQUIRE( dsl_branch_component_add_many(branch_name1.c_str(), branchComps) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_add_many(demuxer_name.c_str(), branches) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

                THEN( "Pipeline is Able to LinkAll and Play" )
                {
                    REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    
                    REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_pipeline_list_size() == 0 );

                    REQUIRE( dsl_tee_branch_remove_many(demuxer_name.c_str(), branches) == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_branch_component_remove_many(branch_name1.c_str(), branchComps) == DSL_RESULT_SUCCESS );

                    REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, Splitter, OSD, and two OverlaySinks can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        // Get the Device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        if (deviceProp.integrated)
        {
            std::wstring splitterName(L"splitter");

            uint offest_x1(160);
            uint offest_y1(240);
            uint offest_x2(750);
            uint offest_y2(340);
            uint sink_width1(720);
            uint sink_height1(360);
            uint sink_width2(1080);
            uint sink_height2(540);

            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
                false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_tee_splitter_new(splitterName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_branch_new(branch_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_branch_new(branch_name2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tiler_new(tiler_name2.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
                bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name1.c_str(), display_id, depth,
                offest_x1, offest_y1, sink_width1, sink_height1) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_overlay_new(overlay_sink_name2.c_str(), display_id, depth,
                offest_x2, offest_y2, sink_width2, sink_height2) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
                model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            
            const wchar_t* branchComps1[] = {L"tiler-1", L"overlay-sink-1", NULL};
            const wchar_t* branchComps2[] = {L"primary-gie", L"tiler-2", L"on-screen-display", L"overlay-sink-2", NULL};
            const wchar_t* branches[] = {L"branch-1", L"branch-2", NULL};
            const wchar_t* components[] = {L"uri-source-1", L"splitter", NULL};

            WHEN( "When the Pipeline is Assembled" ) 
            {
                REQUIRE( dsl_branch_component_add_many(branch_name1.c_str(), branchComps1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_branch_component_add_many(branch_name2.c_str(), branchComps2) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_add_many(splitterName.c_str(), branches) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

                THEN( "Pipeline is Able to LinkAll and Play" )
                {
                    bool currIsClockEnabled(false);
                    
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
}


//SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and DSL_CODEC_H264 FileSink can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
//    {
//        std::wstring source_name1(L"uri-source-1");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadec_mem_type(DSL_NVBUF_MEM_TYPE_DEFAULT);
//        uint intr_decode(false);
//        uint drop_frame_interval(0);
//
//        std::wstring tiler_name1(L"tiler-1");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring fileSinkName(L"file-sink");
//        std::wstring filePath(L"./output.mp4");
//        uint codec(DSL_CODEC_H264);
//        uint muxer(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        std::wstring pipeline_name(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

//SCENARIO( "A new Pipeline with a URI File Source, OFV, Window Sink, and Tiled Display can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, OFV, Window Sink, and Tiled Display" ) 
//    {
//        std::wstring source_name1(L"uri-source-1");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadec_mem_type(DSL_NVBUF_MEM_TYPE_DEFAULT);
//        uint intr_decode(false);
//        uint drop_frame_interval(0);
//
//        std::wstring ofvName(L"ofv");
//
//        std::wstring tiler_name1(L"tiler-1");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring window_sink_name = L"window-sink";
//        uint offest_x(100);
//        uint offest_y(140);
//        uint sink_width(1280);
//        uint sink_height(720);
//
//        std::wstring pipeline_name(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        // create for of the same types of source
//        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
//            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
//
//        // overlay sink for observation 
//        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
//            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_ofv_new(ofvName.c_str()) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source-1", L"ofv", L"tiler-1", L"window-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}
//


SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and Meter PPH can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Tiled Display, and Meter PPH" ) 
    {
        std::wstring meterPphName(L"meter-pph");
        uint interval(1);
        dsl_pph_meter_client_handler_cb client_handler;

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_meter_new(meterPphName.c_str(), interval, 
            client_handler, NULL) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tiler_name1.c_str(), 
            meterPphName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"tiler-1", L"fake-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_pipeline_state_get(pipeline_name.c_str(), &currentState) == DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a Image Source, Window Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_image_stream_new(image_source.c_str(), image_path1.c_str(), false, 
            fps_n, fps_d, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"image-source", L"tiler-1", L"window-sink", NULL};
        
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

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Semantic Segmentation", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Segmentation Visualizer, Primary GIE, and Overlay Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), sem_seg_infer_config_file.c_str(), 
            sem_seg_model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_segvisual_new(seg_visual_name.c_str(), seg_visual_width, seg_visual_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"primary-gie",
            L"segvisual", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
            
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
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

//SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Industrial Segmentation, Window Sink can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, Segmentation Visualizer, Primary GIE, and Window Sink" ) 
//    {
//        REQUIRE( dsl_component_list_size() == 0 );
//
////        REQUIRE( dsl_source_uri_new(source_name1.c_str(), image_path2.c_str(), 
////            false, intr_decode, interval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_source_image_stream_new(image_source.c_str(), image_path1.c_str(), false, 
//            fps_n, fps_d, 0) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), ind_seg_infer_config_file.c_str(), 
//            ind_seg_model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_segvisual_new(seg_visual_name.c_str(), seg_visual_width, seg_visual_height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
//            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
//            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"image-source", L"primary-gie",
//            L"segvisual", L"on-screen-display", L"window-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );
//            
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*20);
//                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

SCENARIO( "A new Pipeline-Stream-Muxer with Tiler 4 URI Sources, Primary GIE, Window Sink", 
    "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Window Sink, and Tiled Display" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_uri_new(source_name2.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_uri_new(source_name3.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_uri_new(source_name4.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), text_enabled, clock_enabled,
            bbox_enabled, mask_enabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), tiler_width, tiler_height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source-1", L"uri-source-2", L"uri-source-3", L"uri-source-4", 
            L"primary-gie", L"on-screen-display", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled and a Tiler is added to the output of the Stream-Muxer" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_streammux_tiler_add(pipeline_name.c_str(), 
                tiler_name1.c_str()) == DSL_RESULT_SUCCESS );

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

