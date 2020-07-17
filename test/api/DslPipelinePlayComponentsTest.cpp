/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

SCENARIO( "A new Pipeline with a URI File Source, FakeSink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Tiled Display" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0); 

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring fakeSinkName(L"fake-sink");

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"tiler", L"fake-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                uint currentState(DSL_STATE_NULL);
                REQUIRE( dsl_pipeline_state_get(pipelineName.c_str(), &currentState) == DSL_RESULT_SUCCESS );
                REQUIRE( currentState == DSL_STATE_PLAYING );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, OverlaySink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring overlaySinkName(L"overlay-sink");
        uint overlayId(1);
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"tiler", L"overlay-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
//SCENARIO( "A new Pipeline with a URI https Source, OverlaySink, and Tiled Display can play", "[temp]" )
//{
//    GIVEN( "A Pipeline, URI source, Overlay Sink, and Tiled Display" ) 
//    {
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri = L"https://www.radiantmediaplayer.com/media/bbb-360p.mp4";
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring tilerName(L"tiler");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring overlaySinkName(L"overlay-sink");
//        uint overlayId(1);
//        uint displayId(0);
//        uint depth(0);
//        uint offsetX(100);
//        uint offsetY(140);
//        uint sinkW(1280);
//        uint sinkH(720);
//
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        // create for of the same types of source
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            true, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        // overlay sink for observation 
//        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
//            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
//
//        // new tiler for this scenario
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source", L"tiler", L"overlay-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
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
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring windowSinkName = L"window-sink";
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"tiler", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Overlay Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Overlay Sink, and Tiled Display" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring modelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");
        
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring overlaySinkName(L"overlay-sink");
        uint overlayId(1);
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source",L"primary-gie", L"tiler", L"overlay-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "A new Pipeline with a URI Source, Primary GIE, KTL Tracker, Overlay Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, KTL Tracker, Primary GIE, Overlay Sink, and Tiled Display" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring modelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");
        
        std::wstring trackerName(L"ktl-tracker");
        uint trackerW(480);
        uint trackerH(272);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring overlaySinkName(L"overlay-sink");
        uint overlayId(1);
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source",L"primary-gie", L"ktl-tracker", L"tiler", L"overlay-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, KTL Tracker, Overlay Sink, On-Screen Display, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, KTL Tracker, Primary GIE, Overlay Sink, and Tiled Display" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring modelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");
        
        std::wstring trackerName(L"ktl-tracker");
        uint trackerW(480);
        uint trackerH(272);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring onScreenDisplayName(L"on-screen-display");
        bool isClockEnabled(false);

        std::wstring overlaySinkName(L"overlay-sink");
        uint overlayId(1);
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(onScreenDisplayName.c_str(), isClockEnabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", 
            L"tiler", L"on-screen-display", L"overlay-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

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
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring tilerName(L"tiler");
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
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source", L"tiler", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
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
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring tilerName(L"tiler");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring fileSinkName(L"file-sink");
//        std::wstring filePath(L"./output.mp4");
//        uint codec(DSL_CODEC_H265);
//        uint muxer(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source", L"tiler", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
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
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring tilerName(L"tiler");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring fileSinkName(L"file-sink");
//        std::wstring filePath(L"./output.mp4");
//        uint codec(DSL_CODEC_MPEG4);
//        uint muxer(DSL_CONTAINER_MP4);
//        uint bitrate(2000000);
//        uint interval(0);
//
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source", L"tiler", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
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
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // RTSP sink for this scenario
        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"tiler", L"rtsp-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

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
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring tilerName(L"tiler");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring rtspSinkName(L"rtsp-sink");
//        std::wstring host(L"rjhowell44-desktop.local");
//        uint udpPort(5400);
//        uint rtspPort(8554);
//        uint codec(DSL_CODEC_H265);
//        uint bitrate(4000000);
//        uint interval(0);
//
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        // RTSP sink for this scenario
//        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
//            udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        // new tiler for this scenario
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source", L"tiler", L"rtsp-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Secondary GIE, Overlay Sink, and Tiled Display can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Primary GIE, Overlay Sink, and Tiled Display" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring pgieInferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring pgieModelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");

        std::wstring trackerName(L"ktl-tracker");
        uint trackerW(480);
        uint trackerH(272);
        
        std::wstring secondaryGieName1(L"secondary-gie");
        std::wstring sgieInferConfigFile1(L"./test/configs/config_infer_secondary_carcolor_nano.txt");
        std::wstring sgieModelEngineFile1(L"./test/models/Secondary_CarColor/resnet18.caffemodel_b1_gpu0_fp16.engine");
        
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring onScreenDisplayName(L"on-screen-display");
        bool isClockEnabled(false);

        std::wstring overlaySinkName(L"overlay-sink");
        uint overlayId(1);
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_gie_secondary_new(secondaryGieName1.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(onScreenDisplayName.c_str(), isClockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source",L"primary-gie", L"ktl-tracker", L"secondary-gie", L"tiler", L"on-screen-display", L"overlay-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

//SCENARIO( "A new Pipeline with a URI Source, Primary GIE, Three Secondary GIEs, Overlay Sink, and Tiled Display can play", "[pipeline-play]" )
//{
//    GIVEN( "A Pipeline, URI source, Primary GIE, Overlay Sink, and Tiled Display" ) 
//    {
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring primaryGieName(L"primary-gie");
//        std::wstring pgieInferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
//        std::wstring pgieModelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");
//
//        std::wstring trackerName(L"ktl-tracker");
//        uint trackerW(480);
//        uint trackerH(272);
//        
//        std::wstring secondaryGieName1(L"secondary-gie1");
//        std::wstring sgieInferConfigFile1(L"./test/configs/config_infer_secondary_carcolor_nano.txt");
//        std::wstring sgieModelEngineFile1(L"./test/models/Secondary_CarColor/resnet18.caffemodel_b1_gpu0_fp16.engine");
//        
//        // Note new model is not saved for car color with DS 5.0 ??????
//        // need to let it generate a new engine by loading the previous?
//        std::wstring secondaryGieName2(L"secondary-gie2");
//        std::wstring sgieInferConfigFile2(L"./test/configs/config_infer_secondary_carmake_nano.txt");
//        std::wstring sgieModelEngineFile2(L"./test/models/Secondary_CarMake/resnet18.caffemodel_b1_gpu0_fp16.engine");
//        
//        std::wstring secondaryGieName3(L"secondary-gie3");
//        std::wstring sgieInferConfigFile3(L"./test/configs/config_infer_secondary_vehicletypes_nano.txt");
//        std::wstring sgieModelEngineFile3(L"./test/models/Secondary_VehicleTypes/resnet18.caffemodel_b1_gpu0_fp16.engine");
//        
//        std::wstring tilerName(L"tiler");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring onScreenDisplayName(L"on-screen-display");
//        bool isClockEnabled(false);
//
//        std::wstring overlaySinkName(L"overlay-sink");
//        uint overlayId(1);
//        uint displayId(0);
//        uint depth(0);
//        uint offsetX(100);
//        uint offsetY(140);
//        uint sinkW(1280);
//        uint sinkH(720);
//
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
//            pgieModelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_gie_secondary_new(secondaryGieName1.c_str(), sgieInferConfigFile1.c_str(), 
//            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), 0) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_gie_secondary_new(secondaryGieName2.c_str(), sgieInferConfigFile2.c_str(), 
//            sgieModelEngineFile2.c_str(), primaryGieName.c_str(), 0) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_gie_secondary_new(secondaryGieName3.c_str(), sgieInferConfigFile3.c_str(), 
//            sgieModelEngineFile3.c_str(), primaryGieName.c_str(), 0) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_osd_new(onScreenDisplayName.c_str(), isClockEnabled) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
//            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source",L"primary-gie", L"ktl-tracker", 
//            L"secondary-gie1", L"secondary-gie2", L"secondary-gie3", L"tiler", L"on-screen-display", L"overlay-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//
//                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_pipeline_list_size() == 0 );
//                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
//                REQUIRE( dsl_component_list_size() == 0 );
//            }
//        }
//    }
//}

SCENARIO( "A new Pipeline with a URI File Source, FakeSink, and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        std::wstring sourceName1(L"uri-source1");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring demuxerName(L"demuxer");
        std::wstring fakeSinkName(L"fake-sink");
        std::wstring branchName(L"test-branch");
        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        
        const wchar_t* components[] = {L"uri-source1", L"demuxer", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_branch_component_add(branchName.c_str(), fakeSinkName.c_str())== DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), branchName.c_str())== DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );

                REQUIRE( dsl_component_delete(demuxerName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, FakeSink, OverlaySink and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        std::wstring sourceName1(L"uri-source1");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring demuxerName(L"demuxer");
        std::wstring fakeSinkName(L"fake-sink");
        std::wstring branchName(L"test-branch");
        std::wstring pipelineName(L"test-pipeline");
        std::wstring overlaySinkName(L"overlay-sink");

        uint overlayId(1);
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* pipelineComps[] = {L"uri-source1", L"demuxer", NULL};
        const wchar_t* branchComps[] = {L"fake-sink", L"overlay-sink", NULL};

        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_branch_component_add_many(branchName.c_str(), branchComps) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxerName.c_str(), branchName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), pipelineComps) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );

                REQUIRE( dsl_component_delete(demuxerName.c_str()) == DSL_RESULT_SUCCESS );

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
        std::wstring sourceName1(L"uri-source1");
        std::wstring sourceName2(L"uri-source2");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring demuxerName(L"demuxer");

        std::wstring overlaySinkName1(L"overlay-sink1");
        std::wstring overlaySinkName2(L"overlay-sink2");

        uint overlayId1(1);
        uint displayId1(0);
        uint depth1(0);
        uint overlayId2(2);
        uint displayId2(0);
        uint depth2(0);

        uint offsetX1(100);
        uint offsetY1(140);
        uint offsetX2(400);
        uint offsetY2(440);
        uint sinkW1(720);
        uint sinkH1(360);
        uint sinkW2(720);
        uint sinkH2(360);

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName1.c_str(), overlayId1, displayId1, depth1,
            offsetX1, offsetY1, sinkW1, sinkH1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName2.c_str(), overlayId2, displayId2, depth2,
            offsetX2, offsetY2, sinkW2, sinkH2) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source1", L"uri-source2", L"demuxer", NULL};
        const wchar_t* branches[] = {L"overlay-sink1", L"overlay-sink2", NULL};
        
        WHEN( "When the Sinks are added to the Demuxer and the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_tee_branch_add_many(demuxerName.c_str(), branches) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );

                REQUIRE( dsl_tee_branch_remove_many(demuxerName.c_str(), branches) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with two URI File Sources, PGIE, Demuxer two Overlay Sinks, one OSD, and Demuxer can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        std::wstring sourceName1(L"uri-source1");
        std::wstring sourceName2(L"uri-source2");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring pgieInferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring pgieModelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b2_gpu0_fp16.engine");

        std::wstring demuxerName(L"demuxer");

        std::wstring overlaySinkName1(L"overlay-sink1");
        std::wstring overlaySinkName2(L"overlay-sink2");
        uint overlayId1(1);
        uint displayId1(0);
        uint depth1(0);
        uint overlayId2(2);
        uint displayId2(0);
        uint depth2(0);
        uint offsetX1(160);
        uint offsetY1(240);
        uint offsetX2(750);
        uint offsetY2(340);
        uint sinkW1(720);
        uint sinkH1(360);
        uint sinkW2(1080);
        uint sinkH2(540);
        
        std::wstring osdName(L"on-screen-display");

        std::wstring branchName(L"branch");
        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName1.c_str(), overlayId1, displayId1, depth1,
            offsetX1, offsetY1, sinkW1, sinkH1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName2.c_str(), overlayId2, displayId2, depth2,
            offsetX2, offsetY2, sinkW2, sinkH2) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_demuxer_new(demuxerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source1", L"uri-source2", L"primary-gie", L"demuxer", NULL};
        const wchar_t* branchComps[] = {L"on-screen-display", L"overlay-sink1", NULL};
        const wchar_t* branches[] = {L"branch", L"overlay-sink2", NULL};
        
        WHEN( "When the Sinks are added to Sources the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_branch_component_add_many(branchName.c_str(), branchComps) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add_many(demuxerName.c_str(), branches) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );

                REQUIRE( dsl_tee_branch_remove_many(demuxerName.c_str(), branches) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_branch_component_remove_many(branchName.c_str(), branchComps) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, Splitter, OSD, and two OverlaySinks can play", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Fake Sink, and Demuxer" ) 
    {
        std::wstring sourceName1(L"uri-source1");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring splitterName(L"splitter");

        std::wstring branchName1(L"branch1");
        std::wstring branchName2(L"branch2");

        std::wstring primaryGieName(L"primary-gie");
        std::wstring pgieInferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring pgieModelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");

        std::wstring tilerName1(L"tiler1");
        std::wstring tilerName2(L"tiler2");
        uint width(1280);
        uint height(720);

        std::wstring osdName(L"on-screen-display");

        std::wstring pipelineName(L"test-pipeline");
        std::wstring overlaySinkName1(L"overlay-sink1");
        std::wstring overlaySinkName2(L"overlay-sink2");
        uint overlayId1(1);
        uint displayId1(0);
        uint depth1(0);
        uint overlayId2(2);
        uint displayId2(0);
        uint depth2(0);
        uint offsetX1(160);
        uint offsetY1(240);
        uint offsetX2(750);
        uint offsetY2(340);
        uint sinkW1(720);
        uint sinkH1(360);
        uint sinkW2(1080);
        uint sinkH2(540);
        

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_splitter_new(splitterName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branchName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tiler_new(tilerName1.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tiler_new(tilerName2.c_str(), width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName1.c_str(), overlayId1, displayId1, depth1,
            offsetX1, offsetY1, sinkW1, sinkH1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_overlay_new(overlaySinkName2.c_str(), overlayId2, displayId2, depth2,
            offsetX2, offsetY2, sinkW2, sinkH2) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* branchComps1[] = {L"tiler1", L"overlay-sink1", NULL};
        const wchar_t* branchComps2[] = {L"primary-gie", L"tiler2", L"on-screen-display", L"overlay-sink2", NULL};
        const wchar_t* branches[] = {L"branch1", L"branch2", NULL};
        const wchar_t* components[] = {L"uri-source1", L"splitter", NULL};

        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_branch_component_add_many(branchName1.c_str(), branchComps1) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_branch_component_add_many(branchName2.c_str(), branchComps2) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add_many(splitterName.c_str(), branches) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

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
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring tilerName(L"tiler");
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
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
//            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );
//
//        const wchar_t* components[] = {L"uri-source", L"tiler", L"file-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                bool currIsClockEnabled(false);
//                
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
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
//        std::wstring sourceName1(L"uri-source");
//        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
//        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
//        uint intrDecode(false);
//        uint dropFrameInterval(0);
//
//        std::wstring ofvName(L"ofv");
//
//        std::wstring tilerName(L"tiler");
//        uint width(1280);
//        uint height(720);
//
//        std::wstring windowSinkName = L"window-sink";
//        uint offsetX(100);
//        uint offsetY(140);
//        uint sinkW(1280);
//        uint sinkH(720);
//
//        std::wstring pipelineName(L"test-pipeline");
//        
//        REQUIRE( dsl_component_list_size() == 0 );
//
//        // create for of the same types of source
//        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
//            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
//
//        // overlay sink for observation 
//        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(), 
//            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
//
//        REQUIRE( dsl_ofv_new(ofvName.c_str()) == DSL_RESULT_SUCCESS );
//        
//        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
//        
//        const wchar_t* components[] = {L"uri-source", L"ofv", L"tiler", L"window-sink", NULL};
//        
//        WHEN( "When the Pipeline is Assembled" ) 
//        {
//            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//        
//            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );
//
//            THEN( "Pipeline is Able to LinkAll and Play" )
//            {
//                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
//                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
//                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
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


SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and ImageSink can capture frames", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Tiled Display, and Image Sink" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring imageSinkName(L"image-sink");
        std::wstring outdir(L"./");

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_image_new(imageSinkName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_image_frame_capture_enabled_set(imageSinkName.c_str(), true) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"tiler", L"image-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI File Source, Tiled Display, and ImageSink can capture objects", "[pipeline-play]" )
{
    GIVEN( "A Pipeline, URI source, Tiled Display, and Image Sink" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring modelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_gpu0_fp16.engine");
        
        std::wstring trackerName(L"ktl-tracker");
        uint trackerW(480);
        uint trackerH(272);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        std::wstring imageSinkName(L"image-sink");
        std::wstring outdir(L"./");

        std::wstring pipelineName(L"test-pipeline");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_image_new(imageSinkName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_image_object_capture_enabled_set(imageSinkName.c_str(), true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_image_object_capture_class_add(imageSinkName.c_str(), 0, false, 0) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_image_object_capture_class_add(imageSinkName.c_str(), 2, false, 0) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"image-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                bool currIsClockEnabled(false);
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
