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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(2000)

SCENARIO( "A new Pipeline with an ODE Handler without any child ODE Types can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, URI source, KTL Tracker, Primary GIE, Tiled Display, ODE Hander, and Overlay Sink" ) 
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
        
        std::wstring odeHandlerName(L"ode-handler");

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

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Occurrence ODE Type, and Print ODE Action can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Type, and Print ODE Action" ) 
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

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring odeTypeName(L"occurrence");
        uint classId(0);
        uint limit(10);
        std::wstring odeActionName(L"print");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName.c_str(), classId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_print_new(odeActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(odeTypeName.c_str(), odeActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), odeTypeName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Two Occurrence ODE Types, each with Redact ODE Actions can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Type, and Print ODE Action" ) 
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

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring odeCarOccurrenceName(L"car-occurrence");
        uint carClassId(0);
        std::wstring odePersonOccurrenceName(L"person-occurrence");
        uint personClassId(2);

        std::wstring odeRedactActionName(L"redact");
        
        uint limit(0);
        
        std::wstring osdName(L"osd");
        boolean clockEnabled(false);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_occurrence_new(odeCarOccurrenceName.c_str(), carClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odePersonOccurrenceName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );

        // shared redaction action
        REQUIRE( dsl_ode_action_redact_new(odeRedactActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(odeCarOccurrenceName.c_str(), odeRedactActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(odePersonOccurrenceName.c_str(), odeRedactActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), odeCarOccurrenceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), odePersonOccurrenceName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(osdName.c_str(), clockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"osd", L"overlay-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Two Occurrence ODE Types sharing a Capture ODE Action can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Type, and Capture ODE Action" ) 
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

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring firstCarOccurrenceName(L"first-car-occurrence");
        uint carClassId(0);
        std::wstring firstPersonOccurrenceName(L"first-person-occurrence");
        uint personClassId(2);
        
        uint limit(1);
        std::wstring captureActionName(L"capture-action");
        std::wstring outdir(L"./");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_occurrence_new(firstCarOccurrenceName.c_str(), carClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(firstPersonOccurrenceName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_capture_object_new(captureActionName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
        
        // Add the same capture Action to both ODE Types
        REQUIRE( dsl_ode_trigger_action_add(firstCarOccurrenceName.c_str(), captureActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(firstPersonOccurrenceName.c_str(), captureActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), firstCarOccurrenceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), firstPersonOccurrenceName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, an Occurrence ODE Type, with a Pause Pipeline ODE Action can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Type, and Capture ODE Action" ) 
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

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring firstPersonOccurrenceName(L"first-person-occurrence");
        uint personClassId(2);
        
        uint limit(1);
        std::wstring pauseActionName(L"pause-action");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_occurrence_new(firstPersonOccurrenceName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_pause_new(pauseActionName.c_str(), pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(firstPersonOccurrenceName.c_str(), pauseActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), firstPersonOccurrenceName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Four Occurrence ODE Type with a shared Display ODE Action can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Occurrence ODE Type, and Display ODE Action" ) 
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

        std::wstring osdName(L"osd");
        boolean clockEnabled(false);

        std::wstring pipelineName(L"test-pipeline");

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring carOccurrenceName(L"Car");
        uint carClassId(0);
        std::wstring bicycleOccurrenceName(L"Bicycle");
        uint bicycleClassId(1);
        std::wstring personOccurrenceName(L"Person");
        uint personClassId(2);
        std::wstring roadsignOccurrenceName(L"Roadsign");
        uint roadsignClassId(3);
        
        uint limit(0);
        std::wstring displayActionName(L"display-action");
        uint textOffsetX(10);
        uint textOffsetY(20);
        
        std::wstring font(L"arial");
        std::wstring fontName(L"arial-14");
        uint size(14);

        std::wstring fullBlack(L"full-black");
        REQUIRE( dsl_display_type_rgba_color_new(fullBlack.c_str(), 
            0.0, 0.0, 0.0, 1.0) == DSL_RESULT_SUCCESS );


        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, fullBlack.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        // Single display action shared by all ODE Occurrence Types
        REQUIRE( dsl_ode_action_display_new(displayActionName.c_str(), textOffsetX, textOffsetX, true,
            fontName.c_str(), false, fullBlack.c_str()) == DSL_RESULT_SUCCESS );
        
        // Create all occurrences
        REQUIRE( dsl_ode_trigger_occurrence_new(carOccurrenceName.c_str(), carClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(carOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(bicycleOccurrenceName.c_str(), bicycleClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(bicycleOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(personOccurrenceName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(personOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(roadsignOccurrenceName.c_str(), roadsignClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(roadsignOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* odeTypes[] = {L"Car", L"Bicycle", L"Person", L"Roadsign", NULL};
        
        REQUIRE( dsl_ode_handler_trigger_add_many(odeHandlerName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osdName.c_str(), clockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"osd", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Four Summation ODE Type with a shared Display ODE Action can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Four Summation ODE Types, and Display ODE Action" ) 
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

        std::wstring osdName(L"osd");
        boolean clockEnabled(false);

        std::wstring pipelineName(L"test-pipeline");

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring carOccurrenceName(L"Car");
        uint carClassId(0);
        std::wstring bicycleOccurrenceName(L"Bicycle");
        uint bicycleClassId(1);
        std::wstring personOccurrenceName(L"Person");
        uint personClassId(2);
        std::wstring roadsignOccurrenceName(L"Roadsign");
        uint roadsignClassId(3);
        
        uint limit(0);
        std::wstring displayActionName(L"display-action");
        uint textOffsetX(10);
        uint textOffsetY(20);

        std::wstring fullBlack(L"full-black");
        REQUIRE( dsl_display_type_rgba_color_new(fullBlack.c_str(), 
            0.0, 0.0, 0.0, 1.0) == DSL_RESULT_SUCCESS );

        std::wstring font(L"arial");
        std::wstring fontName(L"arial-14");
        uint size(14);

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, fullBlack.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        // Single display action shared by all ODT Occurrence Types
        REQUIRE( dsl_ode_action_display_new(displayActionName.c_str(), textOffsetX, textOffsetX, true,
            fontName.c_str(), false, fullBlack.c_str()) == DSL_RESULT_SUCCESS );
        
        // Create all occurrences
        REQUIRE( dsl_ode_trigger_summation_new(carOccurrenceName.c_str(), carClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(carOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_summation_new(bicycleOccurrenceName.c_str(), bicycleClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(bicycleOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_summation_new(personOccurrenceName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(personOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_summation_new(roadsignOccurrenceName.c_str(), roadsignClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(roadsignOccurrenceName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* odeTypes[] = {L"Car", L"Bicycle", L"Person", L"Roadsign", NULL};
        
        REQUIRE( dsl_ode_handler_trigger_add_many(odeHandlerName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osdName.c_str(), clockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"osd", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Four Summation ODE Types with a shared Display ODE Action can play", "[ode-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Four Summation ODE Types, and Display ODE Action" ) 
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

        std::wstring osdName(L"osd");
        boolean clockEnabled(false);

        std::wstring pipelineName(L"test-pipeline");

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring carSummationName(L"Car");
        uint carClassId(0);
        std::wstring bicycleSummationName(L"Bicycle");
        uint bicycleClassId(1);
        std::wstring personSummationName(L"Person");
        std::wstring personOccurrenceName(L"person-occurrence");
        uint personClassId(2);
        std::wstring roadsignSummationName(L"Roadsign");
        uint roadsignClassId(3);
        
        uint limit(0);
        std::wstring displayActionName(L"display-action");
        uint textOffsetX(10);
        uint textOffsetY(20);
        
        std::wstring printActionName(L"print-action");
        std::wstring fillActionName(L"fill-action");
        
        std::wstring areaName(L"area");
        
        std::wstring lightRed(L"light-red");
        REQUIRE( dsl_display_type_rgba_color_new(lightRed.c_str(), 
            0.2, 0.0, 0.0, 0.5) == DSL_RESULT_SUCCESS );

        std::wstring fullWhite(L"full-white");
        REQUIRE( dsl_display_type_rgba_color_new(fullWhite.c_str(), 
            1.0, 1.0, 1.0, 1.0) == DSL_RESULT_SUCCESS );

        std::wstring fullBlack(L"full-black");
        REQUIRE( dsl_display_type_rgba_color_new(fullBlack.c_str(), 
            1.0, 1.0, 1.0, 1.0) == DSL_RESULT_SUCCESS );

        std::wstring font(L"arial");
        std::wstring fontName(L"arial-14");
        uint size(14);

        REQUIRE( dsl_display_type_rgba_font_new(fontName.c_str(), font.c_str(),
            size, fullWhite.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        // Set Area critera, and The fill action for ODE occurrence caused by overlap
        REQUIRE( dsl_ode_trigger_occurrence_new(personOccurrenceName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_fill_object_new(fillActionName.c_str(), lightRed.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(personOccurrenceName.c_str(), fillActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        // Create a new ODE Area for criteria
        REQUIRE( dsl_ode_area_new(areaName.c_str(), 500, 0, 10, 1080, true) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_area_add(personOccurrenceName.c_str(), areaName.c_str()) == DSL_RESULT_SUCCESS );

        // Single display action shared by all ODT Summation Types
        REQUIRE( dsl_ode_action_display_new(displayActionName.c_str(), textOffsetX, textOffsetX, true,
            fontName.c_str(), true, fullBlack.c_str()) == DSL_RESULT_SUCCESS );
        
        // Create all Summation types and add common Display action
        REQUIRE( dsl_ode_trigger_summation_new(carSummationName.c_str(), carClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(carSummationName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_summation_new(bicycleSummationName.c_str(), bicycleClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(bicycleSummationName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_summation_new(personSummationName.c_str(), personClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(personSummationName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_summation_new(roadsignSummationName.c_str(), roadsignClassId, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_action_add(roadsignSummationName.c_str(), displayActionName.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* odeTypes[] = {L"Car", L"Bicycle", L"Person", L"Roadsign", L"person-occurrence", NULL};
        
        REQUIRE( dsl_ode_handler_trigger_add_many(odeHandlerName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osdName.c_str(), clockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"ode-handler", L"osd", L"overlay-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an ODE Handler, Occurrence ODE Trigger, Start Record ODE Action can play", "[new]" )
{
    GIVEN( "A Pipeline, ODE Handler, Four Summation ODE Types, and Display ODE Action" ) 
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

        std::wstring osdName(L"osd");
        boolean clockEnabled(false);

        std::wstring recordSinkName(L"record-sink");
        std::wstring outdir(L"./");
        uint codec(DSL_CODEC_H265);
        uint bitrate(2000000);
        uint interval(0);

        uint container(DSL_CONTAINER_MKV);

        std::wstring pipelineName(L"test-pipeline");

        std::wstring odeHandlerName(L"ode-handler");
        
        std::wstring bicycleOccurrenceName(L"Bicycle");
        uint bicycleClassId(1);
        
        uint limit(1);
        std::wstring recordActionName(L"start-record-action");
        std::wstring printActionName(L"print-action");
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_action_print_new(printActionName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_sink_record_start_new(recordActionName.c_str(), 
            recordSinkName.c_str(), 2, 5, NULL) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_occurrence_new(bicycleOccurrenceName.c_str(), bicycleClassId, limit) == DSL_RESULT_SUCCESS );
        
        const wchar_t* actions[] = {L"start-record-action", L"print-action", NULL};

        REQUIRE( dsl_ode_trigger_action_add_many(bicycleOccurrenceName.c_str(), actions) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_handler_trigger_add(odeHandlerName.c_str(), bicycleOccurrenceName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osdName.c_str(), clockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), overlayId, displayId, depth,
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_record_new(recordSinkName.c_str(), outdir.c_str(),
            codec, container, bitrate, interval, NULL) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler",
            L"ode-handler", L"osd", L"overlay-sink", L"record-sink", NULL};
        
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
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
            }
        }
    }
}
