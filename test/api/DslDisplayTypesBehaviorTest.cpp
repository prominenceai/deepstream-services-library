/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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

SCENARIO( "All DisplayTypes can be displayed by and eODE Action", "[display-types-behavior]" )
{
    GIVEN( "A Pipeline, ODE Handler, Always ODE Trigger, and Display Meta Action" ) 
    {
        std::wstring sourceName1(L"uri-source");
        std::wstring uri(L"./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_NVBUF_MEM_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile(L"./test/configs/config_infer_primary_nano.txt");
        std::wstring modelEngineFile(L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
        
        std::wstring trackerName(L"ktl-tracker");
        uint trackerW(480);
        uint trackerH(272);

        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);
        
        std::wstring windowSinkName(L"window-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName(L"test-pipeline");
        
        std::wstring odeAlwaysTriggerName(L"always-trigger");

        std::wstring odeDisplayMetaActionName(L"display-meta");
        
        uint limit(0);
        
        std::wstring osdName(L"osd");
        boolean textEnabled(false);
        boolean clockEnabled(false);

        std::wstring odePphName(L"ode-handler");
        
        std::wstring lineName(L"line");
        uint x1(300), y1(600), x2(600), y2(620);
        uint line_width(5);

        std::wstring polygonName(L"polygon");
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220,300},{110,330}};
        uint num_coordinates(4);
        uint border_width(3);

        std::wstring colorName(L"opaque-white");
        double red(1.0), green(1.0), blue(1.0), alpha(1.0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), trackerW, trackerH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_new(odePphName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tiler_pph_add(tilerName.c_str(), odePphName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_color_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
            x1, y1, x2, y2, line_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
            border_width, colorName.c_str())== DSL_RESULT_SUCCESS );
            

        REQUIRE( dsl_ode_trigger_always_new(odeAlwaysTriggerName.c_str(), 
            NULL, DSL_ODE_PRE_OCCURRENCE_CHECK) == DSL_RESULT_SUCCESS );

        const wchar_t* display_types[] = {L"line", L"polygon", NULL};

        REQUIRE( dsl_ode_action_display_meta_add_many_new(odeDisplayMetaActionName.c_str(), 
            display_types) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_ode_trigger_action_add(odeAlwaysTriggerName.c_str(), 
            odeDisplayMetaActionName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pph_ode_trigger_add(odePphName.c_str(), odeAlwaysTriggerName.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(osdName.c_str(), textEnabled, clockEnabled) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"primary-gie", L"ktl-tracker", L"tiler", L"osd", L"window-sink", NULL};
        
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
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_list_size() == 0 );
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
