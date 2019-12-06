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

SCENARIO( "A new Pipeline with an On-Screen Display can be updated", "[PipelineOsd]" )
{
    GIVEN( "A Pipeline with four sources and minimal components" ) 
    {
        std::wstring sourceName1 = L"uri-source";
        std::wstring uri = L"./test/streams/sample_1080p_h264.mp4";
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tiledDisplayName = L"tiled-display";
        uint width(1280);
        uint height(720);

        std::wstring primaryGieName = L"primary-gie";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_fp16.engine";
        
        std::wstring onScreenDisplayName = L"on-screen-display";
        bool isClockEnabled(false);

        std::wstring overlaySinkName = L"overlay-sink";
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        // new display for this scenario
        REQUIRE( dsl_display_new(tiledDisplayName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_osd_new(onScreenDisplayName.c_str(), isClockEnabled) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {L"uri-source", L"primary-gie", 
            L"tiled-display", L"on-screen-display", L"overlay-sink", NULL};
        
        WHEN( "When the Display Tiles are set, and the Pipeline is Assembled and Played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            THEN( "The OSD settings are correct and can't be updated" )
            {
                bool currIsClockEnabled(false);
                
//                REQUIRE( dsl_display_tiles_get(tiledDisplayName.c_str(), &currRows, &currCols) == DSL_RESULT_SUCCESS );
//                REQUIRE( currIsClockEnabled == isClockEnabled );
        
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}