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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(500)

SCENARIO( "A new Pipeline with a Tiled Display can be updated", "[PipelineTiler]" )
{
    GIVEN( "A Pipeline with four sources and minimal components" ) 
    {
        std::wstring sourceName1 = L"test-uri-source-1";
        std::wstring sourceName2 = L"test-uri-source-2";
        std::wstring sourceName3 = L"test-uri-source-3";
        std::wstring uri = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";
        uint cudadecMemType(DSL_NVBUF_MEM_TYPE_DEFAULT);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName = L"tiler";
        uint width(1280);
        uint height(720);

        std::wstring windowSinkName(L"egl-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName3.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        // new tiler for this scenario
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
    
            
        const wchar_t* components[] = {L"test-uri-source-1", L"test-uri-source-2", L"test-uri-source-3", 
            L"tiler", L"egl-sink", NULL};
        
        WHEN( "When the Display Tiles are set, and the Pipeline is Assembled and Played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_tiler_tiles_set(tilerName.c_str(), 2, 2) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tiler_dimensions_set(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            THEN( "The Display settings are correct and can be updated" )
            {
                uint currRows(0), currCols(0), currWidth(0), currHeight(0);
                
                REQUIRE( dsl_tiler_tiles_get(tilerName.c_str(), &currCols, &currRows) == DSL_RESULT_SUCCESS );
                REQUIRE( currCols == 2 );
                REQUIRE( currRows == 2 );

                REQUIRE( dsl_tiler_dimensions_get(tilerName.c_str(), &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( currWidth == width );
                REQUIRE( currHeight == height );

                uint newWidth(428), newHeight(720);
                REQUIRE( dsl_tiler_tiles_set(tilerName.c_str(), 1, 3) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tiler_dimensions_set(tilerName.c_str(), newWidth, newHeight) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_tiler_tiles_get(tilerName.c_str(), &currCols, &currRows) == DSL_RESULT_SUCCESS );
                REQUIRE( currCols == 1 );
                REQUIRE( currRows == 3 );
                REQUIRE( dsl_tiler_dimensions_get(tilerName.c_str(), &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );
        
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }

        WHEN( "When the Display Tiles are set to 2x2" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_tiler_tiles_set(tilerName.c_str(), 2, 2) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tiler_dimensions_set(tilerName.c_str(), 1280, 720) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "a Tilers Show Source Settings can be updated" )
            {

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_tiler_source_show_set(tilerName.c_str(), 
                    sourceName1.c_str(), 0, true) == DSL_RESULT_SUCCESS);
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_tiler_source_show_all(tilerName.c_str()) == DSL_RESULT_SUCCESS);
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                
                // Show cycle must fail if no timeout
                REQUIRE( dsl_tiler_source_show_cycle(tilerName.c_str(), 
                    0) == DSL_RESULT_TILER_SET_FAILED);
                
                // Show cycle with timeout must succeed
                REQUIRE( dsl_tiler_source_show_cycle(tilerName.c_str(), 
                    1) == DSL_RESULT_SUCCESS);
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_tiler_source_show_all(tilerName.c_str()) == DSL_RESULT_SUCCESS);
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                // Selecting an open tile must fail
                REQUIRE( dsl_tiler_source_show_select(tilerName.c_str(), 
                    sinkW-10, sinkH-10, sinkW, sinkH, 1) == DSL_RESULT_SUCCESS);

                // Selecting a valid tile must succeed
                REQUIRE( dsl_tiler_source_show_select(tilerName.c_str(), 
                    10, 10, sinkW, sinkH, 1) == DSL_RESULT_SUCCESS);
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