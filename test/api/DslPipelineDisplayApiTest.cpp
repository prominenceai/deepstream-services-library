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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(300)

SCENARIO( "A new Pipeline with a Tiled Display can be updated", "[PipelineDisplay]" )
{
    GIVEN( "A Pipeline with four sources and minimal components" ) 
    {
        std::wstring sourceName1 = L"test-uri-source-1";
        std::wstring sourceName2 = L"test-uri-source-2";
        std::wstring sourceName3 = L"test-uri-source-3";
        std::wstring uri = L"./test/streams/sample_1080p_h264.mp4";
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tiledDisplayName = L"tiled-display-name";
        uint width(1280);
        uint height(720);

        std::wstring overlaySinkName = L"overlay-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), cudadecMemType, 
            intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), cudadecMemType, 
            intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName3.c_str(), uri.c_str(), cudadecMemType, 
            intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        // overlay sink for observation 
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        // new display for this scenario
        REQUIRE( dsl_display_new(tiledDisplayName.c_str(), width, height) == DSL_RESULT_SUCCESS );
    
            
        const wchar_t* components[] = {L"test-uri-source-1", L"test-uri-source-2", L"test-uri-source-3", 
            L"tiled-display-name", L"overlay-sink", NULL};
        
        WHEN( "When the Display Tiles are set, and the Pipeline is Assembled and Played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_display_tiles_set(tiledDisplayName.c_str(), 2, 4) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_dimensions_set(tiledDisplayName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            THEN( "The Display settings are correct and can't be updated" )
            {
                uint currRows(0), currCols(0), currWidth(0), currHeight(0);
                
                REQUIRE( dsl_display_tiles_get(tiledDisplayName.c_str(), &currRows, &currCols) == DSL_RESULT_SUCCESS );
                REQUIRE( currRows == 2 );
                REQUIRE( currCols == 4 );

                REQUIRE( dsl_display_dimensions_get(tiledDisplayName.c_str(), &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( currWidth == width );
                REQUIRE( currHeight == height );

                REQUIRE( dsl_display_tiles_set(tiledDisplayName.c_str(), 4, 4) == DSL_RESULT_DISPLAY_IS_IN_USE );
                REQUIRE( dsl_display_dimensions_set(tiledDisplayName.c_str(), 200, 200) == DSL_RESULT_DISPLAY_IS_IN_USE );
        
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }

        WHEN( "When the Display Tiles are set to a single Row multiple Source are display correctly" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_display_tiles_set(tiledDisplayName.c_str(), 1, 3) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_display_dimensions_set(tiledDisplayName.c_str(), 1280, 240) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            THEN( "The Display settings are correct and can't be updated" )
            {
                uint rows(0), cols(0);
                REQUIRE( dsl_display_tiles_get(tiledDisplayName.c_str(), &rows, &cols) == DSL_RESULT_SUCCESS );
                
                REQUIRE( rows == 1 );
                REQUIRE( cols == 3 );


                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }

    }
}