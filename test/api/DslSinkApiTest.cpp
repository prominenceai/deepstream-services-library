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
#include "DslApi.h"

SCENARIO( "The Components container is updated correctly on new Overlay Sink", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        WHEN( "A new Primary GIE is created" ) 
        {

            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( *(dsl_component_list_all()) != NULL );
                
                std::wstring returnedName = *(dsl_component_list_all());
                REQUIRE( returnedName == overlaySinkName );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Overlay Sink delete", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A new Overlay Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( *(dsl_component_list_all()) == NULL );
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be deleted", "[sink-api]" )
{
    GIVEN( "A new Overlay Sinak and new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Primary GIE is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary GIE can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
    }
}

SCENARIO( "An Overlay Sink, once removed from a Pipeline, can be deleted", "[sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary GIE is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary GIE can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
    }
}

