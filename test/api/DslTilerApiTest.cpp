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
#include "Dsl.h"

SCENARIO( "The Components container is updated correctly on new Tiled Display", "[tiler-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Tiled Display is created" ) 
        {

            REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on Tiled Display delete", "[tiler-api]" )
{
    GIVEN( "A new Tiled Display in memory" ) 
    {
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Tiled Display is deleted" ) 
        {
            REQUIRE( dsl_component_delete(tilerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tiled Display in use can't be deleted", "[tiler-api]" )
{
    GIVEN( "A new Tiled Display and new pPipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Tiled Display is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                tilerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tiled Display can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(tilerName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tiled Display, once removed from a Pipeline, can be deleted", "[tiler-api]" )
{
    GIVEN( "A new pPipeline with a child Tiled Display" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            tilerName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Tiled Display is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                tilerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tiled Display can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(tilerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Tiled Display in use can't be added to a second Pipeline", "[tiler-api]" )
{
    GIVEN( "A new Tiled Display and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Tiled Display is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                tilerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tiled Display can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    tilerName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static boolean batch_meta_handler_cb1(void* batch_meta, void* user_data)
{
}
static boolean batch_meta_handler_cb2(void* batch_meta, void* user_data)
{
}
    
SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a Tiled Display", "[tiler-api]" )
{
    GIVEN( "A new pPipeline with a new Tiled Display" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            tilerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Batch Meta Handler is added to the Tiled Display" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_tiler_batch_meta_handler_remove(tilerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_TILER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_tiler_batch_meta_handler_add(tilerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_tiler_batch_meta_handler_remove(tilerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Batch Meta Handler can be added and removed froma a Tiled Display", "[tiler-api]" )
{
    GIVEN( "A new pPipeline with a new Tiled Display" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            tilerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Batch Meta Handler is added to the Tiled Display" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_tiler_batch_meta_handler_remove(tilerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_TILER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_tiler_batch_meta_handler_add(tilerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_tiler_batch_meta_handler_remove(tilerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The same Sink Pad Meta Batch Handler can not be added to a Tiled Display twice", "[tiler-api]" )
{
    GIVEN( "A new pPipeline with a new Tiled Display" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            tilerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Meta Batch Handler is added to the Tiled Display " ) 
        {
            REQUIRE( dsl_tiler_batch_meta_handler_add(tilerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE( dsl_tiler_batch_meta_handler_add(tilerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_TILER_HANDLER_ADD_FAILED );
                
                REQUIRE( dsl_tiler_batch_meta_handler_remove(tilerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The same Source Pad Meta Batch Handler can not be added to a Tiled Display twice", "[tiler-api]" )
{
    GIVEN( "A new pPipeline with a new Tiled Display" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            tilerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Meta Batch Handler is added to the Tiled Display " ) 
        {
            REQUIRE( dsl_tiler_batch_meta_handler_add(tilerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE( dsl_tiler_batch_meta_handler_add(tilerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_TILER_HANDLER_ADD_FAILED );
                
                REQUIRE( dsl_tiler_batch_meta_handler_remove(tilerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An invalid Tiler is caught by all Set and Get API calls", "[tiler-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Tiler Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            

        WHEN( "The Tiler Get-Set APIs are called with a Fake sink" )
        {
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The Tiler Get-Set APIs fail correctly")
            {
                uint width(0), height(0);
                uint rows(0), cols(0);
                const wchar_t* config;
                
                REQUIRE( dsl_tiler_dimensions_get(fakeSinkName.c_str(), &width, &height) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);
                REQUIRE( dsl_tiler_dimensions_set(fakeSinkName.c_str(), 500, 300) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);
                REQUIRE( dsl_tiler_tiles_get(fakeSinkName.c_str(), &rows, &cols) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);
                REQUIRE( dsl_tiler_tiles_set(fakeSinkName.c_str(), 1, 1) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);

                REQUIRE ( dsl_tiler_batch_meta_handler_add(fakeSinkName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE ( dsl_tiler_batch_meta_handler_remove(fakeSinkName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tiled Display can Set and Get all properties", "[tiler-api]" )
{
    GIVEN( "A new pPipeline with a new Tiled Display" ) 
    {
        std::wstring tilerName(L"tiler");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );

        WHEN( "A Tiler's Tiles are Set " ) 
        {
            uint newCols(3), newRows(2), retCols(0), retRows(0);
            REQUIRE( dsl_tiler_tiles_set(tilerName.c_str(), newRows, newCols) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                REQUIRE( dsl_tiler_tiles_get(tilerName.c_str(), &retRows, &retCols) == DSL_RESULT_SUCCESS);
                REQUIRE( retRows == newRows );
                REQUIRE( retCols == newCols );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Tiler's Dimensions are Set " ) 
        {
            uint newWidth(640), newHeight(360), retWidth(0), retHeight(0);
            REQUIRE( dsl_tiler_dimensions_set(tilerName.c_str(), newWidth, newHeight) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                REQUIRE( dsl_tiler_dimensions_get(tilerName.c_str(), &retWidth, &retHeight) == DSL_RESULT_SUCCESS);
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == newHeight );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "Setting A Tiler's Show Source when the Tiler is in a non-linked state" ) 
        {
            std::wstring anySourceName(L"any-source-name");
            
            REQUIRE( dsl_tiler_source_show_set(tilerName.c_str(), anySourceName.c_str()) == DSL_RESULT_TILER_SET_FAILED);
            THEN( "The Set will fail and the current Show Source is unchanged" ) 
            {
                const wchar_t* retSource;
                REQUIRE( dsl_tiler_source_show_get(tilerName.c_str(), &retSource) == DSL_RESULT_SUCCESS);
                REQUIRE( retSource == NULL );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

