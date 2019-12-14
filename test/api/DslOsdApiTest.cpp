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

SCENARIO( "The Components container is updated correctly on new OSD", "[osd-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new OSD is created" ) 
        {

            REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on OSD delete", "[osd-api]" )
{
    GIVEN( "A new OSD in memory" ) 
    {
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new OSD is created" ) 
        {
            REQUIRE( dsl_component_delete(osdName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "An OSD in use can't be deleted", "[osd-api]" )
{
    GIVEN( "A new OSD and new pPipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The OSD is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                osdName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The OSD can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(osdName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An OSD, once removed from a Pipeline, can be deleted", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a child OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The OSD is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                osdName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The OSD can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(osdName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An OSD in use can't be added to a second Pipeline", "[osd-api]" )
{
    GIVEN( "A new OSD and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The OSD is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                osdName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The OSD can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    osdName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static void batch_meta_handler_cb1(void* batch_meta, void* user_data)
{
}
static void batch_meta_handler_cb2(void* batch_meta, void* user_data)
{
}
    
SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a OSD", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Batch Meta Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE ( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SINK) == DSL_RESULT_OSD_HANDLER_REMOVE_FAILED );

            REQUIRE ( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE ( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Batch Meta Handler can be added and removed froma a OSD", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Batch Meta Handler is added to the OSD" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE ( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SRC) == DSL_RESULT_OSD_HANDLER_REMOVE_FAILED );

            REQUIRE ( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE ( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A second Sink Pad Meta Batch Handler can not be added to a OSD", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Meta Batch Handler is added to the OSD " ) 
        {
            REQUIRE ( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Sink Pad Meta Batch Handler can not be added" ) 
            {
                REQUIRE ( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_OSD_HANDLER_ADD_FAILED );
                
                REQUIRE ( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A second Source Pad Meta Batch Handler can not be added to a OSD", "[osd-api]" )
{
    GIVEN( "A new pPipeline with a new OSD" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring osdName(L"on-screen-display");

        REQUIRE( dsl_osd_new(osdName.c_str(), false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            osdName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Meta Batch Handler is added to the OSD " ) 
        {
            REQUIRE ( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Sink Pad Meta Batch Handler can not be added" ) 
            {
                REQUIRE ( dsl_osd_batch_meta_handler_add(osdName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_OSD_HANDLER_ADD_FAILED );
                
                REQUIRE ( dsl_osd_batch_meta_handler_remove(osdName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}
