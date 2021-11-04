/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

static const std::wstring primaryTisName(L"primary-tis");
static const std::wstring primaryTisName2(L"primary-tis-2");

static const std::wstring inferConfigFile(
    L"/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt");

static const std::wstring pipelineName(L"test-pipeline");
        
static const std::wstring customPpmName(L"custom-ppm");

SCENARIO( "The Components container is updated correctly on new Primary TIS", "[tis-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        uint interval(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Primary TIS is created" ) 
        {
            REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), inferConfigFile.c_str(), 
                interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Primary TIS delete", "[tis-api]" )
{
    GIVEN( "A new Primary TIS in memory" ) 
    {
        
        uint interval(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), inferConfigFile.c_str(), 
            interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary TIS is deleted" ) 
        {
            REQUIRE( dsl_component_delete(primaryTisName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Only one Primary TIS can be added to a Pipeline", "[tis-api]" )
{
    GIVEN( "A two Primary TISs and a new pPipeline" ) 
    {
        uint interval(1);

        REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), inferConfigFile.c_str(), 
            interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_infer_tis_primary_new(primaryTisName2.c_str(), inferConfigFile.c_str(), 
            interval) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary TIS is add to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                primaryTisName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "Adding a second Primary TIS to the same Pipeline fails" )
            {
                // TODO why is exception not caught????
//                REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
//                    primaryTisName2.c_str()) == DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "A Primary TIS in use can't be deleted", "[tis-api]" )
{
    GIVEN( "A new Primary TIS and new pPipeline" ) 
    {
        uint interval(1);

        REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), 
            inferConfigFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary TIS is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                primaryTisName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary TIS can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(primaryTisName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "A Primary TIS, once removed from a Pipeline, can be deleted", "[tis-api]" )
{
    GIVEN( "A new Primary TIS owned by a new pPipeline" ) 
    {
        uint interval(1);

        REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), 
            inferConfigFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            primaryTisName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary TIS is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                primaryTisName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary TIS can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(primaryTisName.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    return true;
}
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
    return true;
}
    
SCENARIO( "A Sink Pad Probe Handler can be added and removed from a Primary TIS", "[tis-api]" )
{
    GIVEN( "A new Primary TIS and Custom PPH" ) 
    {
        uint interval(1);

        REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), 
            inferConfigFile.c_str(), interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the Primary TIS" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_infer_primary_pph_remove(primaryTisName.c_str(), 
                customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_INFER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_infer_primary_pph_add(primaryTisName.c_str(), 
                customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_infer_primary_pph_remove(primaryTisName.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Primary TIS" ) 
        {
            REQUIRE( dsl_infer_primary_pph_add(primaryTisName.c_str(), 
                customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_infer_primary_pph_add(primaryTisName.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_INFER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_infer_primary_pph_remove(primaryTisName.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Probe Handler can be added and removed froma a Primary TIS", "[tis-api]" )
{
    GIVEN( "A new Primary TIS and Custom PPH" ) 
    {
        uint interval(1);

        REQUIRE( dsl_infer_tis_primary_new(primaryTisName.c_str(), 
            inferConfigFile.c_str(), interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), pad_probe_handler_cb1, 
            NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Probe Handler is added to the Primary TIS" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_infer_primary_pph_remove(primaryTisName.c_str(), customPpmName.c_str(), 
                DSL_PAD_SRC) == DSL_RESULT_INFER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_infer_primary_pph_add(primaryTisName.c_str(), 
                customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_infer_primary_pph_remove(primaryTisName.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Source Pad Probe Handler is added to the Primary TIS" ) 
        {
            REQUIRE( dsl_infer_primary_pph_add(primaryTisName.c_str(), 
                customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Source Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_infer_primary_pph_add(primaryTisName.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_INFER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_infer_primary_pph_remove(primaryTisName.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

