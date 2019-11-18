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

SCENARIO( "The Components container is updated correctly on new source", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName  = L"csi-source";

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        WHEN( "A new Source is created" ) 
        {

            REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( *(dsl_component_list_all()) != NULL );
                
                std::wstring returnedName = *(dsl_component_list_all());
                REQUIRE( returnedName == sourceName );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    
    
SCENARIO( "The Components container is updated correctly on Source Delete", "[source-api]" )
{
    GIVEN( "One Source im memory" ) 
    {
        std::wstring sourceName  = L"csi-source";

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( *(dsl_component_list_all()) != NULL );
        
        WHEN( "The Source is deleted" )
        {
            REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( *(dsl_component_list_all()) == NULL );
            }
        }
    }
}

SCENARIO( "A Source in use can't be deleted", "[source-api]" )
{
    GIVEN( "A new Source and new pPipeline" ) 
    {
        std::wstring sourceName  = L"csi-source";
        std::wstring pipelineName  = L"test-pipeline";

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
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

SCENARIO( "A Source, once removed from a Pipeline, can be deleted", "[source-api]" )
{
    GIVEN( "A new Pipeline with a Child CSI Source" ) 
    {
        std::wstring sourceName  = L"csi-source";
        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Source is removed from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can be deleted successfully" ) 
            {
                REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
    }
}

SCENARIO( "A new CSI Camaera Source is live", "[source]" )
{
    std::wstring sourceName  = L"csi-source";

    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        WHEN( "A new Source is created" ) 
        {

            REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_source_is_live(sourceName.c_str()) == true );
            }
        }
        
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    


SCENARIO( "A Client is able to update the Source in-use max" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
        REQUIRE( dsl_source_get_num_in_use_max() == DSL_DEFAULT_SOURCE_IN_USE_MAX );
        REQUIRE( dsl_source_get_num_in_use() == 0 );
        
        WHEN( "The in-use-max is updated by the client" )   
        {
            uint new_max = 128;
            
            dsl_source_set_num_in_use_max(new_max);
            
            THEN( "The new in-use-max will be returned to the client on get" )
            {
                REQUIRE( dsl_source_get_num_in_use_max() == new_max );
            }
        }
    }
}

SCENARIO( "A Source added to a Pipeline updates the in-use number", "[source-api]" )
{
    std::wstring sourceName  = L"csi-source";
    std::wstring pipelineName  = L"test-pipeline";

    GIVEN( "A new Source and new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_get_num_in_use() == 0 );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_get_num_in_use() == 1 );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "A Source removed from a Pipeline updates the in-use number", "[source-api]" )
{
    GIVEN( "A new Pipeline with a Source" ) 
    {
        std::wstring sourceName  = L"csi-source";
        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_get_num_in_use() == 1 );

        WHEN( "The Source is removed from, the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_get_num_in_use() == 0 );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "Adding multiple Sources to a Pipelines updates the in-use number", "[source-api]" )
{
    std::wstring sourceName1  = L"csi-source1";
    std::wstring pipelineName1  = L"test-pipeline1";
    std::wstring sourceName2  = L"csi-source2";
    std::wstring pipelineName2  = L"test-pipeline2";

    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(sourceName1.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_get_num_in_use() == 0 );

        WHEN( "Each Sources is added to a different Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sourceName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sourceName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_get_num_in_use() == 2 );
            }
        }
        REQUIRE( dsl_pipeline_component_remove(pipelineName1.c_str(), 
            sourceName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_component_remove(pipelineName2.c_str(), 
            sourceName2.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_get_num_in_use() == 0 );
    }
}
