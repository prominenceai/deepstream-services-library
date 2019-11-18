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

SCENARIO( "The Components container is updated correctly on new Primary GIE", "[gie-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring primaryGieName = L"primary-gie";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint uniqueId(1);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        WHEN( "A new Primary GIE is created" ) 
        {

            REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
                modelEngineFile.c_str(), interval, uniqueId) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( *(dsl_component_list_all()) != NULL );
                
                std::wstring returnedName = *(dsl_component_list_all());
                REQUIRE( returnedName == primaryGieName );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    


SCENARIO( "The Components container is updated correctly on Primary GIE delete", "[gie-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring primaryGieName = L"primary-gie";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint uniqueId(1);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval, uniqueId) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary GIE is deleted" ) 
        {
            REQUIRE( dsl_component_delete(primaryGieName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( *(dsl_component_list_all()) == NULL );
            }
        }
    }
}

SCENARIO( "Only one Primary GIE can be added to a Pipeline", "[gie-api]" )
{
    GIVEN( "A two Primary GIEs and a new pPipeline" ) 
    {
        std::wstring primaryGieName1 = L"primary-gie-1";
        std::wstring primaryGieName2 = L"primary-gie-2";
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint uniqueId(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName1.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval, uniqueId) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gie_primary_new(primaryGieName2.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval, uniqueId) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary GIE is add to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                primaryGieName1.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "Adding a second Primary GIE to the same Pipeline fails" )
            {
                // TODO why is exception not caught????
//                REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
//                    primaryGieName2.c_str()) == DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED );
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

SCENARIO( "A Primary GIE in use can't be deleted", "[gie-api]" )
{
    GIVEN( "A new Primary GIE and new pPipeline" ) 
    {
        std::wstring primaryGieName = L"primary-gie";
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint uniqueId(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval, uniqueId) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary GIE is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary GIE can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(primaryGieName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
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

SCENARIO( "A Primary GIE, once removed from a Pipeline, can be deleted", "[gie-api]" )
{
    GIVEN( "A new Primary GIE owned by a new pPipeline" ) 
    {
        std::wstring primaryGieName = L"primary-gie";
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint uniqueId(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval, uniqueId) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary GIE is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary GIE can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(primaryGieName.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
    }
}

