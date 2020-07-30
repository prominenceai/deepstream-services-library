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

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Primary GIE is created" ) 
        {

            REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
                modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Primary GIE delete", "[gie-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        std::wstring primaryGieName = L"primary-gie";
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary GIE is deleted" ) 
        {
            REQUIRE( dsl_component_delete(primaryGieName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
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

        REQUIRE( dsl_gie_primary_new(primaryGieName1.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gie_primary_new(primaryGieName2.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
            
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
        REQUIRE( dsl_component_list_size() == 0 );
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

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
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
        REQUIRE( dsl_component_list_size() == 0 );
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

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
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
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

static boolean batch_meta_handler_cb1(void* batch_meta, void* user_data)
{
}
static boolean batch_meta_handler_cb2(void* batch_meta, void* user_data)
{
}
    
SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a Primary GIE", "[gie-api]" )
{
    GIVEN( "A new pPipeline with a new Primary GIE" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";

        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Batch Meta Handler is added to the Primary GIE" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_gie_primary_batch_meta_handler_remove(primaryGieName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_GIE_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_gie_primary_batch_meta_handler_add(primaryGieName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_gie_primary_batch_meta_handler_remove(primaryGieName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Batch Meta Handler can be added and removed froma a Primary GIE", "[gie-api]" )
{
    GIVEN( "A new pPipeline with a new Primary GIE" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";

        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Batch Meta Handler is added to the Primary GIE" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_gie_primary_batch_meta_handler_remove(primaryGieName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_GIE_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_gie_primary_batch_meta_handler_add(primaryGieName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_gie_primary_batch_meta_handler_remove(primaryGieName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The same Sink Pad Meta Batch Handler can not be added to a Primary GIE twice", "[gie-api]" )
{
    GIVEN( "A new pPipeline with a new Primary GIE" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";

        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Meta Batch Handler is added to the Primary GIE " ) 
        {
            REQUIRE( dsl_gie_primary_batch_meta_handler_add(primaryGieName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE( dsl_gie_primary_batch_meta_handler_add(primaryGieName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_GIE_HANDLER_ADD_FAILED );
                
                REQUIRE( dsl_gie_primary_batch_meta_handler_remove(primaryGieName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A same Source Pad Meta Batch Handler can not be added to a Primary GIE twice", "[gie-api]" )
{
    GIVEN( "A new Pipeline with a new Primary GIE" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";

        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            primaryGieName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Meta Batch Handler is added to the Primary GIE " ) 
        {
            REQUIRE( dsl_gie_primary_batch_meta_handler_add(primaryGieName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE( dsl_gie_primary_batch_meta_handler_add(primaryGieName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_GIE_HANDLER_ADD_FAILED );
                
                REQUIRE( dsl_gie_primary_batch_meta_handler_remove(primaryGieName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE can Enable and Disable raw layer info output",  "[gie-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        
        WHEN( "The Primary GIE's raw output is enabled" )
        {
            REQUIRE( dsl_gie_raw_output_enabled_set(primaryGieName.c_str(), true, L"./") == DSL_RESULT_SUCCESS );

            THEN( "The raw output can then be disabled" )
            {
                REQUIRE( dsl_gie_raw_output_enabled_set(primaryGieName.c_str(), false, L"") == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE fails to Enable raw layer info output given a bad path",  "[gie-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";

        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );
        
        WHEN( "A bad path is constructed" )
        {
            std::wstring badPath(L"this/is/an/invalid/path");
            
            THEN( "The raw output will fail to enale" )
            {
                REQUIRE( dsl_gie_raw_output_enabled_set(primaryGieName.c_str(), true, badPath.c_str()) == DSL_RESULT_GIE_OUTPUT_DIR_DOES_NOT_EXIST );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Secondary GIE can Set and Get its Infer Config and Model Engine Files",  "[gie-api]" )
{
    GIVEN( "A new Secondary GIE in memory" ) 
    {
        std::wstring primaryGieName = L"primary-gie";
        std::wstring secondaryGieName = L"secondary-gie";
        std::wstring inferConfigFile = L"./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Secondary_CarColor/resnet18.caffemodel";
        uint interval(1);
        
        REQUIRE( dsl_gie_secondary_new(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval) == DSL_RESULT_SUCCESS );

        const wchar_t* pRetInferConfigFile;
        REQUIRE( dsl_gie_infer_config_file_get(secondaryGieName.c_str(), &pRetInferConfigFile) == DSL_RESULT_SUCCESS );
        std::wstring retInferConfigFile(pRetInferConfigFile);
        REQUIRE( retInferConfigFile == inferConfigFile );
        
        const wchar_t* pRetModelEngineFile;
        REQUIRE( dsl_gie_model_engine_file_get(secondaryGieName.c_str(), &pRetModelEngineFile) == DSL_RESULT_SUCCESS );
        std::wstring retModelEngineFile(pRetModelEngineFile);
        REQUIRE( retModelEngineFile == modelEngineFile );
        
        WHEN( "The SecondaryGieBintr's Infer Config File and Model Engine are set" )
        {
            std::wstring newInferConfigFile = L"./test/configs/config_infer_secondary_carmake_nano.txt";
            REQUIRE( dsl_gie_infer_config_file_set(secondaryGieName.c_str(), newInferConfigFile.c_str()) == DSL_RESULT_SUCCESS );

            std::wstring newModelEngineFile = L"./test/models/Secondary_CarMake/resnet18.caffemodel";
            REQUIRE( dsl_gie_model_engine_file_set(secondaryGieName.c_str(), newModelEngineFile.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct Files are returned on get" )
            {
                REQUIRE( dsl_gie_infer_config_file_get(secondaryGieName.c_str(), 
                    &pRetInferConfigFile) == DSL_RESULT_SUCCESS);
                retInferConfigFile.assign(pRetInferConfigFile);
                REQUIRE( retInferConfigFile == newInferConfigFile );
                REQUIRE( dsl_gie_model_engine_file_get(secondaryGieName.c_str(), 
                    &pRetModelEngineFile) == DSL_RESULT_SUCCESS);
                retModelEngineFile.assign(pRetModelEngineFile);
                REQUIRE( retModelEngineFile == newModelEngineFile );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE can Get and Set its Interval",  "[gie-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        std::wstring primaryGieName(L"primary-gie");
        std::wstring inferConfigFile = L"./test/configs/config_infer_primary_nano.txt";
        std::wstring modelEngineFile = L"./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        uint interval(1);

        REQUIRE( dsl_gie_primary_new(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval) == DSL_RESULT_SUCCESS );

        uint retInterval(0);
        REQUIRE( dsl_gie_interval_get(primaryGieName.c_str(), &retInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( retInterval == interval );
        
        WHEN( "The Primary GIE's Interval is set" )
        {
            uint newInterval(5);
            REQUIRE( dsl_gie_interval_set(primaryGieName.c_str(), newInterval) == DSL_RESULT_SUCCESS );

            THEN( "The correct Interval is returned on get" )
            {
                REQUIRE( dsl_gie_interval_get(primaryGieName.c_str(), &retInterval) == DSL_RESULT_SUCCESS );
                REQUIRE( retInterval == newInterval );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}


