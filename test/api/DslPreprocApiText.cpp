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
#include "DslApi.h"

static const std::wstring preproc_config(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_preprocess.txt");

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring preproc_name(L"preprocessor");


SCENARIO( "The Components container is updated correctly on new Preprocessor",
    "[preproc-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Preprocessor is created" ) 
        {
            REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
                preproc_config.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on Preprocessor delete",
    "[preproc-api]" )
{
    GIVEN( "A new Preprocessor in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Preprocessor is created" ) 
        {
            REQUIRE( dsl_component_delete(preproc_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "An Preprocessor in use can't be deleted",
    "[preproc-api]" )
{
    GIVEN( "A new Preprocessor and new pPipeline" ) 
    {
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Preprocessor is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                preproc_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Preprocessor can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(preproc_name.c_str()) == 
                    DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Preprocessor, once removed from a Pipeline, can be deleted",
    "[preproc-api]" )
{
    GIVEN( "A new pPipeline with a child Preprocessor" ) 
    {
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
            preproc_name.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Preprocessor is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                preproc_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Preprocessor can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(preproc_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Preprocessor in use can't be added to a second Pipeline",
    "[preproc-api]" )
{
    GIVEN( "A new Preprocessor and two new pPipelines" ) 
    {
        std::wstring pipeline_name1(L"test-pipeline-1");
        std::wstring pipeline_name2(L"test-pipeline-2");
        std::wstring preproc_name(L"preprocessor");

        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Preprocessor is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name1.c_str(), 
                preproc_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Preprocessor can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipeline_name2.c_str(), 
                    preproc_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Preprocessor can Set and Get its Config File correctly",
    "[preproc-api]" )
{
    GIVEN( "A new Preprocessor and new pPipeline" ) 
    {
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );

        const wchar_t* pRetConfigFile;
        REQUIRE( dsl_preproc_config_file_get(preproc_name.c_str(), 
            &pRetConfigFile) == DSL_RESULT_SUCCESS );
        std::wstring retConfigFile(pRetConfigFile);
        REQUIRE( retConfigFile == preproc_config );
        
        WHEN( "The Preprocessor's Config File is set" )
        {
            std::wstring new_preproc_config(
                L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-action-recognition/config_preprocess_3d_custom.txt");
            REQUIRE( dsl_preproc_config_file_set(preproc_name.c_str(), 
                new_preproc_config.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct Config File is returned on get" )
            {
                REQUIRE( dsl_preproc_config_file_get(preproc_name.c_str(), 
                    &pRetConfigFile) == DSL_RESULT_SUCCESS );
                retConfigFile = pRetConfigFile;
                REQUIRE( retConfigFile == new_preproc_config );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Preprocessor can be disabled correctly",
    "[preproc-api]" )
{
    GIVEN( "A new Preprocessor and new pPipeline" ) 
    {
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );

        // check default state first
        boolean enabled(false);
        REQUIRE( dsl_preproc_enabled_get(preproc_name.c_str(), 
            &enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( enabled == DSL_TRUE );
        
        WHEN( "The Preprocessor is disabled" )
        {
            REQUIRE( dsl_preproc_enabled_set(preproc_name.c_str(), 
                false) == DSL_RESULT_SUCCESS );

            THEN( "The correct Config File is returned on get" )
            {
                REQUIRE( dsl_preproc_enabled_get(preproc_name.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == DSL_FALSE );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Preprocessor returns its unique id correctly",  "[preproc-api]" )
{
    GIVEN( "Attributes for a new  Preprocessor" ) 
    {
        // Defined at the top of the file

        WHEN( "When a new Preprocess is created" )
        {
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct unique id is returned" )
            {
                uint retId(99);
                REQUIRE( dsl_preproc_unique_id_get(preproc_name.c_str(), 
                    &retId) == DSL_RESULT_SUCCESS );

                REQUIRE( retId == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Preprocessor API checks for NULL input parameters", "[preproc-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        const wchar_t* pRetConfigFile;
        uint retId(0);
        boolean enabled(0);
               
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_preproc_new(NULL, 
                    preproc_config.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_config_file_get(NULL, 
                    &pRetConfigFile) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_config_file_get(preproc_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_config_file_set(NULL, 
                    preproc_config.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_config_file_set(preproc_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_enabled_get(NULL, 
                    &enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_enabled_get(preproc_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_unique_id_get(NULL, 
                    &retId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_preproc_unique_id_get(preproc_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}
