/*
The MIT License

Copyright (c) 2023, Prominence AI, Inc.

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

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static std::wstring dewarper_name(L"dewarper");

static std::wstring dewarper_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper.txt");

static uint camera_id(6); 


SCENARIO( "A new Dewarper returns the correct attribute values", "[dewarper-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Dewarper is created" ) 
        {
            REQUIRE( dsl_dewarper_new(dewarper_name.c_str(), 
                dewarper_config_file.c_str(), camera_id) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_list_size() == 1 );

            THEN( "All default attributes are returned correctly" ) 
            {
                const wchar_t* ret_c_config_file;
                REQUIRE( dsl_dewarper_config_file_get(dewarper_name.c_str(), 
                    &ret_c_config_file) == DSL_RESULT_SUCCESS );
                std::wstring ret_config_file(ret_c_config_file);
                REQUIRE( ret_config_file == dewarper_config_file );
                
                uint ret_camera_id(0);
                REQUIRE( dsl_dewarper_camera_id_get(dewarper_name.c_str(),
                    &ret_camera_id) ==  DSL_RESULT_SUCCESS );
                REQUIRE( ret_camera_id == camera_id );
                
                uint ret_num_batch_buffers(0);
                uint def_num_batch_buffers(4); // as specified in config file
                REQUIRE( dsl_dewarper_num_batch_buffers_get(dewarper_name.c_str(),
                    &ret_num_batch_buffers) ==  DSL_RESULT_SUCCESS );
                REQUIRE( ret_num_batch_buffers == def_num_batch_buffers );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new Dewarper returns can update its settings correctly", "[dewarper-api]" )
{
    GIVEN( "A new Dewarper component" ) 
    {
        REQUIRE( dsl_dewarper_new(dewarper_name.c_str(), 
            dewarper_config_file.c_str(), camera_id) == DSL_RESULT_SUCCESS );

        WHEN( "A Dewarper's config-file is set" ) 
        {
            std::wstring new_dewarper_config_file(
                L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper_perspective.txt");
            REQUIRE( dsl_dewarper_config_file_set(dewarper_name.c_str(), 
                new_dewarper_config_file.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct config-file is returned on get" ) 
            {
                const wchar_t* ret_c_config_file;
                REQUIRE( dsl_dewarper_config_file_get(dewarper_name.c_str(), 
                    &ret_c_config_file) == DSL_RESULT_SUCCESS );
                std::wstring ret_config_file(ret_c_config_file);
                REQUIRE( ret_config_file == new_dewarper_config_file );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Dewarper's camera-id is set" ) 
        {
            uint new_camera_id(1);
            REQUIRE( dsl_dewarper_camera_id_set(dewarper_name.c_str(), 
                new_camera_id) == DSL_RESULT_SUCCESS );

            THEN( "The correct camera-id is returned on get" ) 
            {
                uint ret_camera_id(0);
                REQUIRE( dsl_dewarper_camera_id_get(dewarper_name.c_str(),
                    &ret_camera_id) ==  DSL_RESULT_SUCCESS );
                REQUIRE( ret_camera_id == new_camera_id );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Dewarper's num-batch-buffers is set" ) 
        {
            uint new_batch_buffers(2);
            REQUIRE( dsl_dewarper_num_batch_buffers_set(dewarper_name.c_str(), 
                new_batch_buffers) == DSL_RESULT_SUCCESS );

            THEN( "The correct num-batch-buffers is returned on get" ) 
            {
                uint ret_num_batch_buffers(0);
                REQUIRE( dsl_dewarper_num_batch_buffers_get(dewarper_name.c_str(),
                    &ret_num_batch_buffers) ==  DSL_RESULT_SUCCESS );
                REQUIRE( ret_num_batch_buffers == new_batch_buffers );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Dewarper API checks for NULL input parameters", "[dewarper-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_dewarper_new(NULL, 
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_new(dewarper_name.c_str(), 
                    NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_dewarper_config_file_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_config_file_get(dewarper_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_config_file_set(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_config_file_set(dewarper_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_dewarper_camera_id_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_camera_id_get(dewarper_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_camera_id_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_dewarper_num_batch_buffers_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_num_batch_buffers_get(dewarper_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_dewarper_num_batch_buffers_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}