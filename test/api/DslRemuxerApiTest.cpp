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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 


SCENARIO( "The Components container is updated correctly on new Remuxer",
    "[remuxer-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring remuxer_name(L"remuxer");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Remuxer is created" ) 
        {
            REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on Remuxer delete",
    "[remuxer-api]" )
{
    GIVEN( "A new Remuxer in memory" ) 
    {
        std::wstring remuxer_name(L"remuxer");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Remuxer is deleted" ) 
        {
            REQUIRE( dsl_component_delete(remuxer_name.c_str()) 
                == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Remuxer can add and remove a Branch", "[remuxer-api]" )
{
    GIVEN( "A Remuxer and Branch" ) 
    {
        std::wstring remuxer_name(L"remuxer");
        std::wstring branch_name(L"branch");

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branch_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 2 );

        uint count(0);
        
        REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "A the Branch is added to the Remuxer" ) 
        {
            REQUIRE( dsl_tee_branch_add(remuxer_name.c_str(), 
                branch_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            THEN( "The same branch can be removed" ) 
            {
                REQUIRE( dsl_tee_branch_remove(remuxer_name.c_str(), 
                    branch_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete(remuxer_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Remuxer can add-to and remove a Branch", "[remuxer-api]" )
{
    GIVEN( "A Remuxer and Branch" ) 
    {
        std::wstring remuxer_name(L"remuxer");
        std::wstring branch_name(L"branch");

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branch_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 2 );

        uint count(0);
        uint stream_ids[] = {1,2,3,4};
        
        REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        WHEN( "A the Branch is added to the Remuxer" ) 
        {
            REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
                branch_name.c_str(), stream_ids, 4) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
                &count) == DSL_RESULT_SUCCESS );
            REQUIRE( count == 1 );

            THEN( "The same branch can be removed" ) 
            {
                REQUIRE( dsl_tee_branch_remove(remuxer_name.c_str(), 
                    branch_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete(remuxer_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Remuxer can Set and Get all properties", "[remuxer-api]" )
{
    GIVEN( "A new Remuxer" ) 
    {
        std::wstring remuxer_name(L"remuxer");
        uint ret_batch_size(0);

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tee_remuxer_batch_size_get(remuxer_name.c_str(), 
            &ret_batch_size) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_batch_size == 0 );

        WHEN( "A Remuxer's batch-size is Set " ) 
        {
            uint new_batch_size(4);
            REQUIRE( dsl_tee_remuxer_batch_size_set(remuxer_name.c_str(), 
                new_batch_size) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct value is returned on Get" ) 
            {
                REQUIRE( dsl_tee_remuxer_batch_size_get(remuxer_name.c_str(), 
                    &ret_batch_size) == DSL_RESULT_SUCCESS);
                REQUIRE( ret_batch_size == new_batch_size );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Remuxer can set a Branch config-file correctly", "[remuxer-api]" )
{
    GIVEN( "A Remuxer and Branch" ) 
    {
        std::wstring remuxer_name(L"remuxer");
        std::wstring branch_name(L"branch");

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_branch_new(branch_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 2 );

        uint count(0);
        uint stream_ids[] = {1,2,3,4};
        
        REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 0 );

        // ensure that the get fails prior to adding as branch
        const wchar_t* c_ret_config_file;
        REQUIRE( dsl_tee_remuxer_branch_config_file_get(remuxer_name.c_str(), 
            branch_name.c_str(), &c_ret_config_file) == 
            DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD );

        REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
            branch_name.c_str(), stream_ids, 4) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
            &count) == DSL_RESULT_SUCCESS );
        REQUIRE( count == 1 );

        REQUIRE( dsl_tee_remuxer_branch_config_file_get(remuxer_name.c_str(), 
            branch_name.c_str(), &c_ret_config_file) == DSL_RESULT_SUCCESS );
            
        std::wstring ret_config_file(c_ret_config_file);
        REQUIRE( ret_config_file == L"" );

        WHEN( "A the Remuxer is called to update the Branches config-file" ) 
        {
            std::wstring new_config_file(L"./test/config/all_sources_30fps.txt");
            
            REQUIRE( dsl_tee_remuxer_branch_config_file_set(remuxer_name.c_str(), 
                branch_name.c_str(), new_config_file.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct config-file is returned on get" ) 
            {
                REQUIRE( dsl_tee_remuxer_branch_config_file_get(remuxer_name.c_str(), 
                    branch_name.c_str(), &c_ret_config_file) == 
                    DSL_RESULT_SUCCESS );
                    
                ret_config_file = c_ret_config_file;
                REQUIRE( ret_config_file == new_config_file );
                
                REQUIRE( dsl_tee_branch_remove(remuxer_name.c_str(), 
                    branch_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_tee_branch_count_get(remuxer_name.c_str(), 
                    &count) == DSL_RESULT_SUCCESS );
                REQUIRE( count == 0 );
                REQUIRE( dsl_component_delete(remuxer_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Remuxer API checks for NULL input parameters", "[remuxer-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring remuxer_name(L"remuxer");
        std::wstring branch_name(L"branch");
        
        uint batch_size(0);

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_tee_remuxer_new(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_tee_remuxer_new_branch_add_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_new_branch_add_many(remuxer_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_new_branch_add_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_tee_remuxer_branch_add_to(NULL, 
                    NULL, NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
                    NULL, NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_branch_add_to(remuxer_name.c_str(), 
                    branch_name.c_str(), NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tee_remuxer_batch_size_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_batch_size_get(remuxer_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_batch_size_set(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tee_remuxer_branch_config_file_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_branch_config_file_get(remuxer_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_branch_config_file_get(remuxer_name.c_str(), 
                    branch_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tee_remuxer_branch_config_file_set(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_branch_config_file_set(remuxer_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_branch_config_file_set(remuxer_name.c_str(), 
                    branch_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}