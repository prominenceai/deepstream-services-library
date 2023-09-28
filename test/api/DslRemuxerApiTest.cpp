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
        uint ret_width(0);
        uint ret_height(0);
        uint ret_batch_size(0);
        int ret_batch_timeout(0);

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tee_remuxer_batch_properties_get(remuxer_name.c_str(), 
            &ret_batch_size, &ret_batch_timeout) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_batch_size == 0 );
        REQUIRE( ret_batch_timeout == -1 );

        // Check the defaults
        REQUIRE( dsl_tee_remuxer_dimensions_get(remuxer_name.c_str(), 
            &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_width == DSL_STREAMMUX_DEFAULT_WIDTH );
        REQUIRE( ret_height == DSL_STREAMMUX_DEFAULT_HEIGHT );

        WHEN( "A Remuxer's Batch Properties are Set " ) 
        {
            uint new_batch_size(4), new_batch_timeout(40000);
            REQUIRE( dsl_tee_remuxer_batch_properties_set(remuxer_name.c_str(), 
                new_batch_size, new_batch_timeout) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                REQUIRE( dsl_tee_remuxer_batch_properties_get(remuxer_name.c_str(), 
                    &ret_batch_size, &ret_batch_timeout) == DSL_RESULT_SUCCESS);
                REQUIRE( ret_batch_size == new_batch_size );
                REQUIRE( ret_batch_timeout == new_batch_timeout );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Remuxer's Dimensions are Set " ) 
        {
            uint new_width(640), new_height(360);
            REQUIRE( dsl_tee_remuxer_dimensions_set(remuxer_name.c_str(), 
                new_width, new_height) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                REQUIRE( dsl_tee_remuxer_dimensions_get(remuxer_name.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS);
                REQUIRE( ret_width == new_width );
                REQUIRE( ret_height == new_height );
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
        
        uint batch_size(0);
        uint width(0);

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_tee_remuxer_dimensions_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_dimensions_get(remuxer_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_dimensions_get(remuxer_name.c_str(), 
                    &batch_size, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tee_remuxer_dimensions_set(NULL, 
                    1, 1) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}