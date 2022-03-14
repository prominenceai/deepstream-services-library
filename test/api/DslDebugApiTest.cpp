/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

SCENARIO( "The Debug API set and get the GST_DEBUG level correctly", "[debug-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring level;
        std::wstring returned_level;
        std::wstring initial_level;
        const wchar_t* c_returned_level;

        // save the initial value to restore
        REQUIRE( dsl_debug_log_level_get(&c_returned_level) == 
            DSL_RESULT_SUCCESS );
            
        initial_level.assign(c_returned_level);
        
        WHEN( "When the debug level is updated" ) 
        {
            level.assign(L"1,DSL:4");
            REQUIRE( dsl_debug_log_level_set(level.c_str()) == 
                DSL_RESULT_SUCCESS );
            
            THEN( "The correct level is returned on get" ) 
            {
                REQUIRE( dsl_debug_log_level_get(&c_returned_level) == 
                    DSL_RESULT_SUCCESS );
                returned_level.assign(c_returned_level);
                
                REQUIRE( returned_level == level );

                REQUIRE( dsl_debug_log_level_set(initial_level.c_str()) == 
                    DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Debug API set and get the GST_DEBUG_FILE variable correctly", "[debug-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring file_path;
        std::wstring initial_file_path;
        std::wstring returned_file_path;
        const wchar_t* c_returned_file_path;

        REQUIRE( dsl_debug_log_file_get(&c_returned_file_path) == 
            DSL_RESULT_SUCCESS );

        initial_file_path.assign(c_returned_file_path);
        
        
        WHEN( "When the debug file_path is updated" ) 
        {
            file_path.assign(L"/tmp/.dsl/my-log.log");
            REQUIRE( dsl_debug_log_file_set(file_path.c_str()) == 
                DSL_RESULT_SUCCESS );
            
            THEN( "The correct file_path is returned on get" ) 
            {
                REQUIRE( dsl_debug_log_file_get(&c_returned_file_path) == 
                    DSL_RESULT_SUCCESS );
                returned_file_path.assign(c_returned_file_path);
                
                REQUIRE( returned_file_path == file_path );

                REQUIRE( dsl_debug_log_file_set(initial_file_path.c_str()) == 
                    DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Debug API checks for NULL input parameters", "[debug-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_debug_log_level_get(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_debug_log_level_set(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_debug_log_file_get(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_debug_log_file_set(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_debug_log_file_set_with_ts(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}
