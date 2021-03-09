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

//SCENARIO( "All SMTP Properties can be set and returned back correctly", "[comms-api]" )
//{
//    GIVEN( "A set of SMTP credentials" ) 
//    {
//        std::wstring username(L"joe.blow");
//        std::wstring password(L"3littlepigs");
//        std::wstring server_url(L"mail:\")
//        
//        WHEN( "A Properties are set" ) 
//        {
//            REQUIRE( dsl_smtp_credentials_set(username.c_str(), 
//                password.c_str()) == DSL_RESULT_SUCCESS );
//
//            THEN( "The list size is updated correctly" ) 
//            {
//                
//            }
//        }
//    }
//}    

SCENARIO( "The SMTP API checks for NULL input parameters", "[comms-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring username(L"joe.blow");
        std::wstring password(L"3littlepigs");

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                
                REQUIRE( dsl_smtp_credentials_set(NULL, 
                    password.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_smtp_credentials_set(username.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}
