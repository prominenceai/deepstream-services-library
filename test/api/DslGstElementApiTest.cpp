
/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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

SCENARIO( "An invalid factory name causes an exception", "[gst-element-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring element_name(L"element");
        
        std::wstring factory_name(L"non-element");
        
        REQUIRE( dsl_gst_element_list_size() == 0 );

        WHEN( "Several new Elements are created" ) 
        {
            REQUIRE( dsl_gst_element_new(element_name.c_str(),
                factory_name.c_str()) == DSL_RESULT_GST_ELEMENT_THREW_EXCEPTION );

            THEN( "The list size and events are not updated" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
    }
}
                
SCENARIO( "The GST Elements container is updated correctly on multiple new Elements", "[gst-element-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring element_name1(L"element-1");
        std::wstring element_name2(L"element-2");
        std::wstring element_name3(L"element-3");
        
        std::wstring factory_name(L"queue");
        
        REQUIRE( dsl_gst_element_list_size() == 0 );

        WHEN( "Several new Elements are created" ) 
        {
            REQUIRE( dsl_gst_element_new(element_name1.c_str(),
                factory_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_element_new(element_name2.c_str(),
                factory_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_gst_element_new(element_name3.c_str(),
                factory_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 3 );

                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The GST Elements container is updated correctly on Delete GST Element", "[gst-element-api]" )
{
    GIVEN( "A list of several GST Elements" ) 
    {
        std::wstring element_name1(L"element-1");
        std::wstring element_name2(L"element-2");
        std::wstring element_name3(L"element-3");
        
        std::wstring factory_name(L"queue");
        
        REQUIRE( dsl_gst_element_list_size() == 0 );

        REQUIRE( dsl_gst_element_new(element_name1.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        // second creation of the same name must fail
        REQUIRE( dsl_gst_element_new(element_name1.c_str(),
            factory_name.c_str()) == DSL_RESULT_GST_ELEMENT_NAME_NOT_UNIQUE );
            
        REQUIRE( dsl_gst_element_new(element_name2.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_new(element_name3.c_str(),
            factory_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A single Element is deleted" ) 
        {
            REQUIRE( dsl_gst_element_delete(element_name1.c_str()) == 
                DSL_RESULT_SUCCESS );
           
            // Deleting twice must faile
            REQUIRE( dsl_gst_element_delete(element_name1.c_str()) == 
                DSL_RESULT_GST_ELEMENT_NAME_NOT_FOUND );

            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 2 );

                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
        WHEN( "Multiple Elements are deleted" ) 
        {
            const wchar_t* elements[] = {L"element-2", L"element-3", NULL};
            
            REQUIRE( dsl_gst_element_delete_many(elements) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_gst_element_list_size() == 1 );

                REQUIRE( dsl_gst_element_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_gst_element_list_size() == 0 );
            }
        }
    }
}