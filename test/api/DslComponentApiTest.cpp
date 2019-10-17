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

#ifndef _DSL_COMPONENT_API_TEST_H
#define _DSL_COMPONENT_API_TEST_H

#include "catch.hpp"
#include "DslApi.h"

SCENARIO( "A single Component is created and deleted correctly", "[component]" )
{
    std::string actualName  = "csi_source";

    GIVEN( "An empty list of components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
    }

    WHEN( "A new component is created" ) 
    {

        REQUIRE( dsl_source_csi_new(actualName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );

        THEN( "The list size and contents are updated correctly" ) 
        {
            REQUIRE( dsl_component_list_size() == 1 );
            REQUIRE( *(dsl_component_list_all()) != NULL );
            
            std::string returnedName = *(dsl_component_list_all());
            REQUIRE( returnedName == actualName );
        }
    }
    WHEN( "The component is deleted")
    {
        REQUIRE( dsl_component_delete(actualName.c_str()) == DSL_RESULT_SUCCESS );
        
        THEN( "The list and contents are updated correctly")
        {
            REQUIRE( dsl_component_list_size() == 0 );
            REQUIRE( *(dsl_component_list_all()) == NULL );
        }
    }
}

#endif // _DSL_COMPONENT_API_TEST_H
