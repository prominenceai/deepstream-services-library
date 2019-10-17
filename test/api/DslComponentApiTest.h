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

SCENARIO( "Component uniqueness is managed correctly", "[component]" )
{

    GIVEN( "An empty list of components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_component_list_all() == NULL );
    }
    WHEN( "A new component is created" ) 
    {
        char name[] = "csi_source";

        REQUIRE( dsl_source_csi_new(name, 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );

        THEN( "the list size and contents are updated correctly" ) 
        {
            REQUIRE( dsl_component_list_size() == 1 );
            REQUIRE( dsl_component_list_all() != NULL );
        }
    }
}

#endif // _DSL_COMPONENT_API_TEST_H
