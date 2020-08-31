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
#include "Dsl.h"
#include "DslApi.h"

SCENARIO( "The ODE Areas container is updated correctly on multiple new ODE Areas", "[ode-area-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring areaName1(L"area-1");
        std::wstring areaName2(L"area-2");
        std::wstring areaName3(L"area-3");
        boolean display(true);

        std::wstring areaRectangleName(L"area-rectangle");
        uint left(0), top(0), width(100), height(100);
        uint border_width(0);
        
        REQUIRE( dsl_ode_area_list_size() == 0 );

        std::wstring lightWhite(L"light-white");
        REQUIRE( dsl_display_type_rgba_color_new(lightWhite.c_str(), 
            1.0, 1.0, 1.0, 0.25) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_rectangle_new(areaRectangleName.c_str(), left, top, width, height, 
            border_width, lightWhite.c_str(), true, lightWhite.c_str())== DSL_RESULT_SUCCESS );

        WHEN( "Several new Actions are created" ) 
        {
            REQUIRE( dsl_ode_area_inclusion_new(areaName1.c_str(), areaRectangleName.c_str(), display) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_area_inclusion_new(areaName2.c_str(), areaRectangleName.c_str(), display) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_area_inclusion_new(areaName3.c_str(), areaRectangleName.c_str(), display) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_area_list_size() == 3 );

                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The ODE Areas container is updated correctly on Delete ODE Area", "[ode-area-api]" )
{
    GIVEN( "A list of several ODE Areas" ) 
    {
        std::wstring areaName1(L"area-1");
        std::wstring areaName2(L"area-2");
        std::wstring areaName3(L"area-3");
        boolean display(true);
        
        std::wstring areaRectangleName(L"area-rectangle");
        uint left(0), top(0), width(100), height(100);
        uint border_width(0);
        
        REQUIRE( dsl_ode_area_list_size() == 0 );

        std::wstring lightWhite(L"light-white");
        REQUIRE( dsl_display_type_rgba_color_new(lightWhite.c_str(), 
            1.0, 1.0, 1.0, 0.25) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_rectangle_new(areaRectangleName.c_str(), left, top, width, height, 
            border_width, lightWhite.c_str(), true, lightWhite.c_str())== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_area_inclusion_new(areaName1.c_str(), areaRectangleName.c_str(), display) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_area_inclusion_new(areaName2.c_str(), areaRectangleName.c_str(), display) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_area_inclusion_new(areaName3.c_str(), areaRectangleName.c_str(), display) == DSL_RESULT_SUCCESS );

        WHEN( "A single Area is deleted" ) 
        {
            REQUIRE( dsl_ode_area_delete(areaName1.c_str()) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_area_list_size() == 2 );

                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "Multiple Areas are deleted" ) 
        {
            const wchar_t* areas[] = {L"area-2", L"area-3", NULL};
            
            REQUIRE( dsl_ode_area_delete_many(areas) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_area_list_size() == 1 );

                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}


SCENARIO( "The ODE Area API checks for NULL input parameters", "[ode-area-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring areaName  = L"test-area";
        std::wstring otherName  = L"other";
        
        uint interval(0);
        boolean enabled(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_ode_area_inclusion_new(NULL, NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_inclusion_new(areaName.c_str(), NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_area_exclusion_new(NULL, NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_exclusion_new(areaName.c_str(), NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_area_delete(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_delete_many(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
