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

        std::wstring polygonName(L"polygon");
        uint border_width(3);
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint num_coordinates(4);
        
        REQUIRE( dsl_ode_area_list_size() == 0 );

        std::wstring colorName(L"light-white");
        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            1.0, 1.0, 1.0, 0.25) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
            border_width, colorName.c_str())== DSL_RESULT_SUCCESS );

        WHEN( "Several new Actions are created" ) 
        {
            REQUIRE( dsl_ode_area_inclusion_new(areaName1.c_str(), 
                polygonName.c_str(), display, DSL_BBOX_POINT_ANY) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_area_inclusion_new(areaName2.c_str(), 
                polygonName.c_str(), display, DSL_BBOX_POINT_ANY) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_area_inclusion_new(areaName3.c_str(), 
                polygonName.c_str(), display, DSL_BBOX_POINT_ANY) == DSL_RESULT_SUCCESS );
            
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
        
        std::wstring polygonName(L"polygon");
        uint border_width(3);
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint num_coordinates(4);
        
        REQUIRE( dsl_ode_area_list_size() == 0 );

        std::wstring colorName(L"light-white");
        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            1.0, 1.0, 1.0, 0.25) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_polygon_new(polygonName.c_str(), coordinates, num_coordinates, 
            border_width, colorName.c_str())== DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_area_inclusion_new(areaName1.c_str(), 
            polygonName.c_str(), display, DSL_BBOX_POINT_ANY) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_area_inclusion_new(areaName2.c_str(), 
            polygonName.c_str(), display, DSL_BBOX_POINT_ANY) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_area_inclusion_new(areaName3.c_str(), 
            polygonName.c_str(), display, DSL_BBOX_POINT_ANY) == DSL_RESULT_SUCCESS );

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

SCENARIO( "The ODE Line can be created and deleted", "[ode-area-api]" )
{
    GIVEN( "An RGBA Line" ) 
    {
        std::wstring areaName  = L"line-area";
        
        uint interval(0);
        boolean enabled(0);
        
        std::wstring lineName(L"line");
        uint x1(100), y1(100), x2(200), y2(200);
        uint width(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
            x1, y1, x2, y2, width, colorName.c_str())== DSL_RESULT_SUCCESS );

        WHEN( "When a new ODE Line Area is created" ) 
        {
            REQUIRE( dsl_ode_area_line_new(areaName.c_str(), lineName.c_str(), 
                true, DSL_BBOX_EDGE_BOTTOM) == DSL_RESULT_SUCCESS );

            THEN( "The Area can then be deleted" ) 
            {
                REQUIRE( dsl_ode_area_delete(areaName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The ODE Line API checks for an invalid TestPoint parameter", "[ode-area-api]" )
{
    GIVEN( "An RGBA Line" ) 
    {
        std::wstring areaName  = L"line-area";
        
        uint interval(0);
        boolean enabled(0);
        
        std::wstring lineName(L"line");
        uint x1(100), y1(100), x2(200), y2(200);
        uint width(20);

        std::wstring colorName(L"my-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        REQUIRE( dsl_display_type_rgba_color_custom_new(colorName.c_str(), 
            red, green, blue, alpha) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_display_type_rgba_line_new(lineName.c_str(), 
            x1, y1, x2, y2, width, colorName.c_str())== DSL_RESULT_SUCCESS );

        WHEN( "When an invalid TestEdge parameter is used" ) 
        {
            uint invalid_test_point(DSL_BBOX_POINT_ANY+1);
            
            THEN( "The Area fails to create" ) 
            {
                REQUIRE( dsl_ode_area_line_new(areaName.c_str(), lineName.c_str(), 
                    true, invalid_test_point) == DSL_RESULT_ODE_AREA_PARAMETER_INVALID );
                    
                REQUIRE( dsl_display_type_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
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
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_ode_area_inclusion_new(NULL, NULL, 
                    false, DSL_BBOX_POINT_ANY) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_inclusion_new(areaName.c_str(), NULL, 
                    false, DSL_BBOX_POINT_ANY) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_area_exclusion_new(NULL, NULL, 
                    false, DSL_BBOX_POINT_ANY) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_exclusion_new(areaName.c_str(), NULL, 
                    false, DSL_BBOX_POINT_ANY) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_area_line_new(NULL, NULL, 
                    false, DSL_BBOX_EDGE_BOTTOM) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_line_new(areaName.c_str(), NULL, 
                    false, DSL_BBOX_EDGE_BOTTOM) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_area_delete(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_area_delete_many(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
