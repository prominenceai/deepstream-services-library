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
        uint left(0), top(0), width(100), height(100);
        boolean display(true);
        
        REQUIRE( dsl_ode_area_list_size() == 0 );

        WHEN( "Several new Actions are created" ) 
        {
            REQUIRE( dsl_ode_area_new(areaName1.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_area_new(areaName2.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_area_new(areaName3.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_area_list_size() == 3 );

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
        uint left(0), top(0), width(100), height(100);
        boolean display(true);
        
        REQUIRE( dsl_ode_area_list_size() == 0 );

        REQUIRE( dsl_ode_area_new(areaName1.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_area_new(areaName2.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_area_new(areaName3.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );

        WHEN( "A single Area is deleted" ) 
        {
            REQUIRE( dsl_ode_area_delete(areaName1.c_str()) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_area_list_size() == 2 );

                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_area_list_size() == 0 );
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
            }
        }
    }
}

SCENARIO( "The ODE Area's Rectangle can be updated", "[ode-area-api]" )
{
    GIVEN( "An a new ODE Area" ) 
    {
        std::wstring areaName1(L"area-1");
        uint left(0), top(0), width(100), height(100);
        boolean display(true);
        
        REQUIRE( dsl_ode_area_new(areaName1.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );

        WHEN( "The Area's Rectangle is updated" ) 
        {
            uint retLeft(999), retTop(999), retWidth(999), retHeight(999);
            uint newLeft(123), newTop(123), newWidth(432), newHeight(234);
            boolean retDisplay(true), newDisplay(false);

            REQUIRE( dsl_ode_area_set(areaName1.c_str(), 
                newLeft, newTop, newWidth, newHeight, newDisplay)  == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                dsl_ode_area_get(areaName1.c_str(), &retLeft, &retTop, &retWidth, &retHeight, &retDisplay);
                REQUIRE( retLeft == newLeft );
                REQUIRE( retTop == newTop );
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == newHeight );
                REQUIRE( retDisplay == newDisplay );

                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The ODE Area's Background Color can be updated", "[ode-area-api]" )
{
    GIVEN( "An a new ODE Area" ) 
    {
        std::wstring areaName1(L"area-1");
        uint left(0), top(0), width(100), height(100);
        boolean display(true);
        
        REQUIRE( dsl_ode_area_new(areaName1.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );

        WHEN( "The Areas Rectangle is updated" ) 
        {
            double retRed(0.111), retGreen(0.222), retBlue(0.333), retAlpha(0.444);
            double newRed(0.555), newGreen(0.666), newBlue(0.777), newAlpha(0.888);

            REQUIRE( dsl_ode_area_color_set(areaName1.c_str(), newRed, newGreen, newBlue, newAlpha)  == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                dsl_ode_area_color_get(areaName1.c_str(), &retRed, &retGreen, &retBlue, &retAlpha);
                REQUIRE( retRed == newRed );
                REQUIRE( retGreen == newGreen );
                REQUIRE( retBlue == newBlue );
                REQUIRE( retAlpha == newAlpha );
                REQUIRE( retRed == newRed );

                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The ODE Area Color API's fail when invalid Background Color values are used", "[ode-area-api]" )
{
    GIVEN( "An a new ODE Area" ) 
    {
        std::wstring areaName1(L"area-1");
        uint left(0), top(0), width(100), height(100);
        boolean display(true);
        
        REQUIRE( dsl_ode_area_new(areaName1.c_str(), left, top, width, height, display) == DSL_RESULT_SUCCESS );

        WHEN( "An invalid value is used for Red " ) 
        {
            double newRed(1.0001), newGreen(0.999), newBlue(0.999), newAlpha(0.999);

            THEN( "Setting the Area color must fail" ) 
            {
                REQUIRE( dsl_ode_area_color_set(areaName1.c_str(), newRed, newGreen, newBlue, newAlpha)  == DSL_RESULT_ODE_AREA_SET_FAILED );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "An invalid value is used for Green " ) 
        {
            double newRed(0.999), newGreen(1.0001), newBlue(0.999), newAlpha(0.999);

            THEN( "Setting the Area color must fail" ) 
            {
                REQUIRE( dsl_ode_area_color_set(areaName1.c_str(), newRed, newGreen, newBlue, newAlpha)  == DSL_RESULT_ODE_AREA_SET_FAILED );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "An invalid value is used for Blue " ) 
        {
            double newRed(0.999), newGreen(0.999), newBlue(1.0001), newAlpha(0.999);

            THEN( "Setting the Area color must fail" ) 
            {
                REQUIRE( dsl_ode_area_color_set(areaName1.c_str(), newRed, newGreen, newBlue, newAlpha)  == DSL_RESULT_ODE_AREA_SET_FAILED );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "An invalid value is used for Alpha " ) 
        {
            double newRed(0.999), newGreen(0.999), newBlue(0.999), newAlpha(1.0001);

            THEN( "Setting the Area color must fail" ) 
            {
                REQUIRE( dsl_ode_area_color_set(areaName1.c_str(), newRed, newGreen, newBlue, newAlpha)  == DSL_RESULT_ODE_AREA_SET_FAILED );
                REQUIRE( dsl_ode_area_delete_all() == DSL_RESULT_SUCCESS );
            }
        }

    }
}

