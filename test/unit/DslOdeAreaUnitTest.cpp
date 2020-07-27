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
#include "DslOdeArea.h"

using namespace DSL;

SCENARIO( "A new OdeInclusionArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclusionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool display(true);

        std::string rectangleName  = "my-rectangle";
        uint left(12), top(34), width(56), height(78);
        uint borderWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW(rectangleName.c_str(), 
            left, top, width, height, borderWidth, pColor, true, pColor);
       
 
        WHEN( "A new OdeInclusionArea is created" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea = 
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pRectangle, display);

            THEN( "The OdeInclusionArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );

            }
        }
    }
}

SCENARIO( "A new OdeExclusionArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeExclusionArea" ) 
    {
        std::string odeAreaName("ode-exclusion-area");
        bool display(true);

        std::string rectangleName  = "my-rectangle";
        uint left(12), top(34), width(56), height(78);
        uint borderWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW(rectangleName.c_str(), 
            left, top, width, height, borderWidth, pColor, true, pColor);
       
 
        WHEN( "A new OdeExclusionArea is created" )
        {
            DSL_ODE_AREA_EXCLUSION_PTR pOdeArea = 
                DSL_ODE_AREA_EXCLUSION_NEW(odeAreaName.c_str(), pRectangle, display);

            THEN( "The OdeExclusionArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );

            }
        }
    }
}

