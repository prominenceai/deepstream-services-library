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

SCENARIO( "A new OdeArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeArea" ) 
    {
        std::string odeAreaName("ode-area");
        uint left(123), top(321), width(444), height(555);
        bool display(true);

        WHEN( "A new OdeArea is created" )
        {
            DSL_ODE_AREA_PTR pOdeArea = 
                DSL_ODE_AREA_NEW(odeAreaName.c_str(), left, top, width, height, display);

            THEN( "The OdeArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );

                uint retLeft(0), retTop(0), retWidth(0), retHeight(0);
                bool retDisplay;
                pOdeArea->GetArea(&retLeft, &retTop, &retWidth, &retHeight, &retDisplay);
                REQUIRE( retLeft == left );
                REQUIRE( retTop == top );
                REQUIRE( retWidth == width );
                REQUIRE( retHeight == height );

                double retRed(0), retGreen(0), retBlue(0), retAlpha(0);
                pOdeArea->GetColor(&retRed, &retGreen, &retBlue, &retAlpha);
                REQUIRE( retRed == 1.0 );
                REQUIRE( retGreen == 1.0 );
                REQUIRE( retBlue == 1.0 );
                REQUIRE( retAlpha == 0.2 );
            }
        }
    }
}

SCENARIO( "An OdeArea's rectangle area can be Set/Get", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeArea" ) 
    {
        std::string odeAreaName("ode-area");
        uint left(123), top(321), width(444), height(555);
        bool display(true);

        DSL_ODE_AREA_PTR pOdeArea = 
            DSL_ODE_AREA_NEW(odeAreaName.c_str(), left, top, width, height, display);

        WHEN( "An OdeArea's rectangle area is Set" )
        {
            uint newLeft(111), newTop(222), newWidth(333), newHeight(444);
            bool newDisplay(false);
            pOdeArea->SetArea(newLeft, newTop, newWidth, newHeight, newDisplay);

            THEN( "The correct values are returned on get" )
            {
                uint retLeft(0), retTop(0), retWidth(0), retHeight(0);
                bool retDisplay;
                pOdeArea->GetArea(&retLeft, &retTop, &retWidth, &retHeight, &retDisplay);
                REQUIRE( retLeft == newLeft );
                REQUIRE( retTop == newTop );
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == newHeight );
                REQUIRE( retDisplay == newDisplay );
            }
        }
    }
}

SCENARIO( "An OdeArea's background color can be Set/Get", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeArea" ) 
    {
        std::string odeAreaName("ode-area");
        uint left(123), top(321), width(444), height(555);
        bool display(true);

        DSL_ODE_AREA_PTR pOdeArea = 
            DSL_ODE_AREA_NEW(odeAreaName.c_str(), left, top, width, height, display);

        WHEN( "An OdeArea's rectangle area is Set" )
        {
            double newRed(0.3), newGreen(0.4), newBlue(0.5), newAlpha(0.6);
            bool newDisplay(false);
            pOdeArea->SetColor(newRed, newGreen, newBlue, newAlpha);

            THEN( "The correct values are returned on get" )
            {
                double retRed(0), retGreen(0), retBlue(0), retAlpha(0);
                pOdeArea->GetColor(&retRed, &retGreen, &retBlue, &retAlpha);
                REQUIRE( retRed == newRed );
                REQUIRE( retGreen == newGreen );
                REQUIRE( retBlue == newBlue );
                REQUIRE( retAlpha == newAlpha );
            }
        }
    }
}
