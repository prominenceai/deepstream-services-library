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
#include "DslGeosTypes.h"

using namespace DSL;

SCENARIO( "A new GEOS Line is created correctly", "[GeosTypes]" )
{
    GIVEN( "A new NvOSD Line with coordinates and dimensions" ) 
    {
        NvOSD_LineParams testLine{0};
        testLine.x1 = 100;
        testLine.y1 = 100;
        testLine.x2 = 200;
        testLine.y2 = 200;
 
        WHEN( "A new GEOS Line is created" )
        {
            GeosLine testGeosLine(testLine);

            THEN( "The GEOS Line's memebers are setup correctly" )
            {
                REQUIRE( testGeosLine.m_pGeosLine != NULL );
            }
        }
    }
}

SCENARIO( "Two new GEOS Lines are determined to cross", "[GeosTypes]" )
{
    GIVEN( "Two new NvOSD Lines that cross" ) 
    {
        NvOSD_LineParams testLine1{0};
        testLine1.x1 = 100;
        testLine1.y1 = 100;
        testLine1.x2 = 200;
        testLine1.y2 = 200;
 
        NvOSD_LineParams testLine2{0};
        testLine2.x1 = 200;
        testLine2.y1 = 100;
        testLine2.x2 = 100;
        testLine2.y2 = 200;
 
        WHEN( "The GEOS Lines are created" )
        {
            GeosLine testGeosLine1(testLine1);
            GeosLine testGeosLine2(testLine2);

            THEN( "The lines are determined to cross one another" )
            {
                REQUIRE( testGeosLine1.Intersects(testGeosLine2) == true );
                REQUIRE( testGeosLine2.Intersects(testGeosLine1) == true );
            }
        }
    }
}

SCENARIO( "Two new GEOS Lines are determined to NOT cross", "[GeosTypes]" )
{
    GIVEN( "Two new NvOSD Lines that do not cross" ) 
    {
        NvOSD_LineParams testLine1{0};
        testLine1.x1 = 100;
        testLine1.y1 = 100;
        testLine1.x2 = 200;
        testLine1.y2 = 100;
 
        NvOSD_LineParams testLine2{0};
        testLine2.x1 = 100;
        testLine2.y1 = 200;
        testLine2.x2 = 200;
        testLine2.y2 = 200;
 
        WHEN( "The GEOS Lines are created" )
        {
            GeosLine testGeosLine1(testLine1);
            GeosLine testGeosLine2(testLine2);

            THEN( "The lines are determined to NOT cross one another" )
            {
                REQUIRE( testGeosLine1.Intersects(testGeosLine2) != true );
                REQUIRE( testGeosLine2.Intersects(testGeosLine1) != true );
            }
        }
    }
}

SCENARIO( "A new GEOS Polygon is created from a Polygon Display Type correctly", "[GeosTypes]" )
{
    GIVEN( "A new Polygon Display Type" ) 
    {
        std::string polygonName  = "my-polygon";
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);
        
        WHEN( "A new GEOS Line is created" )
        {
            GeosPolygon testGeosPolygon(*pPolygon);

            THEN( "The GEOS Line's memebers are setup correctly" )
            {
                REQUIRE( testGeosPolygon.m_pGeosPolygon != NULL );
            }
        }
    }
}

SCENARIO( "A GEOS Polygon can determine if a point is within correctly", "[GeosTypes]" )
{
    GIVEN( "A new Polygon Display Type" ) 
    {
        std::string polygonName  = "my-polygon";
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        
        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);
        
        GeosPolygon testGeosPolygon(*pPolygon);
 
        WHEN( "A point ouside of the Polygon is used" )
        {
            GeosPoint testGeosPoint(99,99);
            
            THEN( "The GEOS Polygon's Contains function must return false" )
            {
                REQUIRE( testGeosPolygon.Contains(testGeosPoint) == false );
            }
        }
        WHEN( "A point within the Polygon is checked" )
        {
            GeosPoint testGeosPoint(150,250);
            
            THEN( "The GEOS Polygon's Contains function must return true" )
            {
                REQUIRE( testGeosPolygon.Contains(testGeosPoint) == true );
            }
        }
    }
}
