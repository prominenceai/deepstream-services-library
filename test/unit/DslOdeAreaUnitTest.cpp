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
#include "DslOdeArea.h"

using namespace DSL;

SCENARIO( "A new OdeLineArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);
        uint bboxTestPoint(DSL_BBOX_POINT_ANY);

        std::string rgbaLineName  = "rgba-line";
        uint x1(12), x2(34), y1(56), y2(78);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);
       
 
        WHEN( "A new OdeLineArea is created" )
        {
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);

            THEN( "The OdeLineArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeLineName == retName );
                REQUIRE( pOdeArea->GetBboxTestPoint() == DSL_BBOX_POINT_ANY );
            }
        }
    }
}

SCENARIO( "A new horizontal OdeLineArea can determine if IsBboxInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";

        // horizontal line coordinates
        uint x1(100), y1(200), x2(400), y2(200);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        NvOSD_RectParams bbox{0};
        // set to overlap the line
        bbox.left = 200;
        bbox.top = 150;
        bbox.width = 200;
        bbox.height = 200;

        WHEN( "A bbox and test-point are defined to be on the Inside of a Line" )
        {
            uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);
                

            THEN( "The bbox is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == true );
            }
        }
        WHEN( "A bbox and test-point are defined to be on the Outside of a Line" )
        {
            uint bboxTestPoint(DSL_BBOX_POINT_NORTH);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);
                

            THEN( "The bbox is found outside the Line Area" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == false );
            }
        }
    }
}

SCENARIO( "A new vertical OdeLineArea can determine IsBboxInside correctly", 
    "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";
        
        // vertical line coordinates
        uint x1(200), y1(100), x2(200), y2(400);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        NvOSD_RectParams bbox{0};
        // set to overlap the line
        bbox.left = 150;
        bbox.top = 200;
        bbox.width = 200;
        bbox.height = 200;

        WHEN( "A bbox and test-point are defined to be on the Inside of a Line" )
        {
            uint bboxTestPoint(DSL_BBOX_POINT_WEST);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);
                

            THEN( "The bbox is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == true );
            }
        }
        WHEN( "A bbox and test-point are defined to be on the Outside of a Line" )
        {
            uint bboxTestPoint(DSL_BBOX_POINT_EAST);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);
                

            THEN( "The bbox is found outside the Line Area" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == false );
            }
        }
    }
}

SCENARIO( "A new horizontal OdeLineArea can determine IsPointInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";

        // horizontal line coordinates
        uint x1(100), y1(200), x2(400), y2(200);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_LINE_PTR pOdeArea = 
            DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the Inside of a Line" )
        {
            dsl_coordinate coordinate = {200, 100};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == false );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Line" )
        {
            dsl_coordinate coordinate = {200, 300};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == true );
            }
        }
    }
}

SCENARIO( "A new vertical OdeLineArea can determine if IsPointInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";

        // vertical line coordinates
        uint x1(200), y1(100), x2(200), y2(400);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_LINE_PTR pOdeArea = 
            DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the Inside of a Line" )
        {
            dsl_coordinate coordinate = {100, 200};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == true );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Line" )
        {
            dsl_coordinate coordinate = {300, 100};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == false );
            }
        }
    }
}

SCENARIO( "A new Horizonta OdeLineArea can GetPointLocation correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";

        // horizontal line coordinates
        uint x1(100), y1(200), x2(400), y2(200);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_LINE_PTR pOdeArea = 
            DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the outside of a Line" )
        {
            dsl_coordinate coordinate = {200, 100};

            THEN( "The point is found outside the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_OUTSIDE );
            }
        }
        WHEN( "A point is defined to be on the inside of a Line" )
        {
            dsl_coordinate coordinate = {200, 300};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_INSIDE );
            }
        }
        WHEN( "A point is defined to be on the Line" )
        {
            dsl_coordinate coordinate = {300, 200};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_ON_LINE );
            }
        }
    }
}

SCENARIO( "A new Vertical OdeLineArea can GetPointLocation correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";

        // vertical line coordinates
        uint x1(100), y1(200), x2(400), y2(200);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_LINE_PTR pOdeArea = 
            DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the Inside of a Line" )
        {
            dsl_coordinate coordinate = {200, 100};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_OUTSIDE );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Line" )
        {
            dsl_coordinate coordinate = {200, 300};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == 
                    DSL_AREA_POINT_LOCATION_INSIDE );
            }
        }
    }
}

SCENARIO( "A new OdeLineArea can determine IsPointOnLine correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";

        // horizontal line coordinates
        uint x1(100), y1(200), x2(400), y2(200);
        uint width(10);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_LINE_PTR pOdeArea = 
            DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the Inside of a Line" )
        {
            dsl_coordinate coordinate = {200, 194};

            THEN( "The point is found not on the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointOnLine(coordinate) == false );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Line" )
        {
            dsl_coordinate coordinate = {200, 195};

            THEN( "The point is found on the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointOnLine(coordinate) == true );
            }
        }
    }
}

SCENARIO( "A new MultiLineArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeMultiLineArea" ) 
    {
        std::string odeAreaName("ode-multi-line-area");
        bool show(true);

        std::string multiLineName  = "my-multi-line";
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);

        DSL_RGBA_MULTI_LINE_PTR pMultiLine = DSL_RGBA_MULTI_LINE_NEW(
            multiLineName.c_str(), coordinates, numCoordinates, lineWidth, pColor);
 
        WHEN( "A new OdeMultiLineArea is created" )
        {
            DSL_ODE_AREA_MULTI_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_MULTI_LINE_NEW(odeAreaName.c_str(), 
                    pMultiLine, show, DSL_BBOX_POINT_ANY);

            THEN( "The OdeMultiLineArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );
                REQUIRE( pOdeArea->GetBboxTestPoint() == DSL_BBOX_POINT_ANY );
            }
        }
    }
}


SCENARIO( "A new OdeMulLineArea can determine if IsPointInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeMultiLineArea" ) 
    {
        std::string odeMultiLineName("ode-line-area");
        bool show(true);

        std::string rgbaMultiLineName  = "rgba-multi-line";

        // multi-line coordinates
        uint numCoordinates(4);
        dsl_coordinate coordinates[4] = {{100,100},{200,90},{300,130},{400,150}};
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_MULTI_LINE_PTR pMultiLine = DSL_RGBA_MULTI_LINE_NEW(rgbaMultiLineName.c_str(), 
            coordinates, numCoordinates, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_MULTI_LINE_PTR pOdeArea = 
            DSL_ODE_AREA_MULTI_LINE_NEW(odeMultiLineName.c_str(), pMultiLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the Inside of a Line" )
        {
            dsl_coordinate coordinate = {250, 80};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == false );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Line" )
        {
            dsl_coordinate coordinate = {200, 140};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == true );
            }
        }
    }
}

SCENARIO( "A new Horizonta OdeMultLineArea can GetPointLocation correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeMultiLineName("ode-line-area");
        bool show(true);

        std::string rgbaMultiLineName  = "rgba-multi-line";

        // multi-line coordinates
        uint numCoordinates(4);
        dsl_coordinate coordinates[4] = {{100,100},{200,90},{300,130},{400,150}};
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_MULTI_LINE_PTR pMultiLine = DSL_RGBA_MULTI_LINE_NEW(
            rgbaMultiLineName.c_str(), coordinates, numCoordinates, width, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
        
        DSL_ODE_AREA_MULTI_LINE_PTR pOdeArea = DSL_ODE_AREA_MULTI_LINE_NEW(
            odeMultiLineName.c_str(), pMultiLine, show, bboxTestPoint);

        WHEN( "A point is defined to be on the outside of a Line" )
        {
            dsl_coordinate coordinate = {200, 50};

            THEN( "The point is found outside the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_OUTSIDE );
            }
        }
        WHEN( "A point is defined to be on the inside of a Line" )
        {
            dsl_coordinate coordinate = {200, 300};

            THEN( "The point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_INSIDE );
            }
        }
        WHEN( "A point is defined to be on the the Line" )
        {
            dsl_coordinate coordinate = {200, 90};

            THEN( "The point is found on the Line Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_ON_LINE );
            }
        }
    }
}

SCENARIO( "A new OdeInclusionArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclusionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        uint numCoordinates(4);
        dsl_coordinate coordinates[] = {{100,100},{210,110},{220, 300},{110,330}};
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);

        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);
 
        WHEN( "A new OdeInclusionArea is created" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea = DSL_ODE_AREA_INCLUSION_NEW(
                odeAreaName.c_str(), pPolygon, show, DSL_BBOX_POINT_ANY);

            THEN( "The OdeInclusionArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );
                REQUIRE( pOdeArea->GetBboxTestPoint() == DSL_BBOX_POINT_ANY );
            }
        }
    }
}

SCENARIO( "A new OdeInclussionArea can determine if IsBboxInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclussionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        uint numCoordinates(4);
        dsl_coordinate coordinates[] = {{100,100},{210,110},{220, 300},{110,330}};
        uint lineWidth(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);

        NvOSD_RectParams bbox{0};
        // set to overlap the line
        bbox.left = 100;
        bbox.top = 100;
        bbox.width = 200;
        bbox.height = 200;

        WHEN( "A bounding box and test-point are defined to be on the Inside of a Polygon" )
        {
            uint bboxTestPoint(DSL_BBOX_POINT_SOUTH);
            
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea = DSL_ODE_AREA_INCLUSION_NEW(
                odeAreaName.c_str(), pPolygon, show, bboxTestPoint);

            THEN( "The test point is found inside the Line Area" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == true );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Line" )
        {
            uint bboxTestPoint(DSL_BBOX_POINT_NORTH);
            
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea = DSL_ODE_AREA_INCLUSION_NEW(
                odeAreaName.c_str(), pPolygon, show, bboxTestPoint);

            THEN( "The test point is found outside the Line Area" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == false );
            }
        }
    }
}

SCENARIO( "A new OdeInclussionArea can determine if IsPointInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclussionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        uint numCoordinates(4);
        dsl_coordinate coordinates[] = {{100,100},{210,110},{220, 300},{110,330}};
        uint lineWidth(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_NORTH);
        
        DSL_ODE_AREA_INCLUSION_PTR pOdeArea = DSL_ODE_AREA_INCLUSION_NEW(
            odeAreaName.c_str(), pPolygon, show, bboxTestPoint);

        WHEN( "A point is defined to be on the Inside of a Polygon" )
        {
            dsl_coordinate coordinate = {150, 140};

            THEN( "The point is found inside the Polygon Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == true );
            }
        }
        WHEN( "A point is defined to be on the Outside of a Polygon" )
        {
            dsl_coordinate coordinate = {20, 20};

            THEN( "The point is found inside the Polygon Area" )
            {
                REQUIRE(pOdeArea->IsPointInside(coordinate) == false );
            }
        }
    }
}

SCENARIO( "A new OdeInclussionArea can GetPointLocation correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclussionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        uint numCoordinates(4);
        dsl_coordinate coordinates[] = {{100,100},{210,110},{220, 300},{110,330}};
        uint lineWidth(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_NORTH);
        
        DSL_ODE_AREA_INCLUSION_PTR pOdeArea = DSL_ODE_AREA_INCLUSION_NEW(
            odeAreaName.c_str(), pPolygon, show, bboxTestPoint);

        WHEN( "A point is defined to be on the outside of a Polygon" )
        {
            dsl_coordinate coordinate = {50, 50};

            THEN( "The point is found outside the Polygon Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_OUTSIDE );
            }
        }
        WHEN( "A point is defined to be on the inside of a Polygon" )
        {
            dsl_coordinate coordinate = {150, 140};

            THEN( "The point is found inside the Polygon Area" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_INSIDE );
            }
        }
        WHEN( "A point is defined to be on the Polygon" )
        {
            dsl_coordinate coordinate = {220, 300};

            THEN( "The point is found on the Polygon Area's line" )
            {
                REQUIRE(pOdeArea->GetPointLocation(coordinate) == 
                    DSL_AREA_POINT_LOCATION_ON_LINE );
            }
        }
    }
}

SCENARIO( "A new OdeInclussionArea can determine IsPointOnLine correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclussionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        uint numCoordinates(4);
        dsl_coordinate coordinates[] = {{100,100},{210,110},{220, 300},{110,330}};
        uint lineWidth(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);

        uint bboxTestPoint(DSL_BBOX_POINT_NORTH);
        
        DSL_ODE_AREA_INCLUSION_PTR pOdeArea = DSL_ODE_AREA_INCLUSION_NEW(
            odeAreaName.c_str(), pPolygon, show, bboxTestPoint);

        WHEN( "A point is defined to be on the outside of a Polygon" )
        {
            dsl_coordinate coordinate = {50, 50};

            THEN( "The point is found outside the Polygon Area" )
            {
                REQUIRE(pOdeArea->IsPointOnLine(coordinate) == false );
            }
        }
        WHEN( "A point is defined to be on the inside of a Polygon" )
        {
            dsl_coordinate coordinate = {150, 140};

            THEN( "The point is found inside the Polygon Area" )
            {
                REQUIRE(pOdeArea->IsPointOnLine(coordinate) == false );
            }
        }
        WHEN( "A point is defined to be on the Polygon" )
        {
            dsl_coordinate coordinate = {220, 300};

            THEN( "The point is found on the Polygon Area's line" )
            {
                REQUIRE(pOdeArea->IsPointOnLine(coordinate) == true );
            }
        }
    }
}

SCENARIO( "A new OdeExclusionArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeExclusionArea" ) 
    {
        std::string odeAreaName("ode-exclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), 
            red, green, blue, alpha);

        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);
 
        WHEN( "A new OdeExclusionArea is created" )
        {
            DSL_ODE_AREA_EXCLUSION_PTR pOdeArea = DSL_ODE_AREA_EXCLUSION_NEW(
                odeAreaName.c_str(), pPolygon, show, DSL_BBOX_POINT_ANY);

            THEN( "The OdeExclusionArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );
                REQUIRE( pOdeArea->GetBboxTestPoint() == DSL_BBOX_POINT_ANY );
            }
        }
    }
}

