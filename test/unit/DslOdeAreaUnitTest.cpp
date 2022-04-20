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

SCENARIO( "A new OdeInclusionArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeInclusionArea" ) 
    {
        std::string odeAreaName("ode-inclusion-area");
        bool show(true);

        std::string polygonName  = "my-polygon";
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);

        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);
 
        WHEN( "A new OdeInclusionArea is created" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea = 
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, show, DSL_BBOX_POINT_ANY);

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
        bool show(true);

        std::string polygonName  = "my-polygon";
        dsl_coordinate coordinates[4] = {{100,100},{210,110},{220, 300},{110,330}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);

        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);
 
        WHEN( "A new OdeExclusionArea is created" )
        {
            DSL_ODE_AREA_EXCLUSION_PTR pOdeArea = 
                DSL_ODE_AREA_EXCLUSION_NEW(odeAreaName.c_str(), pPolygon, show, DSL_BBOX_POINT_ANY);

            THEN( "The OdeExclusionArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeAreaName == retName );

            }
        }
    }
}

SCENARIO( "A new OdeLineArea is created correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);
        uint bboxEdge(DSL_BBOX_EDGE_BOTTOM);

        std::string rgbaLineName  = "rgba-line";
        uint x1(12), x2(34), y1(56), y2(78);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);
       
 
        WHEN( "A new OdeLineArea is created" )
        {
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxEdge);

            THEN( "The OdeLineArea's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeArea->GetCStrName();
                REQUIRE( odeLineName == retName );

            }
        }
    }
}

SCENARIO( "A new OdeLineArea can determine if IsBboxInside correctly", "[OdeArea]" )
{
    GIVEN( "Attributes for a new OdeLineArea" ) 
    {
        std::string odeLineName("ode-line-area");
        bool show(true);

        std::string rgbaLineName  = "rgba-line";
        uint x1(300), y1(200), x2(400), y2(200);
        uint width(4);

        std::string colorName  = "custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        
        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        DSL_RGBA_LINE_PTR pLine = DSL_RGBA_LINE_NEW(rgbaLineName.c_str(), 
            x1, y1, x2, y2, width, pColor);

        NvOSD_RectParams bbox{0};

        WHEN( "A bbox is defined without a crossing DSL_BBOX_EDGE_LEFT " )
        {
            uint bboxEdge(DSL_BBOX_EDGE_LEFT);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxEdge);
                
            bbox.left = 100;
            bbox.top = 100;
            bbox.width = 200;
            bbox.height = 200;

            THEN( "The bbox and Line Area are found to overlap" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == false);
            }
        }
        WHEN( "A bbox is defined with a crossing DSL_BBOX_EDGE_LEFT " )
        {
            uint bboxEdge(DSL_BBOX_EDGE_LEFT);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxEdge);
                
            bbox.left = 320;
            bbox.top = 100;
            bbox.width = 200;
            bbox.height = 200;

            THEN( "The bbox and Line Area are found to overlap" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == true);
            }
        }
        
        WHEN( "A bbox is defined without a crossing DSL_BBOX_EDGE_RIGHT " )
        {
            uint bboxEdge(DSL_BBOX_EDGE_RIGHT);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxEdge);
                
            bbox.left = 99;
            bbox.top = 100;
            bbox.width = 200;
            bbox.height = 200;

            THEN( "The bbox and Line Area are found to overlap" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == false);
            }
        }
        
        WHEN( "A bbox is defined with a crossing DSL_BBOX_EDGE_RIGHT " )
        {
            uint bboxEdge(DSL_BBOX_EDGE_RIGHT);
            
            DSL_ODE_AREA_LINE_PTR pOdeArea = 
                DSL_ODE_AREA_LINE_NEW(odeLineName.c_str(), pLine, show, bboxEdge);
                
            bbox.left = 199;
            bbox.top = 100;
            bbox.width = 200;
            bbox.height = 200;

            THEN( "The bbox and Line Area are found to overlap" )
            {
                REQUIRE(pOdeArea->IsBboxInside(bbox) == true);
            }
        }
    }
}

