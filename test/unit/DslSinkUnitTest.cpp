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
#include "DslSinkBintr.h"

using namespace DSL;

SCENARIO( "A new OverlaySinkBintr is created correctly",  "[OverlaySinkBintr]" )
{
    GIVEN( "Attributes for a new Overlay Sink" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        WHEN( "The OverlaySinkBintr is created " )
        {
            DSL_OVERLAY_SINK_PTR pSinkBintr = 
                DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetDisplayId() == 0 );
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new OverlaySinkBintr can LinkAll Child Elementrs", "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "A new OverlaySinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The OverlaySinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked OverlaySinkBintr can UnlinkAll Child Elementrs", "[OverlaySinkBintr]" )
{
    GIVEN( "A OsdBintr in a linked state" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A OverlaySinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The OverlaySinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "An OverlaySinkBintr's Display Id can be updated",  "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        uint displayId(123);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
        // ensure display id reflects not is use
        REQUIRE( pSinkBintr->GetDisplayId() == 0 );

        WHEN( "The OverlaySinkBintr's display Id is set " )
        {
            pSinkBintr->SetDisplayId(displayId);
            THEN( "The OverlaySinkBintr's new display Id is returned on Get" )
            {
                REQUIRE( pSinkBintr->GetDisplayId() == displayId );
            }
        }
    }
}

SCENARIO( "An OverlaySinkBintr's Offsets can be updated", "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint initOffsetX(0);
        uint initOffsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), initOffsetX, initOffsetY, sinkW, sinkH);
            
        uint currOffsetX(0);
        uint currOffsetY(0);
    
        pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
        REQUIRE( currOffsetX == initOffsetX );
        REQUIRE( currOffsetY == initOffsetY );

        WHEN( "The OverlaySinkBintr's Offsets are Set" )
        {
            uint newOffsetX(80);
            uint newOffsetY(20);
            
            pSinkBintr->SetOffsets(newOffsetX, newOffsetY);

            THEN( "The OverlaySinkBintr's new demensions are returned on Get")
            {
                pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
                REQUIRE( currOffsetX == newOffsetX );
                REQUIRE( currOffsetY == newOffsetY );
            }
        }
    }
}

SCENARIO( "An OverlaySinkBintr's Dimensions can be updated", "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
            
        uint currSinkW(0);
        uint currSinkH(0);
    
        pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
        REQUIRE( currSinkW == initSinkW );
        REQUIRE( currSinkH == initSinkH );

        WHEN( "The OverlaySinkBintr's dimensions are Set" )
        {
            uint newSinkW(1280);
            uint newSinkH(720);
            
            pSinkBintr->SetDimensions(newSinkW, newSinkH);

            THEN( "The OverlaySinkBintr's new dimensions are returned on Get")
            {
                pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
                REQUIRE( currSinkW == newSinkW );
                REQUIRE( currSinkH == newSinkH );
            }
        }
    }
}

SCENARIO( "A new WindowSinkBintr is created correctly",  "[WindowSinkBintr]" )
{
    GIVEN( "Attributes for a new Window Sink" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        WHEN( "The WindowSinkBintr is created " )
        {
            DSL_WINDOW_SINK_PTR pSinkBintr = 
                DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->IsWindowCapable() == true );
            }
        }
    }
}

SCENARIO( "A new WindowSinkBintr can LinkAll Child Elementrs", "[WindowSinkBintr]" )
{
    GIVEN( "A new WindowSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "A new WindowSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The WindowSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked WindowSinkBintr can UnlinkAll Child Elementrs", "[WindowSinkBintr]" )
{
    GIVEN( "A WindowSinkBintr in a linked state" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A WindowSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The OverlaySinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A WindowSinkBintr's Offsets can be updated", "[WindowSinkBintr]" )
{
    GIVEN( "A new WindowSinkBintr in memory" ) 
    {
        std::string sinkName("window-sink");
        uint initOffsetX(0);
        uint initOffsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), initOffsetX, initOffsetY, sinkW, sinkH);
            
        uint currOffsetX(0);
        uint currOffsetY(0);
    
        pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
        REQUIRE( currOffsetX == initOffsetX );
        REQUIRE( currOffsetY == initOffsetY );

        WHEN( "The WindowSinkBintr's Offsets are Set" )
        {
            uint newOffsetX(80);
            uint newOffsetY(20);
            
            pSinkBintr->SetOffsets(newOffsetX, newOffsetY);

            THEN( "The WindowSinkBintr's new demensions are returned on Get")
            {
                pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
                REQUIRE( currOffsetX == newOffsetX );
                REQUIRE( currOffsetY == newOffsetY );
            }
        }
    }
}

SCENARIO( "An WindowSinkBintr's Dimensions can be updated", "[WindowSinkBintr]" )
{
    GIVEN( "A new WindowSinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
            
        uint currSinkW(0);
        uint currSinkH(0);
    
        pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
        REQUIRE( currSinkW == initSinkW );
        REQUIRE( currSinkH == initSinkH );

        WHEN( "The WindowSinkBintr's dimensions are Set" )
        {
            uint newSinkW(1280);
            uint newSinkH(720);
            
            pSinkBintr->SetDimensions(newSinkW, newSinkH);

            THEN( "The WindowSinkBintr's new dimensions are returned on Get")
            {
                pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
                REQUIRE( currSinkW == newSinkW );
                REQUIRE( currSinkH == newSinkH );
            }
        }
    }
}

