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
#include "DslTilerBintr.h"

using namespace DSL;

SCENARIO( "A new TilerBintr is created correctly",  "[TilerBintr]" )
{
    GIVEN( "Attribute for a new TilerBintr" ) 
    {
        std::string tilerName("tiled-display");
        uint initWidth(200);
        uint initHeight(100);

        WHEN( "The Tiler's demensions are Set" )
        {
            DSL_TILER_PTR pTilerBintr = 
                DSL_TILER_NEW(tilerName.c_str(), initWidth, initHeight);

            THEN( "The Tiler's new demensions are returned on Get")
            {
                uint currWidth(0);
                uint currHeight(0);

                pTilerBintr->GetDimensions(&currWidth, &currHeight);
                REQUIRE( currWidth == initWidth );
                REQUIRE( currHeight == initHeight );
                REQUIRE( pTilerBintr->GetFrameNumberingEnabled() == false );
            }
        }
    }
}

SCENARIO( "A TilerBintr's dimensions can be updated",  "[TilerBintr]" )
{
    GIVEN( "A new TilerBintr in memory" ) 
    {
        std::string tilerName = "tiled-display";
        uint initWidth(200);
        uint initHeight(100);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), initWidth, initHeight);
            
        uint currWidth(0);
        uint currHeight(0);
    
        pTilerBintr->GetDimensions(&currWidth, &currHeight);
        REQUIRE( currWidth == initWidth );
        REQUIRE( currHeight == initHeight );

        WHEN( "The Tiler's demensions are Set" )
        {
            uint newWidth(1280);
            uint newHeight(720);
            
            pTilerBintr->SetDimensions(newWidth, newHeight);

            THEN( "The Tiler's new demensions are returned on Get")
            {
                pTilerBintr->GetDimensions(&currWidth, &currHeight);
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );
            }
        }
    }
}

SCENARIO( "A Tiled Tiler's tiles can be updated",  "[TilerBintr]" )
{
    GIVEN( "A new Tiled Tiler in memory" ) 
    {
        std::string tilerName = "tiled-display";
        uint width(1280);
        uint height(720);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), width, height);
            
        uint currRows(0);
        uint currColumns(0);
    
        // Tiler element defaults to 1 x 1
        pTilerBintr->GetTiles(&currColumns, &currRows);
        REQUIRE( currColumns == 1 );
        REQUIRE( currRows == 1 );

        WHEN( "The Tiler's tile layout is Set" )
        {
            uint newRows(10);
            uint newColumns(10);
            
            pTilerBintr->SetTiles(newRows, newColumns);

            THEN( "The Tiler's new tile layout is returned on Get")
            {
                pTilerBintr->GetTiles(&currRows, &currColumns);
                REQUIRE( currRows == newRows );
                REQUIRE( currColumns == newColumns );
            }
        }
    }
}


SCENARIO( "A TilerBintr can Get and Set its Show Source setting",  "[TilerBintr]" )
{
    GIVEN( "A new TilerBintr in memory" ) 
    {
        std::string tilerName = "tiled-display";
        uint width(1280);
        uint height(720);
        uint batchSize(4);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), width, height);

        int sourceId(0);
        uint timeout(0);
        pTilerBintr->GetShowSource(&sourceId, &timeout);
        REQUIRE( sourceId == -1 );
        REQUIRE( timeout == 0 );
        REQUIRE( pTilerBintr->SetBatchSize(batchSize) );
        
        WHEN( "The TilerBintr's  Show Source setting is set" )
        {
            uint newTimeout(5);
            REQUIRE( pTilerBintr->SetShowSource(batchSize-1, newTimeout, false) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                pTilerBintr->GetShowSource(&sourceId, &timeout);
                REQUIRE( sourceId == batchSize-1 );
                REQUIRE( timeout == newTimeout );
            }
        }
    }
}

SCENARIO( "A TilerBintr can Get and Set its frame-numbering enabled setting",  "[TilerBintr]" )
{
    GIVEN( "A new TilerBintr in memory" ) 
    {
        std::string tilerName = "tiled-display";
        uint width(1280);
        uint height(720);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), width, height);

        // must disabled by default
        REQUIRE( pTilerBintr->GetFrameNumberingEnabled() == false );

        // disabling when disabled must fail
        REQUIRE( pTilerBintr->SetFrameNumberingEnabled(false) == false );
        
        WHEN( "The TilerBintr's frame-number enabled is set" )
        {
            REQUIRE( pTilerBintr->SetFrameNumberingEnabled(true) == true );

            THEN( "The correct value is returned on get" )
            {
                REQUIRE( pTilerBintr->GetFrameNumberingEnabled() == true );

                // enabling when enabled must fail
                REQUIRE( pTilerBintr->SetFrameNumberingEnabled(true) == false );
            }
        }
    }
}

SCENARIO( "A TilerBintr can Get and Set its GPU ID",  "[TilerBintr]" )
{
    GIVEN( "A new TilerBintr in memory" ) 
    {
        std::string tilerName = "tiled-display";
        uint width(1280);
        uint height(720);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), width, height);

        uint GPUID0(0);
        uint GPUID1(1);

        REQUIRE( pTilerBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The TilerBintr's  GPU ID is set" )
        {
            REQUIRE( pTilerBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pTilerBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}
