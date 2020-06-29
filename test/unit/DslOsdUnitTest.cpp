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
#include "DslOsdBintr.h"

using namespace DSL;

SCENARIO( "A new OsdBintr is created correctly", "[OsdBintr]" )
{
    GIVEN( "Attributes for a new OsdBintr" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(false);

        WHEN( "A new Osd is created" )
        {
            DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

            THEN( "The OsdBintr's memebers are setup and returned correctly" )
            {
                REQUIRE( pOsdBintr->GetGstObject() != NULL );
                pOsdBintr->GetClockEnabled(&enableClock);
                REQUIRE( enableClock == false );
                
                uint left(123), top(123), width(123), height(123);
                
                pOsdBintr->GetCropSettings(&left, &top, &width, &height);
                REQUIRE( left == 0 );
                REQUIRE( top == 0 );
                REQUIRE( width == 0 );
                REQUIRE( height == 0 );
            }
        }
    }
}

SCENARIO( "A new OsdBintr can LinkAll Child Elementrs", "[OsdBintr]" )
{
    GIVEN( "A new OsdBintr in an Unlinked state" ) 
    {
        std::string osdName = "osd";
        bool enableClock(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        WHEN( "A new OsdBintr is Linked" )
        {
            REQUIRE( pOsdBintr->LinkAll() == true );

            THEN( "The OsdBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pOsdBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked OsdBintr can UnlinkAll Child Elementrs", "[OsdBintr]" )
{
    GIVEN( "A OsdBintr in a linked state" ) 
    {
        std::string osdName = "osd";
        bool enableClock(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        REQUIRE( pOsdBintr->LinkAll() == true );

        WHEN( "A OsdBintr is Linked" )
        {
            pOsdBintr->UnlinkAll();

            THEN( "The OsdBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pOsdBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "An OsdBintr's clock can be enabled", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its clock disabled" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        WHEN( "The OsdBintr's clock is enabled" )
        {
            REQUIRE( pOsdBintr->SetClockEnabled(true) == true);
            
            THEN( "The OsdBintr is updated correctly" )
            {
                pOsdBintr->GetClockEnabled(&enableClock);
                REQUIRE( enableClock == true );
            }
        }
    }
}
            
SCENARIO( "An OsdBintr can get and set the clock's X and Y Offsets", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its clock enabled" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(true);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        WHEN( "The clock's offsets are set" )
        {
            uint newOffsetX(500), newOffsetY(300);
            REQUIRE( pOsdBintr->SetClockOffsets(newOffsetX, newOffsetY) == true);
            
            THEN( "The new clock offsets are returned on get" )
            {
                uint retOffsetX(0), retOffsetY(0);
                pOsdBintr->GetClockOffsets(&retOffsetX, &retOffsetY);
                REQUIRE( retOffsetX == newOffsetX );
                REQUIRE( retOffsetY == newOffsetY );
            }
        }
    }
}
            
SCENARIO( "An OsdBintr can get and set the clock's font", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its clock enabled" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(true);
        std::string newName("ariel");

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        WHEN( "The clock font name and size is " )
        {
            REQUIRE( pOsdBintr->SetClockFont(newName.c_str(), 8) == true);
            
            THEN( "The new name and size are returned on get" )
            {
                uint size(0);
                const char* name;
                pOsdBintr->GetClockFont(&name, &size);
                std::string retOsdName(name);
                REQUIRE( retOsdName == newName );
                REQUIRE( size == 8 );
            }
        }
    }
}
            
SCENARIO( "An OsdBintr can get and set the clock's RGB colors", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its clock enabled" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(true);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        WHEN( "The clock's RGB colors are set  " )
        {
            double newRed(0.5), newGreen(0.5), newBlue(0.5), newAlpha(1.0);
            REQUIRE( pOsdBintr->SetClockColor(newRed, newGreen, newBlue, newAlpha) == true);
            
            THEN( "The new name and size are returned on get" )
            {
                double retRed(0.0), retGreen(0.0), retBlue(0.0), retAlpha(0.0);
                pOsdBintr->GetClockColor(&retRed, &retGreen, &retBlue, &retAlpha);
                REQUIRE( retRed == newRed );
                REQUIRE( retGreen == newGreen );
                REQUIRE( retBlue == newBlue );
                REQUIRE( retAlpha == newAlpha );
            }
        }
    }
}

SCENARIO( "An OsdBintr can get and set its crop settings", "[OsdBintr]" )
{
    GIVEN( "A new OsdBintr" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        WHEN( "The OsdBintr's crop settings are updated" )
        {
            uint left(100), top(100), width(320), height(320);
            pOsdBintr->SetCropSettings(left, top, width, height);
            
            THEN( "The correct values are returnd when queried" )
            {
                uint retLeft(0), retTop(0), retWidth(0), retHeight(0);
                
                pOsdBintr->GetCropSettings(&retLeft, &retTop, &retWidth, &retHeight);
                REQUIRE( left == retLeft );
                REQUIRE( top == retTop );
                REQUIRE( width == retWidth );
                REQUIRE( height == retHeight );
            }
        }
    }
}
           
            
SCENARIO( "A OsdBintr can Get and Set its GPU ID",  "[OsdBintr]" )
{
    GIVEN( "A new OsdBintr in memory" ) 
    {
        std::string osdName = "osd";
        boolean enableClock(true);

        uint GPUID0(0);
        uint GPUID1(1);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        REQUIRE( pOsdBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The OsdBintr's  GPU ID is set" )
        {
            REQUIRE( pOsdBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pOsdBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}
            
