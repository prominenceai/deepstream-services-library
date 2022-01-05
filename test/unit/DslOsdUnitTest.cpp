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
#include "DslOsdBintr.h"

using namespace DSL;

SCENARIO( "A new OsdBintr is created correctly", "[OsdBintr]" )
{
    GIVEN( "Attributes for a new OsdBintr" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        WHEN( "A new Osd is created" )
        {
            DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
                textEnabled, clockEnabled, bboxEnabled, maskEnabled);

            THEN( "The OsdBintr's memebers are setup and returned correctly" )
            {
                REQUIRE( pOsdBintr->GetGstObject() != NULL );
                pOsdBintr->GetClockEnabled(&clockEnabled);
                REQUIRE( clockEnabled == false );
            }
        }
    }
}

SCENARIO( "A new OsdBintr can LinkAll Child Elementrs", "[OsdBintr]" )
{
    GIVEN( "A new OsdBintr in an Unlinked state" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

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
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

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

SCENARIO( "An OsdBintr's display-text property can be enabled", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its display-text property disabled" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

        WHEN( "The OsdBintr's display-text property is enabled" )
        {
            REQUIRE( pOsdBintr->SetTextEnabled(true) == true);
            
            THEN( "The OsdBintr is updated correctly" )
            {
                pOsdBintr->GetTextEnabled(&textEnabled);
                REQUIRE( textEnabled == true );
            }
        }
    }
}
            
SCENARIO( "An OsdBintr's clock can be enabled", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its clock disabled" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

        WHEN( "The OsdBintr's clock is enabled" )
        {
            REQUIRE( pOsdBintr->SetClockEnabled(true) == true);
            
            THEN( "The OsdBintr is updated correctly" )
            {
                pOsdBintr->GetClockEnabled(&clockEnabled);
                REQUIRE( clockEnabled == true );
            }
        }
    }
}
            
SCENARIO( "An OsdBintr can get and set the clock's X and Y Offsets", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its clock enabled" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(true);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

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
        std::string osdName("osd");
        boolean clockEnabled(true);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);
        std::string newName("ariel");

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

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
        std::string osdName("osd");
        boolean clockEnabled(true);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

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
            
SCENARIO( "A OsdBintr can Get and Set its GPU ID",  "[OsdBintr]" )
{
    GIVEN( "A new OsdBintr in memory" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(true);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        uint GPUID0(0);
        uint GPUID1(1);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

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
            
SCENARIO( "An OsdBintr's display-bbox property can be enabled", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its display-bbox property disabled" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

        pOsdBintr->GetBboxEnabled(&bboxEnabled);
        REQUIRE( bboxEnabled == false );

        WHEN( "The OsdBintr's display-text property is enabled" )
        {
            REQUIRE( pOsdBintr->SetBboxEnabled(true) == true);
            
            THEN( "The OsdBintr is updated correctly" )
            {
                pOsdBintr->GetBboxEnabled(&bboxEnabled);
                REQUIRE( bboxEnabled == true );
            }
        }
    }
}
            
SCENARIO( "An OsdBintr's display-mask property can be enabled", "[OsdBintr]" )
{
    GIVEN( "An OsdBintr in memory with its display-mask property disabled" ) 
    {
        std::string osdName("osd");
        boolean clockEnabled(false);
        boolean textEnabled(false);
        boolean bboxEnabled(false);
        boolean maskEnabled(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            textEnabled, clockEnabled, bboxEnabled, maskEnabled);

        pOsdBintr->GetMaskEnabled(&maskEnabled);
        REQUIRE( maskEnabled == false );

        WHEN( "The OsdBintr's display-text property is enabled" )
        {
            REQUIRE( pOsdBintr->SetMaskEnabled(true) == true);
            
            THEN( "The OsdBintr is updated correctly" )
            {
                pOsdBintr->GetMaskEnabled(&maskEnabled);
                REQUIRE( maskEnabled == true );
            }
        }
    }
}
            
