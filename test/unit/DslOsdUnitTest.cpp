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
        bool enableClock(false);

        WHEN( "A new Osd is created" )
        {
            DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

            THEN( "The PrimaryGieBintr's memebers are setup and returned correctly" )
            {
                REQUIRE( pOsdBintr->GetGstObject() != NULL );
                REQUIRE( pOsdBintr->IsClockEnabled() == enableClock );
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
        bool enableClock(false);

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), enableClock);

        REQUIRE( pOsdBintr->LinkAll() == true );

        WHEN( "The OsdBintr's clock is enabled" )
        {
            pOsdBintr->EnableClock();
            
            THEN( "The OsdBintr is updated correctly" )
            {
                REQUIRE( pOsdBintr->IsClockEnabled() == true );
            }
        }
    }
}
            
            
