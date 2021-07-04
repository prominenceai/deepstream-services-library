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
#include "DslSegVisualBintr.h"

using namespace DSL;


SCENARIO( "A SegVisualBintr is created correctly",  "[SegVisualBintr]" )
{
    GIVEN( "Attributes for a new SegVisualBintr" ) 
    {
        std::string segVisualName("seg-visual");
        uint width(1920);
        uint height(1080);

        WHEN( "The SegVisualBintr is created" )
        {
            DSL_SEGVISUAL_PTR pSegVisualBintr = 
                DSL_SEGVISUAL_NEW(segVisualName.c_str(), width, height);

            THEN( "The SegVisualBintr's properties are intialized correctly")
            {
                uint retWidth(99), retHeight(99);
                pSegVisualBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == width );
                REQUIRE( retHeight == height );
                REQUIRE( pSegVisualBintr->GetBatchSize() == 0 );
                REQUIRE( pSegVisualBintr->GetGpuId() == 0 );
                REQUIRE( pSegVisualBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A SegVisualBintr updates its properties correctly",  "[SegVisualBintr]" )
{
    GIVEN( "A new SegVisualBintr" ) 
    {
        std::string segVisualName("seg-visual");
        uint width(1920);
        uint height(1080);

        DSL_SEGVISUAL_PTR pSegVisualBintr = 
            DSL_SEGVISUAL_NEW(segVisualName.c_str(), width, height);

        WHEN( "The SegVisualBintr's properties are set" )
        {
            uint newWidth(456), newHeight(345);
            REQUIRE( pSegVisualBintr->SetDimensions(newWidth, newHeight) == true);
            
            uint newBatchSize(4);
            REQUIRE( pSegVisualBintr->SetBatchSize(newBatchSize) == true );

            uint newGpuId(1);
            REQUIRE( pSegVisualBintr->SetGpuId(newGpuId) == true);

            THEN( "The SegVisualBintr's properties are intialized correctly")
            {
                uint retWidth(99), retHeight(99);
                pSegVisualBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == newHeight );
                REQUIRE( pSegVisualBintr->GetBatchSize() == newBatchSize );
                REQUIRE( pSegVisualBintr->GetGpuId() == newGpuId );
            }
        }
    }
}

SCENARIO( "A new SegVisualBintr can LinkAll Child Elementrs", "[SegVisualBintr]" )
{
    GIVEN( "A new SegVisualBintr in an Unlinked state" ) 
    {
        std::string segVisualName("seg-visual");
        uint width(1920);
        uint height(1080);

        DSL_SEGVISUAL_PTR pSegVisualBintr = 
            DSL_SEGVISUAL_NEW(segVisualName.c_str(), width, height);
            
        REQUIRE( pSegVisualBintr->SetBatchSize(1) == true );

        WHEN( "A new SegVisualBintr is Linked" )
        {
            REQUIRE( pSegVisualBintr->LinkAll() == true );

            THEN( "The SegVisualBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSegVisualBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A new SegVisualBintr with a default batch-size fails to LinkAll", "[SegVisualBintr]" )
{
    GIVEN( "A new SegVisualBintr in an Unlinked state" ) 
    {
        std::string segVisualName("seg-visual");
        uint width(1920);
        uint height(1080);

        DSL_SEGVISUAL_PTR pSegVisualBintr = 
            DSL_SEGVISUAL_NEW(segVisualName.c_str(), width, height);

        WHEN( "A new SegVisualBintr is Linked" )
        {
            REQUIRE( pSegVisualBintr->LinkAll() == false );

            THEN( "The SegVisualBintr IsLinked state is not updated" )
            {
                REQUIRE( pSegVisualBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Linked SegVisualBintr can UnlinkAll Child Elementrs", "[SegVisualBintr]" )
{
    GIVEN( "A SegVisualBintr in a linked state" ) 
    {
        std::string segVisualName("seg-visual");
        uint width(1920);
        uint height(1080);

        DSL_SEGVISUAL_PTR pSegVisualBintr = 
            DSL_SEGVISUAL_NEW(segVisualName.c_str(), width, height);

        REQUIRE( pSegVisualBintr->SetBatchSize(1) == true );

        REQUIRE( pSegVisualBintr->LinkAll() == true );

        WHEN( "A SegVisualBintr is Linked" )
        {
            pSegVisualBintr->UnlinkAll();

            THEN( "The SegVisualBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSegVisualBintr->IsLinked() == false );
            }
        }
    }
}
