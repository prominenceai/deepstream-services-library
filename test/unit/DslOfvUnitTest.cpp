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
#include "DslOfvBintr.h"

using namespace DSL;

SCENARIO( "A new OfvBintr is created correctly", "[OfvBintr]" )
{
    GIVEN( "Attributes for a new OfvBintr" ) 
    {
        std::string ofvName = "ofv";

        WHEN( "A new Osd is created" )
        {
            DSL_OFV_PTR pOfvBintr = DSL_OFV_NEW(ofvName.c_str());

            THEN( "The OfvBintr's memebers are setup and returned correctly" )
            {
                REQUIRE( pOfvBintr->GetGstObject() != NULL );
            }
        }
    }
}

SCENARIO( "A new OfvBintr can LinkAll Child Elementrs", "[OfvBintr]" )
{
    GIVEN( "A new OfvBintr in an Unlinked state" ) 
    {
        std::string ofvName = "ofv";

        DSL_OFV_PTR pOfvBintr = DSL_OFV_NEW(ofvName.c_str());

        WHEN( "A new OfvBintr is Linked" )
        {
            REQUIRE( pOfvBintr->LinkAll() == true );

            THEN( "The OfvBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pOfvBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked OfvBintr can UnlinkAll Child Elementrs", "[OfvBintr]" )
{
    GIVEN( "A OfvBintr in a linked state" ) 
    {
        std::string ofvName = "ofv";

        DSL_OFV_PTR pOfvBintr = DSL_OFV_NEW(ofvName.c_str());

        REQUIRE( pOfvBintr->LinkAll() == true );

        WHEN( "A OfvBintr is Linked" )
        {
            pOfvBintr->UnlinkAll();

            THEN( "The OfvBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pOfvBintr->IsLinked() == false );
            }
        }
    }
}