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
#include "DslElementr.h"

using namespace DSL;

SCENARIO( "An Elementr is constructed correctly", "[Elementr]" )
{
    GIVEN( "A name for a new Elementr in memory" )
    {
        std::string elementName  = "test-element";
        std::string giveElementName = "test-element-queue";

        WHEN( "A child Elmentr is created" )
        {
            DSL_ELEMENT_PTR pElementr = DSL_ELEMENT_NEW("queue", elementName.c_str());
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pElementr->GetName() == giveElementName );
                REQUIRE( pElementr->GetParentState() == GST_STATE_NULL );
            }
        }
    }
}

SCENARIO( "Two Elementrs are linked correctly", "[Elementr]" )
{
    GIVEN( "A Queue Elementr and a Tee Elementr in memory" ) 
    {
        std::string queueElementName  = "test-queue";
        std::string teeElementName = "test-tee";
        
        DSL_ELEMENT_PTR pQueue = DSL_ELEMENT_NEW("queue", queueElementName.c_str());
        DSL_ELEMENT_PTR pTee = DSL_ELEMENT_NEW("tee", teeElementName.c_str());
            
        WHEN( "The Queue is linked downstream to the Tee" )
        {
            REQUIRE( pQueue->LinkToSink(pTee) == true );
            
            THEN( "The Source can unlink from the Sink" )
            {
                REQUIRE( pQueue->IsLinkedToSink() == true );
                REQUIRE( pQueue->IsLinkedToSource() == false );
                REQUIRE( pQueue->GetSink() == pTee );
                REQUIRE( pQueue->GetSource() == nullptr );

                REQUIRE( pTee->IsLinkedToSink() == false );
                REQUIRE( pTee->IsLinkedToSource() == false );
                REQUIRE( pTee->GetSink() == nullptr );
                REQUIRE( pTee->GetSource() == nullptr );

                REQUIRE( pQueue->UnlinkFromSink() == true );
            }
        }
        WHEN( "The Tee is linked upstream to the Queue" )
        {
            REQUIRE( pTee->LinkToSource(pQueue) == true );
            
            THEN( "The Sink can unlink from the Source" )
            {
                REQUIRE( pTee->IsLinkedToSink() == false );
                REQUIRE( pTee->IsLinkedToSource() == true );
                REQUIRE( pTee->GetSink() == nullptr );
                REQUIRE( pTee->GetSource() == pQueue );

                REQUIRE( pQueue->IsLinkedToSink() == false );
                REQUIRE( pQueue->IsLinkedToSource() == false );
                REQUIRE( pQueue->GetSink() == nullptr );
                REQUIRE( pQueue->GetSource() == nullptr );

                REQUIRE( pTee->UnlinkFromSource() == true );
            }
        }
    }
}
