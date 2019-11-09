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
#include "DslElementr.h"
#include "DslTestBintr.h"

using namespace DSL;

SCENARIO( "An Elementr is constructed correctly", "[Elementr]" )
{
    GIVEN( "A name for a new Elementr in memory" )
    {
        std::string elementName  = "test-element";

        WHEN( "A child Elmentr is created" )
        {
            DSL_ELEMENT_PTR pElementr = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, elementName.c_str());
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pElementr->m_name == elementName );
                REQUIRE( pElementr->m_pGstObj != NULL );
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
        
        DSL_ELEMENT_PTR pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, queueElementName.c_str());
        DSL_ELEMENT_PTR pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, teeElementName.c_str());
            
        WHEN( "The Queue is linked to the Tee" )
        {
            pQueue->LinkToSink(pTee);
            
            THEN( "The relation ship of source and sink Elementr are setup correctly" )
            {
                REQUIRE( pQueue->m_pSink == pTee );
                REQUIRE( pTee->m_pSource == pQueue );
            }
        }
    }
}

SCENARIO( "Two Elementrs are unlinked correctly", "[Elementr]" )
{
    GIVEN( "A Queue Elementr linked to a Tee Elementr" ) 
    {
        std::string queueElementName  = "test-queue";
        std::string teeElementName = "test-tee";
        
        DSL_ELEMENT_PTR pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, queueElementName.c_str());
        DSL_ELEMENT_PTR pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, teeElementName.c_str());

        pQueue->LinkToSink(pTee);
            
        WHEN( "The Queue is unlinked to the Tee" )
        {
            pQueue->UnlinkFromSink();

            THEN( "The relation ship of source and sink Elementr are setup correctly" )
            {
                REQUIRE( pQueue->m_pSink == nullptr );
                REQUIRE( pTee->m_pSource == nullptr );                
            }
        }
    }
}

SCENARIO( "A new GhostPad can be added to an Elementr", "[Elementr]" )
{
    GIVEN( "A Queue Elementr in memory" ) 
    {
        std::string parentBintrName  = "test-queue";
        std::string queueElementName  = "test-queue";

        DSL_ELEMENT_PTR pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, queueElementName.c_str());
        DSL_TEST_BINTR_PTR pParentBintr = DSL_TEST_BINTR_NEW(parentBintrName.c_str());
        pParentBintr->AddChild(pQueue);

        WHEN( "The StaticPadtr is created" )
        {
            pQueue->AddGhostPadToParent("sink");
            
            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pQueue->IsInUse() == true );
            }
        }
    }
}
