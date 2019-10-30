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

SCENARIO( "An Elementr is constructed correctly", "[Elementr]" )
{
    GIVEN( "A Parent bin in memory" )
    {
        std::string binName = "test-bin";
        std::string elementName  = "test-element";
        
        GstElement* pBin = gst_bin_new((gchar*)binName.c_str());

        WHEN( "A child Elmentr is created" )
        {
            std::shared_ptr<DSL::Elementr> pElement = 
                std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
                NVDS_ELEM_QUEUE, elementName.c_str(), pBin));
            
            THEN( "Its member variables are initialized correctly" )
            {
                REQUIRE( pElement->m_name == elementName );
                REQUIRE( pElement->m_pElement != NULL );
                REQUIRE( pElement->m_pParentBin == pBin );
            }
        }
    }
}

SCENARIO( "A Elementr is constructed correctly", "[Elementr]" )
{
    GIVEN( "A Queue Elementr and a Tee Elementr in memory" ) 
    {
        std::string binName = "test-bin";
        std::string queueElementName  = "test-queue";
        std::string teeElementName = "test-tee";
        
        GstElement* pBin = gst_bin_new((gchar*)binName.c_str());

        std::shared_ptr<DSL::Elementr> pQueue = 
            std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
            NVDS_ELEM_QUEUE, queueElementName.c_str(), pBin));
        std::shared_ptr<DSL::Elementr> pTee = 
            std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
            NVDS_ELEM_TEE, teeElementName.c_str(), pBin));
            
        WHEN( "The Queue is Linked to the Tee" )
        {
            pQueue->LinkTo(pTee);
            
            REQUIRE( pQueue->m_pLinkedSinkElementr == pTee );
            REQUIRE( pTee->m_pLinkedSourceElementr == pQueue );
        }
    }
}

SCENARIO( "Two Elementrs can be linked correctly", "[Elementr]" )
{
    GIVEN( "A Queue Elementr and a Tee Elementr in memory" ) 
    {
        std::string binName = "test-bin";
        std::string queueElementName  = "test-queue";
        std::string teeElementName = "test-tee";
        
        GstElement* pBin = gst_bin_new((gchar*)binName.c_str());

        std::shared_ptr<DSL::Elementr> pQueue = 
            std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
            NVDS_ELEM_QUEUE, queueElementName.c_str(), pBin));
        std::shared_ptr<DSL::Elementr> pTee = 
            std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
            NVDS_ELEM_TEE, teeElementName.c_str(), pBin));
            
        WHEN( "The Queue is Linked to the Tee" )
        {
            pQueue->LinkTo(pTee);
            
            THEN( "The relation ship of source and sink Elementr are setup correctly" )
            {
                REQUIRE( pQueue->m_pLinkedSinkElementr == pTee );
                REQUIRE( pTee->m_pLinkedSourceElementr == pQueue );                
            }
        }
    }
}

SCENARIO( "Two Elementrs can be unlinked correctly", "[Elementr]" )
{
    GIVEN( "A Queue Elementr linked to a Tee Elementr " ) 
    {
        std::string binName = "test-bin";
        std::string queueElementName  = "test-queue";
        std::string teeElementName = "test-tee";
        
        GstElement* pBin = gst_bin_new((gchar*)binName.c_str());

        std::shared_ptr<DSL::Elementr> pQueue = 
            std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
            NVDS_ELEM_QUEUE, queueElementName.c_str(), pBin));
        std::shared_ptr<DSL::Elementr> pTee = 
            std::shared_ptr<DSL::Elementr>(new DSL::Elementr(
            NVDS_ELEM_TEE, teeElementName.c_str(), pBin));

        pQueue->Unlink();
            
        WHEN( "The Queue is unLinked from the Tee" )
        {
            
            THEN( "The relation ship of source and sink Elementr are removed correctly" )
            {
                REQUIRE( pQueue->m_pLinkedSinkElementr == nullptr );
                REQUIRE( pTee->m_pLinkedSourceElementr == nullptr );                
            }
        }
    }
}
