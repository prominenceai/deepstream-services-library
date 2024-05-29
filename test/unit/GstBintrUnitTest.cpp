/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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
#include "DslGstBintr.h"

using namespace DSL;

static const std::string gstBinName("test-bin");

SCENARIO( "A GstBintr is created correctly",  "[GstBintr]" )
{
    GIVEN( "Attributes for a new Gst" ) 
    {

        WHEN( "The GstBintr is created" )
        {
            DSL_GST_BINTR_PTR pGstBintr = DSL_GST_BINTR_NEW(gstBinName.c_str());

            THEN( "The object is created correctly")
            {
                std::string retName = pGstBintr->GetName();
                REQUIRE( retName == gstBinName );
            }
        }
    }
}

SCENARIO( "A GstBintr can add and remove a child element",  "[GstBintr]" )
{
    GIVEN( "A new GstBintr and Elementr" ) 
    {
        static const std::string elementName("element");
        DSL_GST_BINTR_PTR pGstBintr = DSL_GST_BINTR_NEW(gstBinName.c_str());
        DSL_ELEMENT_PTR pGstElementr = DSL_ELEMENT_NEW("queue", elementName.c_str());
        
        WHEN( "The an Element is added to the GstBintr" )
        {
            REQUIRE( pGstBintr->AddChild(pGstElementr) == true );
            
            // The second call must fail
            REQUIRE( pGstBintr->AddChild(pGstElementr) == false );
            THEN( "The same Element can be removed correctly")
            {
                REQUIRE( pGstBintr->RemoveChild(pGstElementr) == true );
            
                // The second call must fail
                REQUIRE( pGstBintr->RemoveChild(pGstElementr) == false );
            }
        }
    }
}

SCENARIO( "A GstBintr can can link and unlink with a single child element",  "[GstBintr]" )
{
    GIVEN( "A new GstBintr with an Elementr" ) 
    {
        static const std::string elementName("element");
        DSL_GST_BINTR_PTR pGstBintr = DSL_GST_BINTR_NEW(gstBinName.c_str());
        DSL_ELEMENT_PTR pGstElementr = DSL_ELEMENT_NEW("queue", elementName.c_str());
        
        REQUIRE( pGstBintr->AddChild(pGstElementr) == true );
        
        WHEN( "The GstBintr is successfully linked" )
        {
            REQUIRE( pGstBintr->LinkAll() == true );
            
            THEN( "The GstBintr can be successfully unlinked")
            {
                pGstBintr->UnlinkAll();
            }
        }
    }
}
SCENARIO( "A GstBintr can can link and unlink with a multiple child elements",  
    "[GstBintr]" )
{
    GIVEN( "A new GstBintr with three Elementrs" ) 
    {
        static const std::string elementName1("element-1");
        static const std::string elementName2("element-2");
        static const std::string elementName3("element-3");
        DSL_GST_BINTR_PTR pGstBintr = DSL_GST_BINTR_NEW(gstBinName.c_str());
        DSL_ELEMENT_PTR pGstElementr1 = DSL_ELEMENT_NEW("queue", 
            elementName1.c_str());
        DSL_ELEMENT_PTR pGstElementr2 = DSL_ELEMENT_NEW("queue",
            elementName2.c_str());
        DSL_ELEMENT_PTR pGstElementr3 = DSL_ELEMENT_NEW("queue", 
            elementName3.c_str());
        
        REQUIRE( pGstBintr->AddChild(pGstElementr1) == true );
        REQUIRE( pGstBintr->AddChild(pGstElementr2) == true );
        REQUIRE( pGstBintr->AddChild(pGstElementr3) == true );
        
        WHEN( "The GstBintr is successfully linked" )
        {
            REQUIRE( pGstBintr->LinkAll() == true );
            
            THEN( "The GstBintr can be successfully unlinked")
            {
                pGstBintr->UnlinkAll();
            }
        }
    }
}