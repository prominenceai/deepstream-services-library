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
#include "DslCustomBintr.h"

using namespace DSL;

static const std::string customComponentName("custom-component");

SCENARIO( "A CustomBintr is created correctly",  "[CustomBintr]" )
{
    GIVEN( "Attributes for a new Gst" ) 
    {

        WHEN( "The CustomBintr is created" )
        {
            DSL_CUSTOM_BINTR_PTR pCustomBintr = DSL_CUSTOM_BINTR_NEW(customComponentName.c_str());

            THEN( "The object is created correctly")
            {
                std::string retName = pCustomBintr->GetName();
                REQUIRE( retName == customComponentName );
            }
        }
    }
}

SCENARIO( "A CustomBintr can add and remove a child element",  "[CustomBintr]" )
{
    GIVEN( "A new CustomBintr and Elementr" ) 
    {
        static const std::string elementName("element");
        DSL_CUSTOM_BINTR_PTR pCustomBintr = DSL_CUSTOM_BINTR_NEW(customComponentName.c_str());
        DSL_ELEMENT_PTR pGstElementr = DSL_ELEMENT_NEW("identity", elementName.c_str());
        
        WHEN( "The an Element is added to the CustomBintr" )
        {
            REQUIRE( pCustomBintr->AddChild(pGstElementr) == true );
            
            // The second call must fail
            REQUIRE( pCustomBintr->AddChild(pGstElementr) == false );
            THEN( "The same Element can be removed correctly")
            {
                REQUIRE( pCustomBintr->RemoveChild(pGstElementr) == true );
            
                // The second call must fail
                REQUIRE( pCustomBintr->RemoveChild(pGstElementr) == false );
            }
        }
    }
}

SCENARIO( "A CustomBintr can can link and unlink with a single child element",  
    "[CustomBintr]" )
{
    GIVEN( "A new CustomBintr with an Elementr" ) 
    {
        static const std::string elementName("element");
        DSL_CUSTOM_BINTR_PTR pCustomBintr = DSL_CUSTOM_BINTR_NEW(customComponentName.c_str());
        DSL_ELEMENT_PTR pGstElementr = DSL_ELEMENT_NEW("identity", elementName.c_str());
        
        REQUIRE( pCustomBintr->AddChild(pGstElementr) == true );
        
        WHEN( "The CustomBintr is successfully linked" )
        {
            REQUIRE( pCustomBintr->LinkAll() == true );
            
            THEN( "The CustomBintr can be successfully unlinked")
            {
                pCustomBintr->UnlinkAll();
            }
        }
    }
}
SCENARIO( "A CustomBintr can can link and unlink with a multiple child elements",  
    "[CustomBintr]" )
{
    GIVEN( "A new CustomBintr with three Elementrs" ) 
    {
        static const std::string elementName1("element-1");
        static const std::string elementName2("element-2");
        static const std::string elementName3("element-3");
        DSL_CUSTOM_BINTR_PTR pCustomBintr = DSL_CUSTOM_BINTR_NEW(customComponentName.c_str());
        DSL_ELEMENT_PTR pGstElementr1 = DSL_ELEMENT_NEW("identity", 
            elementName1.c_str());
        DSL_ELEMENT_PTR pGstElementr2 = DSL_ELEMENT_NEW("identity",
            elementName2.c_str());
        DSL_ELEMENT_PTR pGstElementr3 = DSL_ELEMENT_NEW("identity", 
            elementName3.c_str());
        
        REQUIRE( pCustomBintr->AddChild(pGstElementr1) == true );
        REQUIRE( pCustomBintr->AddChild(pGstElementr2) == true );
        REQUIRE( pCustomBintr->AddChild(pGstElementr3) == true );
        
        WHEN( "The CustomBintr is successfully linked" )
        {
            REQUIRE( pCustomBintr->LinkAll() == true );
            
            THEN( "The CustomBintr can be successfully unlinked")
            {
                pCustomBintr->UnlinkAll();
            }
        }
    }
}