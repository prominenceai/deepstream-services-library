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
#include "DslOdeHandlerBintr.h"

using namespace DSL;

SCENARIO( "A new OdeHandlerBintr is created correctly", "[OdeHandlerBintr]" )
{
    GIVEN( "Attributes for a new OdeHandlerBintr" ) 
    {
        std::string odeHandlerName = "ode-handler";

        WHEN( "A new OdeHandler is created" )
        {
            DSL_ODE_HANDLER_PTR pOdeHandlerBintr = DSL_ODE_HANDLER_NEW(odeHandlerName.c_str());

            THEN( "The OdeHandlerBintr's memebers are setup and returned correctly" )
            {
                REQUIRE( pOdeHandlerBintr->GetEnabled() == true );
                REQUIRE( pOdeHandlerBintr->GetGstObject() != NULL );
            }
        }
    }
}

SCENARIO( "A new OdeHandlerBintr can Disable and Re-enable", "[OdeHandlerBintr]" )
{
    GIVEN( "A new OdeHandlerBintr" ) 
    {
        std::string odeHandlerName = "ode-handler";

        DSL_ODE_HANDLER_PTR pOdeHandlerBintr = DSL_ODE_HANDLER_NEW(odeHandlerName.c_str());
        REQUIRE( pOdeHandlerBintr->GetEnabled() == true );

        // Attempting to enable and enabled OdeHandler must fail
        REQUIRE( pOdeHandlerBintr->SetEnabled(true) == false );

        WHEN( "A new OdeHandler is Disabled'" )
        {
            REQUIRE( pOdeHandlerBintr->SetEnabled(false) == true );

            // Attempting to disable a disabled OdeHandler must fail
            REQUIRE( pOdeHandlerBintr->SetEnabled(false) == false );

            THEN( "The OdeHandlerBintr can be enabled again" )
            {
                REQUIRE( pOdeHandlerBintr->SetEnabled(true) == true );
                REQUIRE( pOdeHandlerBintr->GetEnabled() == true );
            }
        }
    }
}

SCENARIO( "A new OdeHandlerBintr can LinkAll Child Elementrs", "[OdeHandlerBintr]" )
{
    GIVEN( "A new OdeHandlerBintr in an Unlinked state" ) 
    {
        std::string odeHandlerName = "ode-handler";

        DSL_ODE_HANDLER_PTR pOdeHandlerBintr = DSL_ODE_HANDLER_NEW(odeHandlerName.c_str());

        WHEN( "A new OdeHandlerBintr is Linked" )
        {
            REQUIRE( pOdeHandlerBintr->LinkAll() == true );

            THEN( "The OdeHandlerBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pOdeHandlerBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked OdeHandlerBintr can UnlinkAll Child Elementrs", "[OdeHandlerBintr]" )
{
    GIVEN( "A OdeHandlerBintr in a linked state" ) 
    {
        std::string odeHandlerName = "ode-handler";

        DSL_ODE_HANDLER_PTR pOdeHandlerBintr = DSL_ODE_HANDLER_NEW(odeHandlerName.c_str());

        REQUIRE( pOdeHandlerBintr->LinkAll() == true );

        WHEN( "A OdeHandlerBintr is Linked" )
        {
            pOdeHandlerBintr->UnlinkAll();

            THEN( "The OdeHandlerBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pOdeHandlerBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A OdeHandlerBintr can add and remove a DetectionEvent", "[OdeHandlerBintr]" )
{
    GIVEN( "A new OdeHandlerBintr and DetectionEvent" ) 
    {
        std::string odeHandlerName = "ode-handler";
        std::string odeTypeName = "first-occurence";
        uint classId(1);

        DSL_ODE_HANDLER_PTR pOdeHandlerBintr = DSL_ODE_HANDLER_NEW(odeHandlerName.c_str());

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(odeTypeName.c_str(), classId);

        WHEN( "A the Event is added to the ReportBintr" )
        {
            REQUIRE( pOdeHandlerBintr->AddChild(pFirstOccurrenceEvent) == true );
            
            // ensure that the Event can not be added twice
            REQUIRE( pOdeHandlerBintr->AddChild(pFirstOccurrenceEvent) == false );
            
            THEN( "The Event can be found and removed" )
            {
                REQUIRE( pOdeHandlerBintr->IsChild(pFirstOccurrenceEvent) == true );
                REQUIRE( pFirstOccurrenceEvent->IsParent(pOdeHandlerBintr) == true );
                REQUIRE( pFirstOccurrenceEvent->IsInUse() == true );
                
                REQUIRE( pOdeHandlerBintr->RemoveChild(pFirstOccurrenceEvent) == true );
                
                REQUIRE( pOdeHandlerBintr->IsChild(pFirstOccurrenceEvent) == false );
                REQUIRE( pFirstOccurrenceEvent->GetName() == odeTypeName );
                REQUIRE( pFirstOccurrenceEvent->IsParent(pOdeHandlerBintr) == false );
                REQUIRE( pFirstOccurrenceEvent->IsInUse() == false );
                // ensure removal fails on second call, 
                REQUIRE( pOdeHandlerBintr->RemoveChild(pFirstOccurrenceEvent) == false );
            }
        }
    }
}

