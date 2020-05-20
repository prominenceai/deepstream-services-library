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
#include "DslReporterBintr.h"

using namespace DSL;

SCENARIO( "A new ReporterBintr is created correctly", "[ReporterBintr]" )
{
    GIVEN( "Attributes for a new ReporterBintr" ) 
    {
        std::string reporterName = "reporter";

        WHEN( "A new Reporter is created" )
        {
            DSL_REPORTER_PTR pReporterBintr = DSL_REPORTER_NEW(reporterName.c_str());

            THEN( "The ReporterBintr's memebers are setup and returned correctly" )
            {
                REQUIRE( pReporterBintr->GetReportingEnabled() == true );
                REQUIRE( pReporterBintr->GetGstObject() != NULL );
            }
        }
    }
}

SCENARIO( "A new ReporterBintr can Disable and Re-enable Reporting", "[ReporterBintr]" )
{
    GIVEN( "A new ReporterBintr" ) 
    {
        std::string reporterName = "reporter";

        DSL_REPORTER_PTR pReporterBintr = DSL_REPORTER_NEW(reporterName.c_str());
        REQUIRE( pReporterBintr->GetReportingEnabled() == true );

        // Attempting to enable and enabled Reporter must fail
        REQUIRE( pReporterBintr->SetReportingEnabled(true) == false );

        WHEN( "A new Reporter's reporting is Disabled'" )
        {
            REQUIRE( pReporterBintr->SetReportingEnabled(false) == true );

            // Attempting to disable a disabled Reporter must fail
            REQUIRE( pReporterBintr->SetReportingEnabled(false) == false );

            THEN( "The ReporterBintr's reporting can be enabled again" )
            {
                REQUIRE( pReporterBintr->SetReportingEnabled(true) == true );
                REQUIRE( pReporterBintr->GetReportingEnabled() == true );
            }
        }
    }
}

SCENARIO( "A new ReporterBintr can LinkAll Child Elementrs", "[ReporterBintr]" )
{
    GIVEN( "A new ReporterBintr in an Unlinked state" ) 
    {
        std::string reporterName = "reporter";

        DSL_REPORTER_PTR pReporterBintr = DSL_REPORTER_NEW(reporterName.c_str());

        WHEN( "A new ReporterBintr is Linked" )
        {
            REQUIRE( pReporterBintr->LinkAll() == true );

            THEN( "The ReporterBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pReporterBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked ReporterBintr can UnlinkAll Child Elementrs", "[ReporterBintr]" )
{
    GIVEN( "A ReporterBintr in a linked state" ) 
    {
        std::string reporterName = "reporter";

        DSL_REPORTER_PTR pReporterBintr = DSL_REPORTER_NEW(reporterName.c_str());

        REQUIRE( pReporterBintr->LinkAll() == true );

        WHEN( "A ReporterBintr is Linked" )
        {
            pReporterBintr->UnlinkAll();

            THEN( "The ReporterBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pReporterBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A ReporterBintr can add and remove a DetectionEvent", "[ReporterBintr]" )
{
    GIVEN( "A new ReporterBintr and DetectionEvent" ) 
    {
        std::string reporterName = "reporter";
        std::string eventName = "first-occurence";
        uint classId(1);

        DSL_REPORTER_PTR pReporterBintr = DSL_REPORTER_NEW(reporterName.c_str());

        DSL_EVENT_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_EVENT_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

        WHEN( "A the Event is added to the ReportBintr" )
        {
            REQUIRE( pReporterBintr->AddChild(pFirstOccurrenceEvent) == true );
            
            // ensure that the Event can not be added twice
            REQUIRE( pReporterBintr->AddChild(pFirstOccurrenceEvent) == false );
            
            THEN( "The Event can be found and removed" )
            {
                REQUIRE( pReporterBintr->IsChild(pFirstOccurrenceEvent) == true );
                REQUIRE( pFirstOccurrenceEvent->IsParent(pReporterBintr) == true );
                REQUIRE( pFirstOccurrenceEvent->IsInUse() == true );
                
                REQUIRE( pReporterBintr->RemoveChild(pFirstOccurrenceEvent) == true );
                
                REQUIRE( pReporterBintr->IsChild(pFirstOccurrenceEvent) == false );
                REQUIRE( pFirstOccurrenceEvent->GetName() == eventName );
                REQUIRE( pFirstOccurrenceEvent->IsParent(pReporterBintr) == false );
                REQUIRE( pFirstOccurrenceEvent->IsInUse() == false );

                // ensure removal fails on second call, 
                REQUIRE( pReporterBintr->RemoveChild(pFirstOccurrenceEvent) == false );
            }
        }
    }
}

