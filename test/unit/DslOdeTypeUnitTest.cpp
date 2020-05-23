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
#include "DslOdeType.h"
#include "DslOdeAction.h"

using namespace DSL;

SCENARIO( "A new FirstOccurrenceEvent is created correctly", "[DetectionEvent]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string eventName = "first-occurence";
        uint classId(1);

        WHEN( "A new DetectionEvent is created" )
        {
            DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
                DSL_ODE_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                REQUIRE( pFirstOccurrenceEvent->GetClassId() == classId );
                uint minWidth(123), minHeight(123);
                pFirstOccurrenceEvent->GetMinDimensions(&minWidth, &minHeight);
                REQUIRE( minWidth == 0 );
                REQUIRE( minHeight == 0 );
                uint minFrameCountN(123), minFrameCountD(123);
                pFirstOccurrenceEvent->GetMinFrameCount(&minFrameCountN, &minFrameCountD);
                REQUIRE( minFrameCountN == 0 );
                REQUIRE( minFrameCountD == 0 );
            }
        }
    }
}

SCENARIO( "A FirstOccurrenceEvent can detect an Occurence only once", "[FirstOccurrenceEvent]" )
{
    GIVEN( "A new FirstOccurrenceEvent with default criteria" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);

        std::string eventActionName("event-action");

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

        DSL_ODE_ACTION_LOG_PTR pEventAction = 
            DSL_ODE_ACTION_LOG_NEW(eventActionName.c_str());
            
        REQUIRE( pFirstOccurrenceEvent->AddChild(pEventAction) == true );        

        WHEN( "A first occurent event is simulated" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Event's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The FirstOccurenceEvent is detected and only once" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(&frameMeta, &objectMeta) == true );
                // second time must fail
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(&frameMeta, &objectMeta) == false );
            }
        }
    }
}



