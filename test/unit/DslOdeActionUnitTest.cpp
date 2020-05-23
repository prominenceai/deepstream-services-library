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

SCENARIO( "A new LogEventAction is created correctly", "[EventAction]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string eventActionName = "event-action";

        WHEN( "A new EventAction is created" )
        {
            DSL_ODE_ACTION_LOG_PTR pOdeAction = 
                DSL_ODE_ACTION_LOG_NEW(eventActionName.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( eventActionName == retName );
            }
        }
    }
}

SCENARIO( "A LogEventAction handles an Event Occurence correctly", "[EventAction]" )
{
    GIVEN( "A new LogDetectionEvent" ) 
    {
        std::string eventName = "first-occurence";
        uint classId(1);
        
        std::string eventActionName = "event-action";

        DSL_ODE_ACTION_LOG_PTR pOdeAction = 
            DSL_ODE_ACTION_LOG_NEW(eventActionName.c_str());
            
        DSL_ODE_OCCURRENCE_PTR pOdeOccurrence = DSL_ODE_OCCURRENCE_NEW();
        eventName.copy(pOdeOccurrence->event_name, MAX_NAME_SIZE-1, 0);
        
        pOdeOccurrence->event_type = 1;
        pOdeOccurrence->event_id = 444;
        pOdeOccurrence->ntp_timestamp = UINT64_MAX;
        pOdeOccurrence->source_id = 3;
        pOdeOccurrence->frame_num = 12345;
        pOdeOccurrence->source_frame_width = 1280;
        pOdeOccurrence->source_frame_height = 720;
        pOdeOccurrence->class_id = 1;
        pOdeOccurrence->object_id = 123; 
        pOdeOccurrence->box.left = 123;
        pOdeOccurrence->box.top = 123;
        pOdeOccurrence->box.width = 300;
        pOdeOccurrence->box.height = 200;
        pOdeOccurrence->min_confidence = 0.5;
        pOdeOccurrence->box_criteria.top = 0;
        pOdeOccurrence->box_criteria.left = 0;
        pOdeOccurrence->box_criteria.width = 0;
        pOdeOccurrence->box_criteria.height = 0;
        pOdeOccurrence->min_frame_count_n = 10;
        pOdeOccurrence->min_frame_count_d = 30;

        WHEN( "A new Event is created" )
        {
            uint64_t eventId(1);
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pOdeOccurrence);
            }
        }
    }
}
