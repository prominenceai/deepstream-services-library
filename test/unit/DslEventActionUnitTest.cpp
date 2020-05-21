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
#include "DslDetectionEvent.h"
#include "DslEventAction.h"

using namespace DSL;

SCENARIO( "A new LogEventAction is created correctly", "[EventAction]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string eventActionName = "event-action";

        WHEN( "A new EventAction is created" )
        {
            DSL_EVENT_ACTION_LOG_PTR pEventAction = 
                DSL_EVENT_ACTION_LOG_NEW(eventActionName.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pEventAction->GetCStrName();
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

        DSL_EVENT_FIRST_OCCURRENCE_PTR pDetectionEvent = 
            DSL_EVENT_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

        DSL_EVENT_ACTION_LOG_PTR pEventAction = 
            DSL_EVENT_ACTION_LOG_NEW(eventActionName.c_str());
            
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = INT32_MAX;
        objectMeta.object_id = INT64_MAX;
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        WHEN( "A new Event is created" )
        {
            uint64_t eventId(1);
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pEventAction->HandleOccurrence(pDetectionEvent, eventId, &frameMeta, &objectMeta);
            }
        }
    }
}
