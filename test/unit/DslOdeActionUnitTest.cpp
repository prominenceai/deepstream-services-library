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

SCENARIO( "A new LogOdeAction is created correctly", "[LogOdeAction]" )
{
    GIVEN( "Attributes for a new LogOdeAction" ) 
    {
        std::string eventActionName("event-action");

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

SCENARIO( "A LogOdeAction handles an Event Occurence correctly", "[LogOdeAction]" )
{
    GIVEN( "A new LogOdeAction" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);
        
        std::string eventActionName = "event-action";

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

        DSL_ODE_ACTION_LOG_PTR pOdeAction = 
            DSL_ODE_ACTION_LOG_NEW(eventActionName.c_str());

        WHEN( "A new Event is created" )
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
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new PrintOdeAction is created correctly", "[PrintOdeAction]" )
{
    GIVEN( "Attributes for a new PrintOdeAction" ) 
    {
        std::string eventActionName("event-action");

        WHEN( "A new EventAction is created" )
        {
            DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
                DSL_ODE_ACTION_PRINT_NEW(eventActionName.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( eventActionName == retName );
            }
        }
    }
}

SCENARIO( "A PrintOdeAction handles an Event Occurence correctly", "[PrintOdeAction]" )
{
    GIVEN( "A new PrintOdeAction" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);
        
        std::string eventActionName = "event-action";

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(eventActionName.c_str());

        WHEN( "A new Event is created" )
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
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

