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

SCENARIO( "A new OdeType is created correctly", "[OdeType]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string odeTypeName("first-occurence");
        uint classId(1);

        WHEN( "A new OdeType is created" )
        {
            DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
                DSL_ODE_FIRST_OCCURRENCE_NEW(odeTypeName.c_str(), classId);

            THEN( "The OdeTypes's memebers are setup and returned correctly" )
            {
                REQUIRE( pFirstOccurrenceEvent->GetClassId() == classId );
                REQUIRE( pFirstOccurrenceEvent->GetSourceId() == 0 );
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
        std::string odeTypeName("first-occurence");
        uint classId(1);

        std::string odeActionName("event-action");

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(odeTypeName.c_str(), classId);

        DSL_ODE_ACTION_PRINT_PTR pEventAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pFirstOccurrenceEvent->AddChild(pEventAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  // required to process
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match Detections Event's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "A First Occurrence ODE is simulated" )
        {
            REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            
            THEN( "All other objects are ignored" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "A FirstOccurrenceEvent checks for Minimum Confidence correctly", "[FirstOccurrenceEvent]" )
{
    GIVEN( "A new FirstOccurrenceEvent with default criteria" ) 
    {
        std::string odeTypeName("first-occurence");
        uint classId(1);

        std::string odeActionName("event-action");

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(odeTypeName.c_str(), classId);
            
        // Set the minumum confidence value for detection
        pFirstOccurrenceEvent->SetMinConfidence(0.5);    

        DSL_ODE_ACTION_PRINT_PTR pEventAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
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
            
            // set the confidence level to just below the minumum
            objectMeta.confidence = 0.4999; 
            
            THEN( "The FirstOccurenceEvent is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "A FirstOccurrenceEvent checks for SourceId correctly", "[FirstOccurrenceEvent]" )
{
    GIVEN( "A new FirstOccurrenceEvent with default criteria" ) 
    {
        std::string odeTypeName("first-occurence");
        uint classId(1);
        uint sourceId(2);

        std::string odeActionName("event-action");

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(odeTypeName.c_str(), classId);
            
        // Set the minumum confidence value for detection
        pFirstOccurrenceEvent->SetSourceId(sourceId);    
        
        REQUIRE( pFirstOccurrenceEvent->GetSourceId() == sourceId );

        DSL_ODE_ACTION_PRINT_PTR pEventAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pFirstOccurrenceEvent->AddChild(pEventAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  // required to process
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match Detections Event's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.9999; 
        
        WHEN( "The Source ID matches the filter" )
        {
            // match filter for first call
            frameMeta.source_id = sourceId;
            
            THEN( "The FirstOccurenceEvent is detected because of the filter match" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            // unmatch filter for second call
            frameMeta.source_id = sourceId+1;
            
            THEN( "The FirstOccurenceEvent is detected because of the filter match" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "A FirstOccurrenceEvent checks for Minimum Dimensions correctly", "[FirstOccurrenceEvent]" )
{
    GIVEN( "A new FirstOccurrenceEvent with minimum criteria" ) 
    {
        std::string odeTypeName("first-occurence");
        uint classId(1);

        std::string odeActionName("event-action");

        DSL_ODE_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_FIRST_OCCURRENCE_NEW(odeTypeName.c_str(), classId);

        DSL_ODE_ACTION_PRINT_PTR pEventAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pFirstOccurrenceEvent->AddChild(pEventAction) == true );        

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
        objectMeta.confidence = 0.4999; 

        WHEN( "the Min Width is set above the Object's Width" )
        {
            pFirstOccurrenceEvent->SetMinDimensions(201, 0);    
            
            THEN( "The FirstOccurenceEvent is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Width is set below the Object's Width" )
        {
            pFirstOccurrenceEvent->SetMinDimensions(199, 0);    
            
            THEN( "The FirstOccurenceEvent is detected because of the minimum criteria" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Min Height is set above the Object's Height" )
        {
            pFirstOccurrenceEvent->SetMinDimensions(0, 101);    
            
            THEN( "The FirstOccurenceEvent is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Height is set below the Object's Height" )
        {
            pFirstOccurrenceEvent->SetMinDimensions(0, 99);    
            
            THEN( "The FirstOccurenceEvent is detected because of the minimum criteria" )
            {
                REQUIRE( pFirstOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An EveryOccurrenceEvent detects every Occurrence", "[EveryOccurrenceEvent]" )
{
    GIVEN( "An new EveryOccurrenceEvent with default criteria" ) 
    {
        std::string odeTypeName("every-occurence");
        uint classId(1);

        std::string odeActionName("event-action");

        DSL_ODE_EVERY_OCCURRENCE_PTR pEveryOccurrenceEvent = 
            DSL_ODE_EVERY_OCCURRENCE_NEW(odeTypeName.c_str(), classId);

        DSL_ODE_ACTION_PRINT_PTR pEventAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pEveryOccurrenceEvent->AddChild(pEventAction) == true );        

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
            
        WHEN( "A First Occurrence ODE is simulated" )
        {
            REQUIRE( pEveryOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            
            THEN( "Every objects is still detected aftwards" )
            {
                REQUIRE( pEveryOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
                REQUIRE( pEveryOccurrenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A FirstAbsenceEvent detects only the first frame Absense of Object Detection", "[FirstAbsenceEvent]" )
{
    GIVEN( "A new FirstAbsenceEvent with default criteria" ) 
    {
        std::string odeTypeName("first-absence");
        uint classId(1);

        std::string odeActionName("event-action");

        DSL_ODE_FIRST_ABSENCE_PTR pFirstAbsenceEvent = 
            DSL_ODE_FIRST_ABSENCE_NEW(odeTypeName.c_str(), classId);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pFirstAbsenceEvent->AddChild(pOdeAction) == true );        

        WHEN( "An ODE is simulated" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId+1; // must NOT match ODE's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The FirstAbsesnceEvent takes action only once" )
            {
                // No occurrence should be found
                REQUIRE( pFirstAbsenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
                // First Absence should take action
                REQUIRE( pFirstAbsenceEvent->PostProcessFrame(NULL, &frameMeta) == true );
                 // No occurrence should be found
                REQUIRE( pFirstAbsenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
                // second time, First Absence should NOT take action
                REQUIRE( pFirstAbsenceEvent->PostProcessFrame(NULL, &frameMeta) == false );
            }
        }
    }
}

SCENARIO( "A FirstAbsenceEvent checks for Minimum Confidence correctly", "[FirstAbsenceEvent]" )
{
    GIVEN( "A new FirstAbsenceEvent with default criteria" ) 
    {
        std::string odeTypeName("first-absence");
        uint classId(1);

        std::string odeActionName("event-action");

        DSL_ODE_FIRST_ABSENCE_PTR pFirstAbsenceEvent = 
            DSL_ODE_FIRST_ABSENCE_NEW(odeTypeName.c_str(), classId);

        // Set the minumum confidence value for detection
        pFirstAbsenceEvent->SetMinConfidence(0.5);    

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pFirstAbsenceEvent->AddChild(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  // required to process
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; //match ODE Types's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        WHEN( "When an Object's meta confidence is below the minimum" )
        {
            // set the confidence level to just below the minumum
            objectMeta.confidence = 0.4999; 
            
            THEN( "The FirstAbsesnceEvent takes action" )
            {
                // No occurrence should NOT be found
                REQUIRE( pFirstAbsenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
                // First Absence should take action
                REQUIRE( pFirstAbsenceEvent->PostProcessFrame(NULL, &frameMeta) == true );
            }
        }
        WHEN( "When an Object's meta confidence is equal to the minimum" )
        {
            // set the confidence level to just below the minumum
            objectMeta.confidence = 0.5; 
            
            THEN( "The FirstAbsesnceEvent does not take action" )
            {
                // No occurrence should be found
                REQUIRE( pFirstAbsenceEvent->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
                // First Absence should NOT take action
                REQUIRE( pFirstAbsenceEvent->PostProcessFrame(NULL, &frameMeta) == false );
            }
        }
    }
}
