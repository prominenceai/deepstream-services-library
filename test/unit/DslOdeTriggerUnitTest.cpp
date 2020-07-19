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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"
#include "DslOdeArea.h"

using namespace DSL;

SCENARIO( "A new OdeTrigger is created correctly", "[OdeTrigger]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);

        WHEN( "A new OdeTrigger is created" )
        {
            DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

            THEN( "The OdeTriggers's memebers are setup and returned correctly" )
            {
                REQUIRE( pOdeTrigger->GetEnabled() == true );
                REQUIRE( pOdeTrigger->GetClassId() == classId );
                REQUIRE( pOdeTrigger->GetSourceId() == DSL_ODE_ANY_SOURCE );
                uint minWidth(123), minHeight(123);
                pOdeTrigger->GetMinDimensions(&minWidth, &minHeight);
                REQUIRE( minWidth == 0 );
                REQUIRE( minHeight == 0 );
                uint maxWidth(123), maxHeight(123);
                pOdeTrigger->GetMaxDimensions(&maxWidth, &maxHeight);
                REQUIRE( maxWidth == 0 );
                REQUIRE( maxHeight == 0 );
                uint minFrameCountN(123), minFrameCountD(123);
                pOdeTrigger->GetMinFrameCount(&minFrameCountN, &minFrameCountD);
                REQUIRE( minFrameCountN == 1 );
                REQUIRE( minFrameCountD == 1 );
                REQUIRE( pOdeTrigger->GetInferDoneOnlySetting() == false );
            }
        }
    }
}

SCENARIO( "An OdeTrigger checks its enabled setting ", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "The ODE Type is enabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetEnabled(true);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Type is disabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetEnabled(false);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An ODE checks its minimum confidence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.5;
        
        WHEN( "The ODE Type's minimum confidence is less than the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.4999);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Type's minimum confidence is equal to the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.5);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Type's minimum confidence is greater tahn the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.5001);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}


SCENARIO( "A OdeTrigger checks for SourceId correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0);
        uint sourceId(2);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);
            
        // Set the minumum confidence value for detection
        pOdeTrigger->SetSourceId(sourceId);    
        
        REQUIRE( pOdeTrigger->GetSourceId() == sourceId );

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 1;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.9999; 
        
        WHEN( "The the Source ID filter is disabled" )
        {
            pOdeTrigger->SetSourceId(DSL_ODE_ANY_SOURCE);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            pOdeTrigger->SetSourceId(1);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID does not matche the filter" )
        {
            pOdeTrigger->SetSourceId(2);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "A OdeTrigger checks for Minimum Dimensions correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with minimum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        objectMeta.confidence = 0.4999; 

        WHEN( "the Min Width is set above the Object's Width" )
        {
            pOdeTrigger->SetMinDimensions(201, 0);    
            
            THEN( "The OdeTrigger is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Width is set below the Object's Width" )
        {
            pOdeTrigger->SetMinDimensions(199, 0);    
            
            THEN( "The OdeTrigger is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Min Height is set above the Object's Height" )
        {
            pOdeTrigger->SetMinDimensions(0, 101);    
            
            THEN( "The OdeTrigger is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Height is set below the Object's Height" )
        {
            pOdeTrigger->SetMinDimensions(0, 99);    
            
            THEN( "The OdeTrigger is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A OdeTrigger checks for Maximum Dimensions correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with maximum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        objectMeta.confidence = 0.4999; 

        WHEN( "the Maximum Width is set below the Object's Width" )
        {
            pOdeTrigger->SetMaxDimensions(199, 0);    
            
            THEN( "The OdeTrigger is NOT detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Max Width is set above the Object's Width" )
        {
            pOdeTrigger->SetMaxDimensions(201, 0);    
            
            THEN( "The OdeTrigger is detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Max Height is set below the Object's Height" )
        {
            pOdeTrigger->SetMaxDimensions(0, 99);    
            
            THEN( "The OdeTrigger is NOT detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Max Height is set above the Object's Height" )
        {
            pOdeTrigger->SetMaxDimensions(0, 101);    
            
            THEN( "The OdeTrigger is detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}
SCENARIO( "An OdeTrigger checks its InferDoneOnly setting ", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = false;  // set to false to fail criteria  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "The ODE Type's InferOnOnly setting is enable and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetInferDoneOnlySetting(true);
            
            THEN( "The ODE is NOT triggered because the frame's flage is false" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The ODE Type's InferOnOnly setting is disabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetInferDoneOnlySetting(false);
            
            THEN( "The ODE is triggered because the criteria is not set" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A OdeTrigger checks for Area overlap correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with maximum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);

        std::string odeActionName("ode-action");
        std::string odeAreaName("ode-area");

        std::string rectangleName  = "my-rectangle";
        uint left(12), top(34), width(56), height(78);
        uint borderWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);
        DSL_RGBA_RECTANGLE_PTR pRectangle = DSL_RGBA_RECTANGLE_NEW(rectangleName.c_str(), 
            left, top, width, height, borderWidth, pColor, true, pColor);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        DSL_ODE_AREA_PTR pOdeArea =
            DSL_ODE_AREA_NEW(odeAreaName.c_str(), pRectangle, false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        
        REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Type's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 200;
        objectMeta.rect_params.top = 100;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        objectMeta.confidence = 0.4999; 

        WHEN( "The Area is set so that the Object's top left corner overlaps" )
        {
            pRectangle->left = 0; 
            pRectangle->top = 0; 
            pRectangle->width = 201;
            pRectangle->height = 101;    
            
            THEN( "The OdeTrigger is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area is set so that the Object's top right corner overlaps" )
        {
            pRectangle->left = 400; 
            pRectangle->top = 0; 
            pRectangle->width = 100;
            pRectangle->height = 100;    
            
            THEN( "The OdeTrigger is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area is set so that the Object's bottom left corner overlaps" )
        {
            pRectangle->left = 0; 
            pRectangle->top = 199; 
            pRectangle->width = 201;
            pRectangle->height = 100;    
            
            THEN( "The OdeTrigger is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area is set so that the Object's bottom right corner overlaps" )
        {
            pRectangle->left = 400; 
            pRectangle->top = 200; 
            pRectangle->width = 100;
            pRectangle->height = 100;    
            
            THEN( "The OdeTrigger is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area is set so that the Object does not overlap" )
        {
            pRectangle->left = 0; 
            pRectangle->top = 0; 
            pRectangle->width = 10;
            pRectangle->height = 10;    
            
            THEN( "The OdeTrigger is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An Intersection OdeTrigger checks for intersection correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with minimum criteria" ) 
    {
        std::string odeTriggerName("intersection");
        uint classId(1);
        uint limit(0);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_INTERSECTION_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_INTERSECTION_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Type's classId
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Type's classId
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; // must match ODE Type's classId
        
        WHEN( "Two objects occur without overlap" )
        {
            objectMeta1.rect_params.left = 0;
            objectMeta1.rect_params.top = 0;
            objectMeta1.rect_params.width = 100;
            objectMeta1.rect_params.height = 100;

            objectMeta2.rect_params.left = 101;
            objectMeta2.rect_params.top = 101;
            objectMeta2.rect_params.width = 100;
            objectMeta2.rect_params.height = 100;

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "NO ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "Two objects occur with overlap" )
        {
            objectMeta1.rect_params.left = 0;
            objectMeta1.rect_params.top = 0;
            objectMeta1.rect_params.width = 100;
            objectMeta1.rect_params.height = 100;

            objectMeta2.rect_params.left = 99;
            objectMeta2.rect_params.top = 99;
            objectMeta2.rect_params.width = 100;
            objectMeta2.rect_params.height = 100;

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 1 );
            }
        }
        WHEN( "Three objects occur, one overlaping the other two" )
        {
            objectMeta1.rect_params.left = 0;
            objectMeta1.rect_params.top = 0;
            objectMeta1.rect_params.width = 100;
            objectMeta1.rect_params.height = 100;

            objectMeta2.rect_params.left = 99;
            objectMeta2.rect_params.top = 99;
            objectMeta2.rect_params.width = 100;
            objectMeta2.rect_params.height = 100;

            objectMeta3.rect_params.left = 198;
            objectMeta3.rect_params.top = 0;
            objectMeta3.rect_params.width = 100;
            objectMeta3.rect_params.height = 100;

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta3) == true );
            
            THEN( "Three ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 2 );
            }
        }
    }
}

static boolean ode_check_for_occurrence_cb(void* buffer,
    void* frame_meta, void* object_meta, void* client_data)
{    
    return true;
}

static boolean ode_post_process_frame_cb(void* buffer,
    void* frame_meta, void* client_data)
{    
    return true;
}

SCENARIO( "A Custom OdeTrigger checks for and handles Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new CustomOdeTrigger with client occurrence checker" ) 
    {
        std::string odeTriggerName("custom");
        uint classId(1);
        uint limit(0);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_CUSTOM_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_CUSTOM_NEW(odeTriggerName.c_str(), classId, limit, ode_check_for_occurrence_cb, 
                ode_post_process_frame_cb, NULL);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        WHEN( "Minimum ODE criteria is met" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match ODE Type's classId
            
            objectMeta.rect_params.left = 0;
            objectMeta.rect_params.top = 0;
            objectMeta.rect_params.width = 100;
            objectMeta.rect_params.height = 100;
            
            THEN( "The client's custom CheckForOccurrence is called returning true." )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A MaximumOdeTrigger handle ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new MaximumOdeTrigger with Maximum criteria" ) 
    {
        std::string odeTriggerName("maximum");
        uint classId(1);
        uint limit(0);
        uint maximum(2);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_MAXIMUM_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_MAXIMUM_NEW(odeTriggerName.c_str(), classId, limit, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Type's classId
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Type's classId
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; // must match ODE Type's classId
        
        WHEN( "Two objects occur -- equal to the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "NO ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "Three objects occur -- greater than the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta3) == true );
            
            THEN( "ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 3 );
            }
        }
    }
}

SCENARIO( "A MinimumOdeTrigger handle ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new MaximumOdeTrigger with Maximum criteria" ) 
    {
        std::string odeTriggerName("maximum");
        uint classId(1);
        uint limit(0);
        uint minimum(3);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_MINIMUM_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_MINIMUM_NEW(odeTriggerName.c_str(), classId, limit, minimum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Type's classId
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Type's classId
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; // must match ODE Type's classId
        
        WHEN( "Two objects occur -- less than the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "NO ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 2 );
            }
        }
        WHEN( "Three objects occur -- equal than the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta3) == true );
            
            THEN( "ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "A SmallestOdeTrigger handles an ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new SmallestOdeTrigger" ) 
    {
        std::string odeTriggerName("smallest");
        uint classId(1);
        uint limit(0);

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_SMALLEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_SMALLEST_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Type's classId
        objectMeta1.rect_params.left = 0;
        objectMeta1.rect_params.top = 0;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;

        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Type's classId
        objectMeta2.rect_params.left = 0;
        objectMeta2.rect_params.top = 0;
        objectMeta2.rect_params.width = 99;
        objectMeta2.rect_params.height = 100;
        
        
        WHEN( "Two objects occur" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected with the largets reported" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 1 );
            }
        }
    }
}
SCENARIO( "A LargestOdeTrigger handles am ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new LargestOdeTrigger" ) 
    {
        std::string odeTriggerName("smallest");
        uint classId(1);
        uint limit(0);

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_LARGEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_LARGEST_NEW(odeTriggerName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Type's classId
        objectMeta1.rect_params.left = 0;
        objectMeta1.rect_params.top = 0;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;

        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Type's classId
        objectMeta2.rect_params.left = 0;
        objectMeta2.rect_params.top = 0;
        objectMeta2.rect_params.width = 99;
        objectMeta2.rect_params.height = 100;
        
        WHEN( "Two objects occur" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected with the smallest reported" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, &frameMeta) == 1 );
            }
        }
    }
}
