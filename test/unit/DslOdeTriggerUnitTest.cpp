/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslServices.h"

using namespace DSL;

SCENARIO( "A new OdeOccurreceTrigger is created correctly", "[OdeTrigger]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);
        
        std::string source;

        WHEN( "A new OdeTrigger is created" )
        {
            DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

            THEN( "The OdeTriggers's memebers are setup and returned correctly" )
            {
                REQUIRE( pOdeTrigger->GetEnabled() == true );
                REQUIRE( pOdeTrigger->GetClassId() == classId );
                REQUIRE( pOdeTrigger->GetLimit() == limit );
                REQUIRE( pOdeTrigger->GetSource() == NULL );
                float minWidth(123), minHeight(123);
                pOdeTrigger->GetMinDimensions(&minWidth, &minHeight);
                REQUIRE( minWidth == 0 );
                REQUIRE( minHeight == 0 );
                float maxWidth(123), maxHeight(123);
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

SCENARIO( "An OdeOccurrenceTrigger checks its enabled setting ", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string source;

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Type is disabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetEnabled(false);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An ODE Occurrence Trigger checks its minimum confidence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Type's minimum confidence is equal to the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.5);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Type's minimum confidence is greater tahn the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.5001);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "A OdeOccurrenceTrigger checks for Source Name correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0);
        uint sourceId(1);
        
        std::string source("source-1");
        
        Services::GetServices()->_sourceNameSet(sourceId, source.c_str());

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
            
        std::string retSource(pOdeTrigger->GetSource());
        REQUIRE( retSource == source );

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;

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
            frameMeta.source_id = 1;
            pOdeTrigger->SetSource("");
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            frameMeta.source_id = 1;
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID does not match the filter" )
        {
            frameMeta.source_id = 2;
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "A OdeOccurrenceTrigger checks for Minimum Dimensions correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeOccurrenceTrigger with minimum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Width is set below the Object's Width" )
        {
            pOdeTrigger->SetMinDimensions(199, 0);    
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Min Height is set above the Object's Height" )
        {
            pOdeTrigger->SetMinDimensions(0, 101);    
            
            THEN( "The OdeTrigger is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Height is set below the Object's Height" )
        {
            pOdeTrigger->SetMinDimensions(0, 99);    
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A OdeOccurrenceTrigger checks for Maximum Dimensions correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeOccurrenceTrigger with maximum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Max Width is set above the Object's Width" )
        {
            pOdeTrigger->SetMaxDimensions(201, 0);    
            
            THEN( "The ODE Occurrence is detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Max Height is set below the Object's Height" )
        {
            pOdeTrigger->SetMaxDimensions(0, 99);    
            
            THEN( "The OdeTrigger is NOT detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Max Height is set above the Object's Height" )
        {
            pOdeTrigger->SetMaxDimensions(0, 101);    
            
            THEN( "The ODE Occurrence is detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}
SCENARIO( "An OdeOccurrenceTrigger checks its InferDoneOnly setting ", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The ODE Type's InferOnOnly setting is disabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetInferDoneOnlySetting(false);
            
            THEN( "The ODE is triggered because the criteria is not set" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A OdeOccurrenceTrigger checks for Area overlap correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeOccurenceTrigger with criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string odeActionName("ode-action");
        std::string odeAreaName("ode-area");

        std::string polygonName  = "my-polygon";
        dsl_coordinate coordinates[4] = {{100,100},{200,100},{200, 200},{100,200}};
        uint numCoordinates(4);
        uint lineWidth(4);

        std::string colorName  = "my-custom-color";
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);

        DSL_RGBA_POLYGON_PTR pPolygon = DSL_RGBA_POLYGON_NEW(polygonName.c_str(), 
            coordinates, numCoordinates, lineWidth, pColor);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
        objectMeta.confidence = 0.4999; 

        WHEN( "The Area and Object are set so that the Object's Center Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_CENTER);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 140;
            objectMeta.rect_params.top = 140;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's North West point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_NORTH_WEST);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 190;
            objectMeta.rect_params.top = 190;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's North Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_NORTH);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 140;
            objectMeta.rect_params.top = 190;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's North east point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_NORTH_EAST);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 90;
            objectMeta.rect_params.top = 190;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's East Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_EAST);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 90;
            objectMeta.rect_params.top = 140;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's East Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_SOUTH_EAST);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 90;
            objectMeta.rect_params.top = 90;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's South Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_SOUTH);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 140;
            objectMeta.rect_params.top = 90;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's South West Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_SOUTH_WEST);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 190;
            objectMeta.rect_params.top = 90;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that the Object's West Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_WEST);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 190;
            objectMeta.rect_params.top = 140;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Area and Object are set so that Any Point overlaps" )
        {
            DSL_ODE_AREA_INCLUSION_PTR pOdeArea =
                DSL_ODE_AREA_INCLUSION_NEW(odeAreaName.c_str(), pPolygon, false, DSL_BBOX_POINT_ANY);
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea) == true );
        
            objectMeta.rect_params.left = 110;
            objectMeta.rect_params.top = 110;
            objectMeta.rect_params.width = 20;
            objectMeta.rect_params.height = 20;
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A OdeAbsenceTrigger checks for Source Name correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeAbsenceTrigger with default criteria" ) 
    {
        std::string odeTriggerName("absence");
        uint classId(1);
        uint limit(0);
        uint sourceId(1);
        
        std::string source("source-1");
        
        Services::GetServices()->_sourceNameSet(sourceId, source.c_str());

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_ABSENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_ABSENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
            
        std::string retSource(pOdeTrigger->GetSource());
        REQUIRE( retSource == source );

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;

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
            frameMeta.source_id = 1;
            pOdeTrigger->SetSource("");
            
            THEN( "The ODE is not triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            frameMeta.source_id = 1;
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "The Source ID does not match the filter" )
        {
            frameMeta.source_id = 2;
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == false );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 1 );
            }
        }
    }
}

SCENARIO( "An InstanceOdeTrigger handles ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new InstanceOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("instance");
        std::string source("source-1");
        uint sourceId(1);
        uint classId(1);
        uint limit(0);

        std::string odeActionName("event-action");

        Services::GetServices()->_sourceNameSet(sourceId, source.c_str());

        DSL_ODE_TRIGGER_INSTANCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_INSTANCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = sourceId;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; 
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; 
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; 
        
        WHEN( "Three objects have the same object Id" )
        {
            objectMeta1.object_id = 1; 
            objectMeta2.object_id = 1; 
            objectMeta3.object_id = 1; 

            THEN( "Only the first object triggers ODE occurrence" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == false );
            }
        }
        WHEN( "Three objects have different object Id's" )
        {
            objectMeta1.object_id = 1; 
            objectMeta2.object_id = 2; 
            objectMeta3.object_id = 3; 

            THEN( "Only the first object triggers ODE occurrence" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            }
        }
        WHEN( "Two objects have the same object Id and a third object is difference" )
        {
            objectMeta1.object_id = 1; 
            objectMeta2.object_id = 3; 
            objectMeta3.object_id = 1; 

            THEN( "Only the first and second objects trigger ODE occurrence" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == false );
            }
        }
    }
}

SCENARIO( "An Intersection OdeTrigger checks for intersection correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeIntersectionTrigger with minimum criteria" ) 
    {
        std::string odeTriggerName("intersection");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_INTERSECTION_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_INTERSECTION_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "NO ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
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

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 1 );
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

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            
            THEN( "Three ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 2 );
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
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_CUSTOM_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_CUSTOM_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit, ode_check_for_occurrence_cb, 
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "A CountOdeTrigger handles ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new CountOdeTrigger with Maximum criteria" ) 
    {
        std::string odeTriggerName("maximum");
        std::string source;
        uint classId(1);
        uint limit(0);
        uint minimum(2);
        uint maximum(3);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_COUNT_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_COUNT_NEW(odeTriggerName.c_str(), source.c_str(), 
				classId, limit, minimum, maximum);

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

        NvDsObjectMeta objectMeta4 = {0};
        objectMeta4.class_id = classId; // must match ODE Type's classId
        
        WHEN( "Two objects occur -- equal to the Minimum" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "Two ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 2 );
            }
        }
        WHEN( "One object occurs -- less than the Minimum" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            
            THEN( "0 ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "Three objects occur -- equal to the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            
            THEN( "Three ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 3 );
            }
        }
        WHEN( "Four objects occur -- greater than the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta4) == true );
            
            THEN( "0 ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "A SmallestOdeTrigger handles an ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new SmallestOdeTrigger" ) 
    {
        std::string odeTriggerName("smallest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_SMALLEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_SMALLEST_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected with the largets reported" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 1 );
            }
        }
    }
}
SCENARIO( "A LargestOdeTrigger handles am ODE Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new LargestOdeTrigger" ) 
    {
        std::string odeTriggerName("smallest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_LARGEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_LARGEST_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected with the smallest reported" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 1 );
            }
        }
    }
}

SCENARIO( "A PersistenceOdeTrigger adds/updates tracked objects correctly", "[OdeTrigger]" )
{
    GIVEN( "A new PersistenceOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("persistence");
        std::string source;
        uint classId(1);
        uint limit(0);
        uint minimum(1);
        uint maximum(4);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_PERSISTENCE_NEW(odeTriggerName.c_str(), 
				source.c_str(), classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.ntp_timestamp = INT64_MAX;

		NvDsObjectMeta objectMeta1 = {0};
		objectMeta1.class_id = classId;
		NvDsObjectMeta objectMeta2 = {0};
		objectMeta2.class_id = classId;
		NvDsObjectMeta objectMeta3 = {0};
		objectMeta3.class_id = classId;
        
        WHEN( "Three unique objects, each from a unique source, are provided" )
        {
			frameMeta.frame_num = 1;
			objectMeta1.object_id = 1;
			objectMeta2.object_id = 2;
			objectMeta3.object_id = 3;
            
            THEN( "CheckForOccurrence adds the tracked objects correctly " )
            {
				frameMeta.source_id = 1;
				REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
				frameMeta.source_id = 2;
				REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
				frameMeta.source_id = 3;
				REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            }
        }
        WHEN( "Three object metas are provide for two unique objects" )
        {
			frameMeta.source_id = 2;
			objectMeta1.object_id = 0;
			objectMeta2.object_id = 1;
			objectMeta3.object_id = 1;
            
            THEN( "CheckForOccurrence adds the tracked objects correctly " )
            {
				frameMeta.frame_num = 1;
				REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
				REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
				// new frame 
				frameMeta.frame_num = 2;
				REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            }
        }
    }
}

SCENARIO( "A PersistenceOdeTrigger purges tracked objects correctly", "[OdeTrigger]" )
{
    GIVEN( "A new PersistenceOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("persistence");
        std::string source;
        uint classId(1);
        uint limit(0);
        uint minimum(1);
        uint maximum(4);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_PERSISTENCE_NEW(odeTriggerName.c_str(), 
				source.c_str(), classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.ntp_timestamp = INT64_MAX;

		NvDsObjectMeta objectMeta1 = {0};
		objectMeta1.class_id = classId;
		NvDsObjectMeta objectMeta2 = {0};
		objectMeta2.class_id = classId;
		NvDsObjectMeta objectMeta3 = {0};
		objectMeta3.class_id = classId;
        
        WHEN( "Three unique objects, each from a unique source, are added" )
        {
			frameMeta.frame_num = 1;
			objectMeta1.object_id = 1;
			frameMeta.source_id = 1;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
			objectMeta2.object_id = 2;
			frameMeta.source_id = 2;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
			// new frame
			frameMeta.frame_num = 2;
			objectMeta3.object_id = 3;
			frameMeta.source_id = 3;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame purges the first two objects" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "A PersistenceOdeTrigger Post Processes ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new PersistenceOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("persistence");
        std::string source;
        uint classId(1);
        uint limit(0);
        uint minimum(1);
        uint maximum(3);

        std::string odeActionName("event-action");

        DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_PERSISTENCE_NEW(odeTriggerName.c_str(), 
				source.c_str(), classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.ntp_timestamp = INT64_MAX;
		frameMeta.frame_num = 1;

		NvDsObjectMeta objectMeta1 = {0};
		objectMeta1.class_id = classId;
		objectMeta1.object_id = 1;
		NvDsObjectMeta objectMeta2 = {0};
		objectMeta2.class_id = classId;
		objectMeta2.object_id = 2;
		NvDsObjectMeta objectMeta3 = {0};
		objectMeta3.class_id = classId;
		objectMeta3.object_id = 3;
        
        WHEN( "The objects are tracked for < than the minimum time" )
        {
			frameMeta.source_id = 1;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
			frameMeta.source_id = 2;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
			frameMeta.source_id = 3;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
			
            THEN( "PostProcessFrame returns 0 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "The objects are tracked for > the minimum time and < the maximum time" )
        {
			frameMeta.source_id = 1;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
			frameMeta.source_id = 2;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
			frameMeta.source_id = 3;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			frameMeta.frame_num = 2;
			frameMeta.source_id = 1;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
			frameMeta.source_id = 2;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
			frameMeta.source_id = 3;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 3 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 3 );
            }
        }
        WHEN( "The objects are tracked for > the maximum time" )
        {
			frameMeta.source_id = 1;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
			frameMeta.source_id = 2;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
			frameMeta.source_id = 3;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
			std::this_thread::sleep_for(std::chrono::milliseconds(3000));
			frameMeta.frame_num = 2;
			frameMeta.source_id = 1;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
			frameMeta.source_id = 2;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
			frameMeta.source_id = 3;
			REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 0 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "An NewLowOdeTrigger handles ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new NewLowOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("new-low");
        std::string source("source-1");
        uint sourceId(1);
        uint classId(1);
        uint limit(0);
        uint preset(2);

        std::string odeActionName("event-action");

        Services::GetServices()->_sourceNameSet(sourceId, source.c_str());

        DSL_ODE_TRIGGER_NEW_LOW_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_NEW_LOW_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, preset);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = sourceId;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; 
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; 
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; 
        
        WHEN( "When three objects - i.e. more than the current low count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "When two objects - i.e. equal to the current high count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new low" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "When one object - i.e. less than the current low count - is added" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );

            THEN( "PostProcessFrame returns 1 occurrence of new low" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 1 );
                
                // ensure that new low has taken effect - one object is no longer new low
                pOdeTrigger->PreProcessFrame(NULL, NULL, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
           }
        }
    }
}

SCENARIO( "An NewHighOdeTrigger handles ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new NewHighOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("new-high");
        std::string source("source-1");
        uint sourceId(1);
        uint classId(1);
        uint limit(0);
        uint preset(2);

        std::string odeActionName("event-action");

        Services::GetServices()->_sourceNameSet(sourceId, source.c_str());

        DSL_ODE_TRIGGER_NEW_HIGH_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_NEW_HIGH_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, preset);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = sourceId;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; 
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; 
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; 
        
        WHEN( "When one object - i.e. less than the current high count - is checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "When two objects - i.e. equal to the current high count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
            }
        }
        WHEN( "When three objects - i.e. greater than the current high count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 1 occurrence of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 1 );
                
                // ensure that the new high has taken effect - and three objects are not a new high
                pOdeTrigger->PreProcessFrame(NULL, NULL, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, NULL, &frameMeta, &objectMeta3) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, NULL, &frameMeta) == 0 );
           }
        }
    }
}
