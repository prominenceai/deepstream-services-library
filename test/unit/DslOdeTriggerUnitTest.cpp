/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

static std::vector<NvDsDisplayMeta*> displayMetaData;

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

static void ode_occurrence_handler_cb_1(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
{
    std::cout << "Custom Action callback 1. called\n";
}    

static void ode_occurrence_handler_cb_2(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
{
    std::cout << "Custom Action callback 2. called\n";
}    
static void ode_occurrence_handler_cb_3(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
{
    std::cout << "Custom Action callback 3. called\n";
}    

static void limit_state_change_listener_1(uint event_id, uint limit, void* client_data)
{
    std::cout 
        << "Limit state change listener 1 callback called, event = " 
        << event_id << ", limit = " << limit << "\n";
}

static void limit_state_change_listener_2(uint event_id, uint limit, void* client_data)
{
    std::cout 
        << "Limit state change listener 2 callback called, event = " 
        << event_id << ", limit = " << limit << "\n";
}

static void enabled_state_change_listener_1(boolean enabled, void* client_data)
{
    std::cout 
        << "Enabled State Change listener 1 callback called with enabled = " 
        << enabled << "\n";
}

static void enabled_state_change_listener_2(boolean enabled, void* client_data)
{
    std::cout 
        << "Enabled State Change listener 2 callback called with enabled = " 
        << enabled << "\n";
}

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
                REQUIRE( pOdeTrigger->GetEventLimit() == limit );
                REQUIRE( pOdeTrigger->GetFrameLimit() == 0  );
                REQUIRE( pOdeTrigger->GetSource() == NULL );
                REQUIRE( pOdeTrigger->GetResetTimeout() == 0 );
                REQUIRE( pOdeTrigger->GetInterval() == 0 );
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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "The ODE Trigger is enabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetEnabled(true);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger is disabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetEnabled(false);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger calls all enabled-state-change-listeners on state change", "[OdeTrigger]" )
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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        REQUIRE( pOdeTrigger->AddEnabledStateChangeListener(
            enabled_state_change_listener_1, NULL) == true );

        REQUIRE( pOdeTrigger->AddEnabledStateChangeListener(
            enabled_state_change_listener_2, NULL) == true );
        
        WHEN( "The ODE Trigger is disabled" )
        {
            pOdeTrigger->SetEnabled(false);
            
            THEN( "All client callback functions are called" )
            {
                // requires manual/visual verification at this time.
            }
        }
        WHEN( "The ODE Trigger is enabled" )
        {
            pOdeTrigger->SetEnabled(true);
            
            THEN( "All client callback functions are called" )
            {
                // requires manual/visual verification at this time.
            }
        }
    }
}


SCENARIO( "An OdeOccurrenceTrigger executes its ODE Actions in the correct order ", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger and three print actions" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string source;

        // The unindexed Child map is order alpha-numerically
        std::string odeActionName1("1-action");
        std::string odeActionName2("2-action");
        std::string odeActionName3("3-action");
        
        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        // Three custom actions using the calbacks defined above. 
        DSL_ODE_ACTION_CUSTOM_PTR pOdeAction1 = 
            DSL_ODE_ACTION_CUSTOM_NEW(odeActionName1.c_str(), ode_occurrence_handler_cb_1, NULL);
        DSL_ODE_ACTION_CUSTOM_PTR pOdeAction2 = 
            DSL_ODE_ACTION_CUSTOM_NEW(odeActionName2.c_str(), ode_occurrence_handler_cb_2, NULL);
        DSL_ODE_ACTION_CUSTOM_PTR pOdeAction3 = 
            DSL_ODE_ACTION_CUSTOM_NEW(odeActionName3.c_str(), ode_occurrence_handler_cb_3, NULL);

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "The Three actions are added in a specific order" )
        {
            // The indexed Child map is ordered by add-order - used for execution.
            REQUIRE( pOdeTrigger->AddAction(pOdeAction3) == true );        
            REQUIRE( pOdeTrigger->AddAction(pOdeAction1) == true );        
            REQUIRE( pOdeTrigger->AddAction(pOdeAction2) == true );        
            
            THEN( "The actions are executed in the correct order" )
            {
                // Note: this requires manual/visual confirmation at this time.
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                
                // Remove Action 3 and add back in to change order    
                REQUIRE( pOdeTrigger->RemoveAction(pOdeAction3) == true );        
                REQUIRE( pOdeTrigger->AddAction(pOdeAction3) == true );        
                
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger handles a timed reset on event limit correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1); // one-shot tirgger
        uint reset_timeout(1);

        std::string source;

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        // Ensure correct defaults
        REQUIRE( pOdeTrigger->GetResetTimeout() == 0 );
        REQUIRE( pOdeTrigger->IsResetTimerRunning() == false);
        
        WHEN( "The ODE Trigger's ResetTimeout is first set" )
        {
            // Limit has NOT been reached
            pOdeTrigger->SetResetTimeout(reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pOdeTrigger->GetResetTimeout() == reset_timeout );
                REQUIRE( pOdeTrigger->IsResetTimerRunning() == false);
            }
        }

        WHEN( "The ODE Trigger's ResetTimeout is set when limit has been reached" )
        {
            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta) == true );

            // Limit has been reached
            pOdeTrigger->SetResetTimeout(reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pOdeTrigger->GetResetTimeout() == reset_timeout );
                REQUIRE( pOdeTrigger->IsResetTimerRunning() == true);
            }
        }
        WHEN( "The ODE Trigger's ResetTimeout is set when the timer is running" )
        {
            // Timeout is set before limit is reached
            pOdeTrigger->SetResetTimeout(reset_timeout);

            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta) == true );

            // Timer must now be running
            REQUIRE( pOdeTrigger->IsResetTimerRunning() == true);

            uint new_reset_timeout(5);
            
            // Timeout is set before limit is reached
            pOdeTrigger->SetResetTimeout(new_reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pOdeTrigger->GetResetTimeout() == new_reset_timeout );
                REQUIRE( pOdeTrigger->IsResetTimerRunning() == true);
            }
        }
        WHEN( "The ODE Trigger's ResetTimeout is cleared when the timer is running" )
        {
            // Timeout is set before limit is reached
            pOdeTrigger->SetResetTimeout(reset_timeout);

            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta) == true );

            // Timer must now be running
            REQUIRE( pOdeTrigger->IsResetTimerRunning() == true);

            uint new_reset_timeout(0);
            
            // Timeout is set before limit is reached
            pOdeTrigger->SetResetTimeout(new_reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pOdeTrigger->GetResetTimeout() == new_reset_timeout );
                REQUIRE( pOdeTrigger->IsResetTimerRunning() == false);
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger handles a timed reset on frame limit correctly", "[temp]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(DSL_ODE_TRIGGER_LIMIT_NONE); 
        uint reset_timeout(1);

        std::string source;

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
            
        // Setting a frame limit of one.
        pOdeTrigger->SetFrameLimit(1);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        // Ensure correct defaults
        REQUIRE( pOdeTrigger->GetResetTimeout() == 0 );
        REQUIRE( pOdeTrigger->IsResetTimerRunning() == false);
        
        WHEN( "The ODE Trigger's ResetTimeout is set when frame limit has been reached" )
        {
            // First occurrence will reach the Trigger's limit of one
            pOdeTrigger->PreProcessFrame(NULL, 
                displayMetaData, &frameMeta);

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta) == true );

            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                displayMetaData, &frameMeta) == 1 );

            // Limit has been reached
            pOdeTrigger->SetResetTimeout(reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pOdeTrigger->GetResetTimeout() == reset_timeout );
                REQUIRE( pOdeTrigger->IsResetTimerRunning() == true);
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger notifies its limit-state-listeners", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1); // one-shot tirgger
        uint reset_timeout(1);

        std::string source;

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;

        REQUIRE( pOdeTrigger->AddLimitStateChangeListener(
            limit_state_change_listener_1, NULL) == true );

        REQUIRE( pOdeTrigger->AddLimitStateChangeListener(
            limit_state_change_listener_2, NULL) == true );
        
        WHEN( "When an ODE occures and the Trigger reaches its limit" )
        {
            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta) == true );

            THEN( "All client listeners are notified" )
            {
                // NOTE requires manual confirmation at this time.
                
                pOdeTrigger->Reset();
                
                REQUIRE( pOdeTrigger->RemoveLimitStateChangeListener(
                    limit_state_change_listener_1) == true );

                REQUIRE( pOdeTrigger->RemoveLimitStateChangeListener(
                    limit_state_change_listener_2) == true );
            }
        }
    }
}

SCENARIO( "An ODE Occurrence Trigger checks its minimum inference confidence correctly", 
    "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.5;
        
        WHEN( "The ODE Trigger's minimum confidence is less than the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.4999);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's minimum confidence is equal to the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.5);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's minimum confidence is greater tahn the Object's confidence" )
        {
            pOdeTrigger->SetMinConfidence(0.5001);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An ODE Occurrence Trigger checks its maximum inference confidence correctly", 
    "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.5;
        
        WHEN( "The ODE Trigger's maximum confidence is less than the Object's confidence" )
        {
            pOdeTrigger->SetMaxConfidence(0.4999);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The ODE Trigger's maximum confidence is equal to the Object's confidence" )
        {
            pOdeTrigger->SetMaxConfidence(0.5);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's maximum confidence is greater than the Object's confidence" )
        {
            pOdeTrigger->SetMaxConfidence(0.5001);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An ODE Occurrence Trigger checks its minimum tracker confidence correctly", 
    "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.tracker_confidence = 0.5;
        
        WHEN( "The ODE Trigger's minimum confidence is less than the Object's confidence" )
        {
            pOdeTrigger->SetMinTrackerConfidence(0.4999);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's minimum confidence is equal to the Object's confidence" )
        {
            pOdeTrigger->SetMinTrackerConfidence(0.5);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's minimum confidence is greater tahn the Object's confidence" )
        {
            pOdeTrigger->SetMinTrackerConfidence(0.5001);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An ODE Occurrence Trigger checks its maximum tracker confidence correctly", 
    "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.tracker_confidence = 0.5;
        
        WHEN( "The ODE Trigger's maximum confidence is less than the Object's confidence" )
        {
            pOdeTrigger->SetMaxTrackerConfidence(0.4999);
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The ODE Trigger's maximum confidence is equal to the Object's confidence" )
        {
            pOdeTrigger->SetMaxTrackerConfidence(0.5);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's maximum confidence is greater tahn the Object's confidence" )
        {
            pOdeTrigger->SetMaxTrackerConfidence(0.5001);
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks for Source Name correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0);
        
        std::string source("source-1");
        
        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
            
        std::string retSource(pOdeTrigger->GetSource());
        REQUIRE( retSource == source );

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.9999; 
        
        WHEN( "The the Source ID filter is disabled" )
        {
            frameMeta.source_id = sourceId;
            pOdeTrigger->SetSource("");
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            frameMeta.source_id = sourceId;
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "The Source ID does not match the filter" )
        {
            frameMeta.source_id = 99;
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks for Infer Name/Id correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with default criteria" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0);
        uint inferId(12345);
        
        std::string inferName("infer-1");
        
        Services::GetServices()->_inferNameSet(inferId, inferName.c_str());

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), "", classId, limit);
            
        REQUIRE( pOdeTrigger->GetInfer() == NULL );
        
        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.9999; 
        
        WHEN( "The the Source ID filter is disabled" )
        {
            objectMeta.unique_component_id = 1;
            pOdeTrigger->SetInfer("");
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            objectMeta.unique_component_id = inferId;
            pOdeTrigger->SetInfer(inferName.c_str());
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Source ID does not match the filter" )
        {
            objectMeta.unique_component_id = 1;
            pOdeTrigger->SetInfer(inferName.c_str());
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks for Minimum Dimensions correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeOccurrenceTrigger with minimum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Width is set below the Object's Width" )
        {
            pOdeTrigger->SetMinDimensions(199, 0);    
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Min Height is set above the Object's Height" )
        {
            pOdeTrigger->SetMinDimensions(0, 101);    
            
            THEN( "The OdeTrigger is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Min Height is set below the Object's Height" )
        {
            pOdeTrigger->SetMinDimensions(0, 99);    
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks for Maximum Dimensions correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeOccurrenceTrigger with maximum criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Max Width is set above the Object's Width" )
        {
            pOdeTrigger->SetMaxDimensions(201, 0);    
            
            THEN( "The ODE Occurrence is detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Max Height is set below the Object's Height" )
        {
            pOdeTrigger->SetMaxDimensions(0, 99);    
            
            THEN( "The OdeTrigger is NOT detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The Max Height is set above the Object's Height" )
        {
            pOdeTrigger->SetMaxDimensions(0, 101);    
            
            THEN( "The ODE Occurrence is detected because of the maximum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = false;  // set to false to fail criteria  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "The ODE Trigger's InferOnOnly setting is enable and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetInferDoneOnlySetting(true);
            
            THEN( "The ODE is NOT triggered because the frame's flage is false" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The ODE Trigger's InferOnOnly setting is disabled and an ODE occurrence is simulated" )
        {
            pOdeTrigger->SetInferDoneOnlySetting(false);
            
            THEN( "The ODE is triggered because the criteria is not set" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks its interval setting ", "[OdeTrigger]" )
{
    GIVEN( "A new OdeTrigger with a non-zero skip-frame interval" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string source;

        std::string odeActionName("print-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        WHEN( "The ODE Trigger's skip-interval is set" )
        {
            pOdeTrigger->SetInterval(2);
            
            THEN( "Then ODE occurrence is triggered correctly" )
            {
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The ODE Trigger's skip-interval is updated" )
        {
            pOdeTrigger->SetInterval(3);
            
            THEN( "Then ODE occurrence is triggered correctly" )
            {
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
        WHEN( "The ODE Trigger's skip-interval is disabled" )
        {
            pOdeTrigger->SetInterval(0);
            
            THEN( "The ODE is NOT triggered" )
            {
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                pOdeTrigger->PreProcessFrame(NULL, 
                    displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks for Area overlap correctly", "[OdeTrigger]" )
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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);

        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
    }
}

SCENARIO( "An OdeOccurrenceTrigger checks its Areas in the correct order", "[OdeTrigger]" )
{
    GIVEN( "A new OdeOccurenceTrigger with criteria" ) 
    {
        std::string odeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string odeActionName("ode-action");
        std::string odeAreaName1("1-ode-area");
        std::string odeAreaName2("2-ode-area");

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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);

        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );     

        DSL_ODE_AREA_INCLUSION_PTR pOdeArea1 =
            DSL_ODE_AREA_INCLUSION_NEW(odeAreaName1.c_str(), pPolygon, false, DSL_BBOX_POINT_CENTER);
        DSL_ODE_AREA_EXCLUSION_PTR pOdeArea2 =
            DSL_ODE_AREA_EXCLUSION_NEW(odeAreaName2.c_str(), pPolygon, false, DSL_BBOX_POINT_CENTER);

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.confidence = 0.4999; 

        // The P and Object are set so that the Object's Center Point overlaps
        objectMeta.rect_params.left = 140;
        objectMeta.rect_params.top = 140;
        objectMeta.rect_params.width = 20;
        objectMeta.rect_params.height = 20;

        WHEN( "The Inclusion Area is added first" )
        {
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea1) == true );
            REQUIRE( pOdeTrigger->AddArea(pOdeArea2) == true );
            
            THEN( "The ODE Occurrence is detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL,  
                    displayMetaData, &frameMeta, &objectMeta) == true );
            }
        }
        WHEN( "The Exclusion Area is added first" )
        {
                
            REQUIRE( pOdeTrigger->AddArea(pOdeArea2) == true );
            REQUIRE( pOdeTrigger->AddArea(pOdeArea1) == true );
            
            THEN( "The ODE Occurrence is NOT detected because of the minimum criteria" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
            }
        }
    }
}

SCENARIO( "An OdeAbsenceTrigger checks for Source Name correctly", "[OdeTrigger]" )
{
    GIVEN( "A new OdeAbsenceTrigger with default criteria" ) 
    {
        std::string odeTriggerName("absence");
        uint classId(1);
        uint limit(0);
        
        std::string source("source-1");
        
        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_ABSENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_ABSENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
            
        std::string retSource(pOdeTrigger->GetSource());
        REQUIRE( retSource == source );

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match ODE Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.confidence = 0.9999; 
        
        WHEN( "The the Source ID filter is disabled" )
        {
            frameMeta.source_id = sourceId;
            pOdeTrigger->SetSource("");
            
            THEN( "The ODE is not triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "The Source ID matches the filter" )
        {
            frameMeta.source_id = sourceId;
            
            THEN( "The ODE is triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "The Source ID does not match the filter" )
        {
            frameMeta.source_id = 99;
            
            THEN( "The ODE is NOT triggered" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == false );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 1 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
    }
}

//SCENARIO( "An AccumulationOdeTrigger handles ODE Occurrences correctly", "[OdeTrigger]" )
//{
//    GIVEN( "A new AccumulationOdeTrigger with specific Class Id and Source Id criteria" ) 
//    {
//        std::string odeTriggerName("accumulation");
//        std::string source("source-1");
//        uint classId(1);
//        uint limit(0);
//
//        std::string odeActionName("action");
//
//        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());
//
//        DSL_ODE_TRIGGER_ACCUMULATION_PTR pOdeTrigger = 
//            DSL_ODE_TRIGGER_ACCUMULATION_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
//
//        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
//            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
//            
//        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        
//
//        NvDsFrameMeta frameMeta =  {0};
//        frameMeta.frame_num = 444;
//        frameMeta.ntp_timestamp = INT64_MAX;
//        frameMeta.source_id = sourceId;
//
//        NvDsObjectMeta objectMeta1 = {0};
//        objectMeta1.class_id = classId; 
//        
//        NvDsObjectMeta objectMeta2 = {0};
//        objectMeta2.class_id = classId; 
//        
//        NvDsObjectMeta objectMeta3 = {0};
//        objectMeta3.class_id = classId; 
//        
//        WHEN( "Three objects have the same object Id" )
//        {
//            objectMeta1.object_id = 1; 
//            objectMeta2.object_id = 1; 
//            objectMeta3.object_id = 1; 
//
//            THEN( "Only the first object triggers ODE occurrence" )
//            {
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta1) == true );
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta2) == false );
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta3) == false );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//        WHEN( "Three objects have different object Id's" )
//        {
//            objectMeta1.object_id = 1; 
//            objectMeta2.object_id = 2; 
//            objectMeta3.object_id = 3; 
//
//            THEN( "All three object triggers ODE occurrence" )
//            {
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta1) == true );
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta2) == true );
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta3) == true );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//        WHEN( "Two objects have the same object Id and a third object is difference" )
//        {
//            objectMeta1.object_id = 1; 
//            objectMeta2.object_id = 3; 
//            objectMeta3.object_id = 1; 
//
//            THEN( "Only the first and second objects trigger ODE occurrence" )
//            {
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta1) == true );
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta2) == true );
//                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                    displayMetaData, &frameMeta, &objectMeta3) == false );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//    }
//}
//
//SCENARIO( "An AccumulationOdeTrigger accumulates ODE Occurrences correctly", "[OdeTrigger]" )
//{
//    GIVEN( "A new AccumulationOdeTrigger with specific Class Id and Source Id criteria" ) 
//    {
//        std::string odeTriggerName("accumulation");
//        std::string source("source-1");
//        uint classId(1);
//        uint limit(0);
//
//        std::string odeActionName("action");
//
//        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());
//
//        DSL_ODE_TRIGGER_ACCUMULATION_PTR pOdeTrigger = 
//            DSL_ODE_TRIGGER_ACCUMULATION_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
//
//        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
//            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
//            
//        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        
//
//        NvDsFrameMeta frameMeta =  {0};
//        frameMeta.ntp_timestamp = INT64_MAX;
//        frameMeta.source_id = sourceId;
//
//        NvDsObjectMeta objectMeta1 = {0};
//        objectMeta1.class_id = classId; 
//        
//        NvDsObjectMeta objectMeta2 = {0};
//        objectMeta2.class_id = classId; 
//        
//        NvDsObjectMeta objectMeta3 = {0};
//        objectMeta3.class_id = classId; 
//
//        frameMeta.frame_num = 1;
//        objectMeta1.object_id = 1; 
//        objectMeta2.object_id = 2; 
//        objectMeta3.object_id = 3; 
//        REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//            displayMetaData, &frameMeta, &objectMeta1) == true );
//        REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//            displayMetaData, &frameMeta, &objectMeta2) == true );
//        REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//            displayMetaData, &frameMeta, &objectMeta3) == true );
//
//        REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
//            displayMetaData, &frameMeta) == 3 );
//        
//        WHEN( "The same 3 objects are in the next frame" )
//        {
//            frameMeta.frame_num = 2;
//
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta1) == false );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta2) == false );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta3) == false );
//
//            THEN( "The accumulation count is unchanged" )
//            {
//                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
//                    displayMetaData, &frameMeta) == 3 );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//        WHEN( "Only 1 object is new in the next frame" )
//        {
//            frameMeta.frame_num = 2;
//            objectMeta3.object_id = 4; 
//
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta1) == false );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL,
//                displayMetaData, &frameMeta, &objectMeta2) == false );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta3) == true );
//
//            THEN( "The accumulation count is updated correctly" )
//            {
//                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
//                    displayMetaData, &frameMeta) == 4 );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//        WHEN( "All 3 objects in the next frame are new" )
//        {
//            frameMeta.frame_num = 3;
//            objectMeta1.object_id = 5; 
//            objectMeta2.object_id = 6; 
//            objectMeta3.object_id = 7; 
//
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta1) == true );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta2) == true );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta3) == true );
//
//            THEN( "The accumulation count is updated correctly" )
//            {
//                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
//                    displayMetaData, &frameMeta) == 6 );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//    }
//}
//
//SCENARIO( "An AccumulationOdeTrigger clears its count on Reset", "[OdeTrigger]" )
//{
//    GIVEN( "A new AccumulationOdeTrigger with specific Class Id and Source Id criteria" ) 
//    {
//        std::string odeTriggerName("accumulation");
//        std::string source("source-1");
//        uint classId(1);
//        uint limit(0);
//
//        std::string odeActionName("action");
//
//        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());
//
//        DSL_ODE_TRIGGER_ACCUMULATION_PTR pOdeTrigger = 
//            DSL_ODE_TRIGGER_ACCUMULATION_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);
//
//        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
//            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
//            
//        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        
//
//        NvDsFrameMeta frameMeta =  {0};
//        frameMeta.ntp_timestamp = INT64_MAX;
//        frameMeta.source_id = sourceId;
//
//        NvDsObjectMeta objectMeta1 = {0};
//        objectMeta1.class_id = classId; 
//        
//        NvDsObjectMeta objectMeta2 = {0};
//        objectMeta2.class_id = classId; 
//        
//        NvDsObjectMeta objectMeta3 = {0};
//        objectMeta3.class_id = classId; 
//
//        frameMeta.frame_num = 1;
//        objectMeta1.object_id = 1; 
//        objectMeta2.object_id = 2; 
//        objectMeta3.object_id = 3; 
//        REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//            displayMetaData, &frameMeta, &objectMeta1) == true );
//        REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//            displayMetaData, &frameMeta, &objectMeta2) == true );
//        REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//            displayMetaData, &frameMeta, &objectMeta3) == true );
//
//        REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
//            displayMetaData, &frameMeta) == 3 );
//        
//        WHEN( "The same 3 objects are in the next frame after reset" )
//        {
//            frameMeta.frame_num = 2;
//
//            pOdeTrigger->Reset();
//
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta1) == true );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta2) == true );
//            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
//                displayMetaData, &frameMeta, &objectMeta3) == true );
//
//            THEN( "The accumulation count has restarted from 0" )
//            {
//                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
//                    displayMetaData, &frameMeta) == 3 );
//                Services::GetServices()->_sourceNameErase(source.c_str());
//            }
//        }
//    }
//}
//

SCENARIO( "An InstanceOdeTrigger is created correctly", "[OdeTrigger]" )
{
    GIVEN( "A new InstanceOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("instance");
        std::string source("source-1");
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        WHEN( "The InstanceOdeTrigger is created" )
        {
            DSL_ODE_TRIGGER_INSTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_INSTANCE_NEW(odeTriggerName.c_str(), 
                    source.c_str(), classId, limit);
                    
            THEN( "The correct settings are returned on get" )
            {
                uint instanceCount(99), suppressionCount(99);
                
                pOdeTrigger->GetCountSettings(&instanceCount, &suppressionCount);
                    
                REQUIRE( instanceCount == 1 );
                REQUIRE( suppressionCount == 0 );
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
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        DSL_ODE_TRIGGER_INSTANCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_INSTANCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
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

            THEN( "Only the first object triggers ODE occurrence" )
            {
                frameMeta.frame_num = 1;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                    
                frameMeta.frame_num = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );

                frameMeta.frame_num = 3;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "Three objects have different object Id's" )
        {
            objectMeta1.object_id = 1; 
            objectMeta2.object_id = 2; 
            objectMeta3.object_id = 3; 

            THEN( "All three objects trigger ODE occurrence" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "Two objects have the same object Id and a third object is difference" )
        {
            objectMeta1.object_id = 1; 
            objectMeta2.object_id = 3; 
            objectMeta3.object_id = 1; 

            THEN( "Only the first and second objects trigger ODE occurrence" )
            {
                frameMeta.frame_num = 1;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );

                frameMeta.frame_num = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == false );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "Instance count and suppression count are set" )
        {
            pOdeTrigger->SetCountSettings(3, 2);
            objectMeta1.object_id = 1; 

            THEN( "Occurrences are trigger correctly" )
            {
                frameMeta.frame_num = 1;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );

                frameMeta.frame_num = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );

                frameMeta.frame_num = 3;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );

                frameMeta.frame_num = 4;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );

                frameMeta.frame_num = 5;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );

                frameMeta.frame_num = 6;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );

                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
    }
}

SCENARIO( "An InstanceOdeTrigger Accumulates ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new InstanceOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("instance");
        std::string source("source-1");
        uint classId(1);
        uint limit(0);

        std::string odeAccumulatorName("accumulator-name");
        std::string odeActionName("print-action");

        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        DSL_ODE_TRIGGER_INSTANCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_INSTANCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        DSL_ODE_ACCUMULATOR_PTR pOdeAccumulator = 
            DSL_ODE_ACCUMULATOR_NEW(odeAccumulatorName.c_str());
            
        REQUIRE( pOdeAccumulator->AddAction(pOdeAction) == true );        

        REQUIRE( pOdeTrigger->AddAccumulator(pOdeAccumulator) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = sourceId;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; 
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; 
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; 

        WHEN( "Three objects have different object Id's" )
        {
            objectMeta1.object_id = 1; 
            objectMeta2.object_id = 2; 
            objectMeta3.object_id = 3; 
            
            THEN( "Instance Accumulation is handled correctly" )
            {
                frameMeta.frame_num = 1;
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);

                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
                    
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 3 );

                // same object Id's - no new instances
                frameMeta.frame_num = 2;
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);

                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == false );

                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );

                // new object Id's - new instances
                objectMeta1.object_id = 4; 
                objectMeta2.object_id = 5; 
                objectMeta3.object_id = 6; 
                
                frameMeta.frame_num = 3;
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);

                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );

                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 3 );

                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
    }
}

SCENARIO( "A CrossOdeTrigger handles ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new CrossOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("cross-trigger");
        std::string source("source-1");
        uint classId(1);
        uint limit(0);
        uint minFrameCount(0);
        uint maxTracePoints(10);

        std::string colorName("black");
        std::string lineName("line");
        std::string odeAreaName("line-area");
        std::string odeActionName("print-action");

        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        DSL_RGBA_PREDEFINED_COLOR_PTR pBlack = 
            DSL_RGBA_PREDEFINED_COLOR_NEW(colorName.c_str(), 
                DSL_COLOR_PREDEFINED_BLACK, 1.0);

        DSL_ODE_TRIGGER_CROSS_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_CROSS_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, minFrameCount, maxTracePoints,
                DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS, pBlack);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        
                
        DSL_RGBA_LINE_PTR pLine = 
            DSL_RGBA_LINE_NEW(lineName.c_str(), 10,200,1000,200, 2, pBlack);
            
        DSL_ODE_AREA_LINE_PTR pOdeLineArea = 
            DSL_ODE_AREA_LINE_NEW(lineName.c_str(), pLine, true, 
                DSL_BBOX_POINT_SOUTH);

        REQUIRE( pOdeTrigger->AddArea(pOdeLineArea) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = sourceId;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; 
        objectMeta1.object_id = 1; 
        objectMeta1.rect_params.left = 100;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; 
        objectMeta2.object_id = 2; 
        objectMeta2.rect_params.left = 100;
        objectMeta2.rect_params.width = 100;
        objectMeta2.rect_params.height = 100;
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; 
        objectMeta3.object_id = 3; 
        objectMeta3.rect_params.left = 100;
        objectMeta3.rect_params.width = 100;
        objectMeta3.rect_params.height = 100;
        
        WHEN( "The objects start out above the line" )
        {
            objectMeta1.rect_params.top = 10;
            objectMeta2.rect_params.top = 10;
            objectMeta3.rect_params.top = 10;

            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            // first call will start the tracking for each
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == false );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == false );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == false );

            THEN( "Direction is reported as IN" )
            {
                // require manual/visual confirmation
                frameMeta.frame_num = 2;
                objectMeta1.rect_params.top = 400;
                objectMeta2.rect_params.top = 400;
                objectMeta3.rect_params.top = 400;
                
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );

                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "The objects start out below the line" )
        {
            objectMeta1.rect_params.top = 400;
            objectMeta2.rect_params.top = 400;
            objectMeta3.rect_params.top = 400;

            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            // first call will start the tracking for each
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == false );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == false );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == false );

            THEN( "Direction is reported as OUT" )
            {
                // require manual/visual confirmation
                frameMeta.frame_num = 2;
                objectMeta1.rect_params.top = 10;
                objectMeta2.rect_params.top = 10;
                objectMeta3.rect_params.top = 10;
                
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );

                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
    }
}

SCENARIO( "A CrossOdeTrigger Accumulates ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new CrossOdeTrigger with specific Class Id and Source Id criteria" ) 
    {
        std::string odeTriggerName("cross-trigger");
        std::string source("source-1");
        uint classId(1);
        uint limit(0);
        uint minFrameCount(0);
        uint maxTracePoints(10);

        std::string colorName("black");
        std::string lineName("line");
        std::string odeAreaName("line-area");
        std::string odeActionName("print-action");

        std::string odeAccumulatorName("accumulator-name");

        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        DSL_RGBA_PREDEFINED_COLOR_PTR pBlack = 
            DSL_RGBA_PREDEFINED_COLOR_NEW(colorName.c_str(), 
                DSL_COLOR_PREDEFINED_BLACK, 1.0);

        DSL_ODE_TRIGGER_CROSS_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_CROSS_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, minFrameCount, maxTracePoints,
                DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS, pBlack);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        DSL_RGBA_LINE_PTR pLine = 
            DSL_RGBA_LINE_NEW(lineName.c_str(), 10,200,1000,200, 2, pBlack);
            
        DSL_ODE_AREA_LINE_PTR pOdeLineArea = 
            DSL_ODE_AREA_LINE_NEW(lineName.c_str(), pLine, true, 
                DSL_BBOX_POINT_SOUTH);

        REQUIRE( pOdeTrigger->AddArea(pOdeLineArea) == true );      
  
        DSL_ODE_ACCUMULATOR_PTR pOdeAccumulator = 
            DSL_ODE_ACCUMULATOR_NEW(odeAccumulatorName.c_str());
            
        REQUIRE( pOdeAccumulator->AddAction(pOdeAction) == true );        

        REQUIRE( pOdeTrigger->AddAccumulator(pOdeAccumulator) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = sourceId;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; 
        objectMeta1.object_id = 1; 
        objectMeta1.rect_params.left = 100;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; 
        objectMeta2.object_id = 2; 
        objectMeta2.rect_params.left = 100;
        objectMeta2.rect_params.width = 100;
        objectMeta2.rect_params.height = 100;
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; 
        objectMeta3.object_id = 3; 
        objectMeta3.rect_params.left = 100;
        objectMeta3.rect_params.width = 100;
        objectMeta3.rect_params.height = 100;
        
        WHEN( "The objects cross back and forth over the line" )
        {
            THEN( "Accumulative direction occurrences are calculated correctly" )
            {

                objectMeta1.rect_params.top = 10;
                objectMeta2.rect_params.top = 10;
                objectMeta3.rect_params.top = 10;

                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                // first call will start the tracking for each
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == false );
                pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta);
                
                // require manual/visual confirmation
                frameMeta.frame_num = 2;
                objectMeta1.rect_params.top = 400;
                objectMeta2.rect_params.top = 400;
                objectMeta3.rect_params.top = 400;
                
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
                pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta);

                // require manual/visual confirmation
                frameMeta.frame_num = 3;
                
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == false );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == false );
                pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta);

                // require manual/visual confirmation
                frameMeta.frame_num = 4;
                objectMeta1.rect_params.top = 10;
                objectMeta2.rect_params.top = 10;
                objectMeta3.rect_params.top = 10;
                
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
                pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta);

                Services::GetServices()->_sourceNameErase(source.c_str());
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
        uint classIdA(1);
        uint classIdB(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_INTERSECTION_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_INTERSECTION_NEW(odeTriggerName.c_str(), 
                source.c_str(), classIdA, classIdB, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classIdA; // must match ODE Trigger's classId
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classIdA; // must match ODE Trigger's classId
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classIdA; // must match ODE Trigger's classId
        
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

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            
            THEN( "NO ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
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

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 1 );
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

            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "Three ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 2 );
            }
        }
    }
}


SCENARIO( "A Custom OdeTrigger checks for and handles Occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "A new CustomOdeTrigger with client occurrence checker" ) 
    {
        std::string odeTriggerName("custom");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_CUSTOM_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_CUSTOM_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, ode_check_for_occurrence_cb, 
                ode_post_process_frame_cb, NULL);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        WHEN( "Minimum ODE criteria is met" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match ODE Trigger's classId
            
            objectMeta.rect_params.left = 0;
            objectMeta.rect_params.top = 0;
            objectMeta.rect_params.width = 100;
            objectMeta.rect_params.height = 100;
            
            THEN( "The client's custom CheckForOccurrence is called returning true." )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta) == true );
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

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_COUNT_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_COUNT_NEW(odeTriggerName.c_str(), source.c_str(), 
                classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Trigger's classId
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Trigger's classId
        
        NvDsObjectMeta objectMeta3 = {0};
        objectMeta3.class_id = classId; // must match ODE Trigger's classId

        NvDsObjectMeta objectMeta4 = {0};
        objectMeta4.class_id = classId; // must match ODE Trigger's classId
        
        WHEN( "Two objects occur -- equal to the Minimum" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            
            THEN( "Two ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 2 );
            }
        }
        WHEN( "One object occurs -- less than the Minimum" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            
            THEN( "0 ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "Three objects occur -- equal to the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "Three ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 3 );
            }
        }
        WHEN( "Four objects occur -- greater than the Maximum " )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta4) == true );
            
            THEN( "0 ODE occurrences are detected" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Trigger's classId
        objectMeta1.rect_params.left = 0;
        objectMeta1.rect_params.top = 0;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;

        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Trigger's classId
        objectMeta2.rect_params.left = 0;
        objectMeta2.rect_params.top = 0;
        objectMeta2.rect_params.width = 99;
        objectMeta2.rect_params.height = 100;
        
        
        WHEN( "Two objects occur" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected with the largets reported" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 1 );
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
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
        REQUIRE( pOdeTrigger->AddAction(pOdeAction) == true );        

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classId; // must match ODE Trigger's classId
        objectMeta1.rect_params.left = 0;
        objectMeta1.rect_params.top = 0;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;

        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classId; // must match ODE Trigger's classId
        objectMeta2.rect_params.left = 0;
        objectMeta2.rect_params.top = 0;
        objectMeta2.rect_params.width = 99;
        objectMeta2.rect_params.height = 100;
        
        WHEN( "Two objects occur" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            
            THEN( "An ODE occurrence is detected with the smallest reported" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 1 );
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

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_PERSISTENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                frameMeta.source_id = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                frameMeta.source_id = 3;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                // new frame 
                frameMeta.frame_num = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
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

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_PERSISTENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            objectMeta2.object_id = 2;
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            // new frame
            frameMeta.frame_num = 2;
            objectMeta3.object_id = 3;
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame purges the first two objects" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
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

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_PERSISTENCE_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, minimum, maximum);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame returns 0 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The objects are tracked for > the minimum time and < the maximum time" )
        {
            frameMeta.source_id = 1;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            frameMeta.frame_num = 2;
            frameMeta.source_id = 1;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL,
                displayMetaData, &frameMeta, &objectMeta2) == true );
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 3 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 3 );
            }
        }
        WHEN( "The objects are tracked for > the maximum time" )
        {
            frameMeta.source_id = 1;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL,
                displayMetaData, &frameMeta, &objectMeta2) == true );
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
            frameMeta.frame_num = 2;
            frameMeta.source_id = 1;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 0 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "A LatestOdeTrigger adds/updates tracked objects correctly", "[OdeTrigger]" )
{
    GIVEN( "A new LatestOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("latest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_LATEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_LATEST_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                frameMeta.source_id = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                frameMeta.source_id = 3;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta2) == true );
                // new frame 
                frameMeta.frame_num = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                    displayMetaData, &frameMeta, &objectMeta3) == true );
            }
        }
    }
}

SCENARIO( "A LatestOdeTrigger purges tracked objects correctly", "[OdeTrigger]" )
{
    GIVEN( "A new LatestOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("latest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_LATEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_LATEST_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            objectMeta2.object_id = 2;
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            // new frame
            frameMeta.frame_num = 2;
            objectMeta3.object_id = 3;
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame purges the first two objects" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "A LatestOdeTrigger Post Processes ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new LatestOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("latest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_LATEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_LATEST_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
        
        WHEN( "The objects are tracked for for only one frame" )
        {
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame returns 0 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The objects are tracked for two frames" )
        {
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                displayMetaData, &frameMeta) == 0 );

            frameMeta.frame_num = 2;
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 3 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                    displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "when only one object is tracked for three frames" )
        {
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                displayMetaData, &frameMeta) == 0 );

            frameMeta.frame_num = 2;
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, 
                displayMetaData, &frameMeta) == 1 );

            frameMeta.frame_num = 3;
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, 
                displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 1 occurrences" )
            {
                // NOTE: need to manually check the console output to see that the
                // correct Object (source_id=3) is reported
                
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
    }
}

SCENARIO( "A EarliestOdeTrigger adds/updates tracked objects correctly", "[OdeTrigger]" )
{
    GIVEN( "A new EarliestOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("earliest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_EARLIEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_EARLIEST_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                frameMeta.source_id = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                frameMeta.source_id = 3;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
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
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                // new frame 
                frameMeta.frame_num = 2;
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
            }
        }
    }
}

SCENARIO( "A EarliestOdeTrigger purges tracked objects correctly", "[OdeTrigger]" )
{
    GIVEN( "A new EarliesOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("earliest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_EARLIEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_EARLIEST_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            objectMeta2.object_id = 2;
            frameMeta.source_id = 2;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            // new frame
            frameMeta.frame_num = 2;
            objectMeta3.object_id = 3;
            frameMeta.source_id = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame purges the first two objects" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
    }
}

SCENARIO( "A EarliestOdeTrigger Post Processes ODE Occurrences correctly", "[OdeTrigger]" )
{
    GIVEN( "A new EarliestOdeTrigger with criteria" ) 
    {
        std::string odeTriggerName("earliest");
        std::string source;
        uint classId(1);
        uint limit(0);

        std::string odeActionName("action");

        DSL_ODE_TRIGGER_LATEST_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_LATEST_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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

        frameMeta.source_id = 1;
        
        WHEN( "The objects are tracked for for only one frame" )
        {
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
            
            THEN( "PostProcessFrame returns 0 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The objects are tracked for two frames" )
        {
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );

            frameMeta.frame_num = 2;
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 1 occurrences" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "when only one object is tracked for three frames" )
        {
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            frameMeta.frame_num = 2;
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
            frameMeta.frame_num = 3;
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 1 occurrences" )
            {
                // NOTE: need to manually check the console output to see that the
                // correct Object (source_id=2) is reported
                
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
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
        uint classId(1);
        uint limit(0);
        uint preset(2);

        std::string odeActionName("action");

        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        DSL_ODE_TRIGGER_NEW_LOW_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_NEW_LOW_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, preset);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "When two objects - i.e. equal to the current high count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new low" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "When one object - i.e. less than the current low count - is added" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );

            THEN( "PostProcessFrame returns 1 occurrence of new low" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
                
                // ensure that new low has taken effect - one object is no longer new low
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
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
        uint classId(1);
        uint limit(0);
        uint preset(2);

        std::string odeActionName("action");

        uint sourceId = Services::GetServices()->_sourceNameSet(source.c_str());

        DSL_ODE_TRIGGER_NEW_HIGH_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_NEW_HIGH_NEW(odeTriggerName.c_str(), 
                source.c_str(), classId, limit, preset);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);
            
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
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "When two objects - i.e. equal to the current high count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );

            THEN( "PostProcessFrame returns 0 occurrences of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
            }
        }
        WHEN( "When three objects - i.e. greater than the current high count - are checked" )
        {
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
            REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );

            THEN( "PostProcessFrame returns 1 occurrence of new high" )
            {
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
                
                // ensure that the new high has taken effect - and three objects are not a new high
                pOdeTrigger->PreProcessFrame(NULL, displayMetaData, &frameMeta);
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta3) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
                Services::GetServices()->_sourceNameErase(source.c_str());
           }
        }
    }
}

SCENARIO( "A new OdeDistanceTrigger is created correctly", "[OdeTrigger]" )
{
    GIVEN( "Attributes for a new OdeDistanceTrigger" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);
        uint classIdA(1);
        uint classIdB(1);
        uint minimum(10);
        uint maximum(20);
        uint testPoint(DSL_BBOX_POINT_ANY);
        uint testMethod(DSL_DISTANCE_METHOD_PERCENT_HEIGHT_B);
        
        
        std::string source;

        WHEN( "A new OdeTrigger is created" )
        {
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    testPoint, testMethod);

            THEN( "The OdeTriggers's members are setup and returned correctly" )
            {
                uint retClassIdA(0), retClassIdB(0);
                REQUIRE( pOdeTrigger->GetEnabled() == true );
                pOdeTrigger->GetClassIdAB(&retClassIdA, &retClassIdB);
                REQUIRE( retClassIdA == classIdA );
                REQUIRE( retClassIdB == classIdB );
                uint retMinimum(0), retMaximum(0);
                pOdeTrigger->GetRange(&retMinimum, &retMaximum);
                REQUIRE( retMinimum == minimum );
                REQUIRE( retMaximum == maximum);
                uint retTestPoint(0), retTestMethod(0);
                pOdeTrigger->GetTestParams(&retTestPoint, &retTestMethod);
                REQUIRE( retTestPoint == testPoint );
                REQUIRE( retTestMethod == testMethod );
                REQUIRE( pOdeTrigger->GetEventLimit() == limit );
                REQUIRE( pOdeTrigger->GetFrameLimit() == 0 );
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

SCENARIO( "A new OdeDistanceTrigger can Set/Get its AB Class Ids", "[OdeTrigger]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(1);
        uint classIdA(1);
        uint classIdB(1);
        uint minimum(10);
        uint maximum(20);
        
        std::string source;

        DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                classIdA, classIdB, limit, minimum, maximum, 
                DSL_BBOX_POINT_ANY, DSL_DISTANCE_METHOD_FIXED_PIXELS);

        WHEN( "The OdeDistanceTrigger's AB Class Ids are Set" )
        {
            uint newClassIdA(5), newClassIdB(8);
            pOdeTrigger->SetClassIdAB(newClassIdA, newClassIdB);
            
            THEN( "The correct values are returned on Get" )
            {
                uint retClassIdA(0), retClassIdB(0);
                pOdeTrigger->GetClassIdAB(&retClassIdA, &retClassIdB);
                REQUIRE( retClassIdA == newClassIdA );
                REQUIRE( retClassIdB == newClassIdB );
            }
        }
    }
}           

SCENARIO( "A new Fixed-Pixel OdeDistanceTrigger handles occurrence correctly", "[OdeTrigger]" )
{
    GIVEN( "Attributes for a new Distance Trigger" ) 
    {
        std::string odeTriggerName("distance");
        uint limit(0);

        std::string odeActionName("action");
        
        std::string source;

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 0;

        WHEN( "A Single object is detected" )
        {
            uint classIdA(1);
            uint classIdB(1);
            uint minimum(10);
            uint maximum(20);

            NvDsObjectMeta objectMeta1 = {0};
            objectMeta1.class_id = classIdA; 

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_ANY, DSL_DISTANCE_METHOD_FIXED_PIXELS);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "A Two objects are detected beyond maximum distance" )
        {
            uint classIdA(1);
            uint classIdB(1);
            uint minimum(10);
            uint maximum(20);

            NvDsObjectMeta objectMeta1 = {0};
            objectMeta1.class_id = classIdA; 
            objectMeta1.rect_params.left = 0;
            objectMeta1.rect_params.top = 0;
            objectMeta1.rect_params.width = 100;
            objectMeta1.rect_params.height = 100;
            
            NvDsObjectMeta objectMeta2 = {0};
            objectMeta2.class_id = classIdB; 
            objectMeta2.rect_params.left = 300;
            objectMeta2.rect_params.top = 0;
            objectMeta2.rect_params.width = 100;
            objectMeta2.rect_params.height = 100;

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_ANY, DSL_DISTANCE_METHOD_FIXED_PIXELS);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "A Two objects are detected within the minimum distance - same Class Ids" )
        {
            uint classIdA(1);
            uint classIdB(1);
            uint minimum(201);
            uint maximum(UINT32_MAX);

            NvDsObjectMeta objectMeta1 = {0};
            objectMeta1.class_id = classIdA; 
            objectMeta1.rect_params.left = 0;
            objectMeta1.rect_params.top = 0;
            objectMeta1.rect_params.width = 100;
            objectMeta1.rect_params.height = 100;
            
            NvDsObjectMeta objectMeta2 = {0};
            objectMeta2.class_id = classIdB; 
            objectMeta2.rect_params.left = 300;
            objectMeta2.rect_params.top = 0;
            objectMeta2.rect_params.width = 100;
            objectMeta2.rect_params.height = 100;

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_ANY, DSL_DISTANCE_METHOD_FIXED_PIXELS);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "A Two objects are detected within the minimum distance - different Class Ids" )
        {
            uint classIdA(1);
            uint classIdB(2);
            uint minimum(201);
            uint maximum(UINT32_MAX);

            NvDsObjectMeta objectMeta1 = {0};
            objectMeta1.class_id = classIdA; 
            objectMeta1.rect_params.left = 0;
            objectMeta1.rect_params.top = 0;
            objectMeta1.rect_params.width = 100;
            objectMeta1.rect_params.height = 100;
            
            NvDsObjectMeta objectMeta2 = {0};
            objectMeta2.class_id = classIdB; 
            objectMeta2.rect_params.left = 300;
            objectMeta2.rect_params.top = 0;
            objectMeta2.rect_params.width = 100;
            objectMeta2.rect_params.height = 100;

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_ANY, DSL_DISTANCE_METHOD_FIXED_PIXELS);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
    }
}

SCENARIO( "A new Ralational OdeDistanceTrigger handles occurrence correctly", "[yup]" )
{
    GIVEN( "Attributes for a new Distance Trigger" ) 
    {
        std::string odeTriggerName("distance");
        uint limit(0);

        std::string odeActionName("action");
        
        std::string source;

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str(), false);

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 0;

        uint classIdA(1);
        uint classIdB(2);

        NvDsObjectMeta objectMeta1 = {0};
        objectMeta1.class_id = classIdA; 
        objectMeta1.rect_params.left = 0;
        objectMeta1.rect_params.top = 0;
        objectMeta1.rect_params.width = 100;
        objectMeta1.rect_params.height = 100;
        
        NvDsObjectMeta objectMeta2 = {0};
        objectMeta2.class_id = classIdB; 
        objectMeta2.rect_params.left = 200;
        objectMeta2.rect_params.top = 0;
        objectMeta2.rect_params.width = 100;
        objectMeta2.rect_params.height = 100;


        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(200); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_SOUTH, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(201); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_SOUTH, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(200); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_NORTH, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(201); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_NORTH, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(200); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_EAST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(201); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_EAST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(200); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_WEST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(201); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_WEST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_A);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(200); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_SOUTH_WEST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_B);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(201); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_SOUTH_WEST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_B);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(200); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_NORTH_EAST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_B);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 0 );
            }
        }
        WHEN( "The distance between two detected objects is calculated" )
        {
            uint minimum(201); // units of percent
            uint maximum(UINT32_MAX);

            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                DSL_ODE_TRIGGER_DISTANCE_NEW(odeTriggerName.c_str(), source.c_str(), 
                    classIdA, classIdB, limit, minimum, maximum, 
                    DSL_BBOX_POINT_NORTH_EAST, DSL_DISTANCE_METHOD_PERCENT_WIDTH_B);

            THEN( "The correct number of occurrences is returned" )
            {
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta1) == true );
                REQUIRE( pOdeTrigger->CheckForOccurrence(NULL, displayMetaData, &frameMeta, &objectMeta2) == true );
                REQUIRE( pOdeTrigger->PostProcessFrame(NULL, displayMetaData, &frameMeta) == 1 );
            }
        }
    }
}    