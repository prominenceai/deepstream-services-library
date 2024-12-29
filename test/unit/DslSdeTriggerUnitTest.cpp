/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
#include "DslSdeTrigger.h"
#include "DslSdeAction.h"

using namespace DSL;

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

SCENARIO( "A new SdeOccurreceTrigger is created correctly", "[SdeTrigger]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string sdeTriggerName("occurence");
        uint classId(1);
        uint limit(1);
        
        std::string source;

        WHEN( "A new SdeTrigger is created" )
        {
            DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
                DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), source.c_str(), classId, limit);

            THEN( "The SdeTriggers's memebers are setup and returned correctly" )
            {
                REQUIRE( pSdeTrigger->GetEnabled() == true );
                REQUIRE( pSdeTrigger->GetClassId() == classId );
                REQUIRE( pSdeTrigger->GetEventLimit() == limit );
                REQUIRE( pSdeTrigger->GetFrameLimit() == 0  );
                REQUIRE( pSdeTrigger->GetSource() == NULL );
                REQUIRE( pSdeTrigger->GetResetTimeout() == 0 );
                REQUIRE( pSdeTrigger->GetInterval() == 0 );
                uint minFrameCountN(123), minFrameCountD(123);
                pSdeTrigger->GetMinFrameCount(&minFrameCountN, &minFrameCountD);
                REQUIRE( minFrameCountN == 1 );
                REQUIRE( minFrameCountD == 1 );
            }
        }
    }
}

SCENARIO( "An SdeOccurrenceTrigger checks its enabled setting ", "[SdeTrigger]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string source;

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        // Frame Meta test data
        NvDsAudioFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = 1615768434973357000;
        frameMeta.source_id = 2;
        frameMeta.class_id = classId; // must match Trigger's classId
        
        WHEN( "The SDE Trigger is enabled and an SDE occurrence is simulated" )
        {
            pSdeTrigger->SetEnabled(true);
            
            THEN( "The SDE is triggered" )
            {
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == true );
            }
        }
        WHEN( "The SDE Trigger is disabled and an SDE occurrence is simulated" )
        {
            pSdeTrigger->SetEnabled(false);
            
            THEN( "The SDE is NOT triggered" )
            {
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == false );
            }
        }
    }
}

SCENARIO( "An SdeOccurrenceTrigger calls all enabled-state-change-listeners on state change", "[SdeTrigger]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string source;

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        REQUIRE( pSdeTrigger->AddEnabledStateChangeListener(
            enabled_state_change_listener_1, NULL) == true );

        REQUIRE( pSdeTrigger->AddEnabledStateChangeListener(
            enabled_state_change_listener_2, NULL) == true );
        
        WHEN( "The SDE Trigger is disabled" )
        {
            pSdeTrigger->SetEnabled(false);
            
            THEN( "All client callback functions are called" )
            {
                // requires manual/visual verification at this time.
            }
        }
        WHEN( "The SDE Trigger is enabled" )
        {
            pSdeTrigger->SetEnabled(true);
            
            THEN( "All client callback functions are called" )
            {
                // requires manual/visual verification at this time.
            }
        }
    }
}

SCENARIO( "An SdeOccurrenceTrigger handles a timed reset on event limit correctly", "[SdeTrigger]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        uint classId(1);
        uint limit(1); // one-shot tirgger
        uint reset_timeout(1);

        std::string source;

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        // Frame Meta test data
        NvDsAudioFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = 1615768434973357000;
        frameMeta.source_id = 2;
        frameMeta.class_id = classId; // must match Trigger's classId

        // Ensure correct defaults
        REQUIRE( pSdeTrigger->GetResetTimeout() == 0 );
        REQUIRE( pSdeTrigger->IsResetTimerRunning() == false);
        
        WHEN( "The SDE Trigger's ResetTimeout is first set" )
        {
            // Limit has NOT been reached
            pSdeTrigger->SetResetTimeout(reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pSdeTrigger->GetResetTimeout() == reset_timeout );
                REQUIRE( pSdeTrigger->IsResetTimerRunning() == false);
            }
        }

        WHEN( "The SDE Trigger's ResetTimeout is set when limit has been reached" )
        {
            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                &frameMeta) == true );

            // Limit has been reached
            pSdeTrigger->SetResetTimeout(reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pSdeTrigger->GetResetTimeout() == reset_timeout );
                REQUIRE( pSdeTrigger->IsResetTimerRunning() == true);
            }
        }
        WHEN( "The SDE Trigger's ResetTimeout is set when the timer is running" )
        {
            // Timeout is set before limit is reached
            pSdeTrigger->SetResetTimeout(reset_timeout);

            pSdeTrigger->PreProcessFrame(NULL, 
                &frameMeta);
            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                &frameMeta) == true );
            REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                &frameMeta) == true );

            // Timer must now be running
            REQUIRE( pSdeTrigger->IsResetTimerRunning() == true);

            uint new_reset_timeout(5);
            
            // Timeout is set before limit is reached
            pSdeTrigger->SetResetTimeout(new_reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pSdeTrigger->GetResetTimeout() == new_reset_timeout );
                REQUIRE( pSdeTrigger->IsResetTimerRunning() == true);
            }
        }
        WHEN( "The SDE Trigger's ResetTimeout is cleared when the timer is running" )
        {
            // Timeout is set before limit is reached
            pSdeTrigger->SetResetTimeout(reset_timeout);

            pSdeTrigger->PreProcessFrame(NULL, 
                &frameMeta);
            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                &frameMeta) == true );
            REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                &frameMeta) == true );

            // Timer must now be running
            REQUIRE( pSdeTrigger->IsResetTimerRunning() == true);

            uint new_reset_timeout(0);
            
            // Timeout is set before limit is reached
            pSdeTrigger->SetResetTimeout(new_reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pSdeTrigger->GetResetTimeout() == new_reset_timeout );
                REQUIRE( pSdeTrigger->IsResetTimerRunning() == false);
            }
        }
    }
}

SCENARIO( "An SdeOccurrenceTrigger handles a timed reset on frame limit correctly", "[error]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        uint classId(1);
        uint limit(DSL_SDE_TRIGGER_LIMIT_NONE); 
        uint reset_timeout(2);

        std::string source;

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), source.c_str(), classId, limit);
            
        // Setting a frame limit of one.
        pSdeTrigger->SetFrameLimit(1);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        // Frame Meta test data
        NvDsAudioFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = 1615768434973357000;
        frameMeta.source_id = 2;
        frameMeta.class_id = classId; // must match Trigger's classId

        // Ensure correct defaults
        REQUIRE( pSdeTrigger->GetResetTimeout() == 0 );
        REQUIRE( pSdeTrigger->IsResetTimerRunning() == false);
        
        WHEN( "The SDE Trigger's ResetTimeout is set when frame limit has been reached" )
        {
            pSdeTrigger->PreProcessFrame(NULL, 
                &frameMeta);
            REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                &frameMeta) == true );
            REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                &frameMeta) == true );

            // Limit has been reached
            pSdeTrigger->SetResetTimeout(reset_timeout);
            
            THEN( "The correct timeout and is-running values are returned" )
            {
                REQUIRE( pSdeTrigger->GetResetTimeout() == reset_timeout );
                REQUIRE( pSdeTrigger->IsResetTimerRunning() == true);
            }
        }
    }
}

SCENARIO( "An SdeOccurrenceTrigger notifies its limit-state-listeners", "[SdeTrigger]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        uint classId(1);
        uint limit(1); // one-shot tirgger
        uint reset_timeout(1);

        std::string source;

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        // Frame Meta test data
        NvDsAudioFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = 1615768434973357000;
        frameMeta.source_id = 2;
        frameMeta.class_id = classId; // must match Trigger's classId

        REQUIRE( pSdeTrigger->AddLimitStateChangeListener(
            limit_state_change_listener_1, NULL) == true );

        REQUIRE( pSdeTrigger->AddLimitStateChangeListener(
            limit_state_change_listener_2, NULL) == true );
        
        WHEN( "When an SDE occures and the Trigger reaches its limit" )
        {
            pSdeTrigger->PreProcessFrame(NULL, 
                &frameMeta);
            // First occurrence will reach the Trigger's limit of one
            REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                &frameMeta) == true );
            REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                &frameMeta) == true );

            THEN( "All client listeners are notified" )
            {
                // NOTE requires manual confirmation at this time.
                
                pSdeTrigger->Reset();
                
                REQUIRE( pSdeTrigger->RemoveLimitStateChangeListener(
                    limit_state_change_listener_1) == true );

                REQUIRE( pSdeTrigger->RemoveLimitStateChangeListener(
                    limit_state_change_listener_2) == true );
            }
        }
    }
}

SCENARIO( "An SDE Occurrence Trigger checks its minimum inference confidence correctly", 
    "[SdeTrigger]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        // Frame Meta test data
        NvDsAudioFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = 1615768434973357000;
        frameMeta.source_id = 2;
        frameMeta.class_id = classId; // must match Trigger's classId
        
        frameMeta.confidence = 0.5;
        
        WHEN( "The SDE Trigger's minimum confidence is less than the Object's confidence" )
        {
            pSdeTrigger->SetMinConfidence(0.4999);
            
            THEN( "The SDE is triggered" )
            {
                pSdeTrigger->PreProcessFrame(NULL, 
                    &frameMeta);
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == true );
                REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                    &frameMeta) == true );
            }
        }
        WHEN( "The SDE Trigger's minimum confidence is equal to the Object's confidence" )
        {
            pSdeTrigger->SetMinConfidence(0.5);
            
            THEN( "The SDE is triggered" )
            {
                pSdeTrigger->PreProcessFrame(NULL, 
                    &frameMeta);
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == true );
                REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                    &frameMeta) == true );
            }
        }
        WHEN( "The SDE Trigger's minimum confidence is greater tahn the Object's confidence" )
        {
            pSdeTrigger->SetMinConfidence(0.5001);
            
            THEN( "The SDE is NOT triggered" )
            {
                pSdeTrigger->PreProcessFrame(NULL, 
                    &frameMeta);
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == false );
                REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                    &frameMeta) == false );
            }
        }
    }
}

SCENARIO( "An SDE Occurrence Trigger checks its maximum inference confidence correctly", 
    "[SdeTrigger]" )
{
    GIVEN( "A new SdeTrigger with default criteria" ) 
    {
        std::string sdeTriggerName("occurence");
        std::string source;
        uint classId(1);
        uint limit(0); // not limit

        std::string sdeActionName("print-action");

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pSdeTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(sdeTriggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pSdeAction = 
            DSL_SDE_ACTION_PRINT_NEW(sdeActionName.c_str(), false);
            
        REQUIRE( pSdeTrigger->AddAction(pSdeAction) == true );        

        // Frame Meta test data
        NvDsAudioFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = 1615768434973357000;
        frameMeta.source_id = 2;
        frameMeta.class_id = classId; // must match Trigger's classId
        
        frameMeta.confidence = 0.5;
        
        WHEN( "The SDE Trigger's maximum confidence is less than the Object's confidence" )
        {
            pSdeTrigger->SetMaxConfidence(0.4999);
            
            THEN( "The SDE is NOT triggered" )
            {
                pSdeTrigger->PreProcessFrame(NULL, 
                    &frameMeta);
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == false );
                REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                    &frameMeta) == false );
            }
        }
        WHEN( "The SDE Trigger's maximum confidence is equal to the Object's confidence" )
        {
            pSdeTrigger->SetMaxConfidence(0.5);
            
            THEN( "The SDE is triggered" )
            {
                pSdeTrigger->PreProcessFrame(NULL, 
                    &frameMeta);
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == true );
                REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                    &frameMeta) == true );
            }
        }
        WHEN( "The SDE Trigger's maximum confidence is greater than the Object's confidence" )
        {
            pSdeTrigger->SetMaxConfidence(0.5001);
            
            THEN( "The SDE is triggered" )
            {
                pSdeTrigger->PreProcessFrame(NULL, 
                    &frameMeta);
                REQUIRE( pSdeTrigger->CheckForOccurrence(NULL, 
                    &frameMeta) == true );
                REQUIRE( pSdeTrigger->PostProcessFrame(NULL, 
                    &frameMeta) == true );
            }
        }
    }
}

