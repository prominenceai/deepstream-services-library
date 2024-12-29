/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "Dsl.h"
#include "DslApi.h"

SCENARIO( "The SDE Triggers container is updated correctly on multiple new SDE Triggers", 
    "[sde-trigger-api]" )
{
    GIVEN( "An empty list of Triggers" ) 
    {
        std::wstring sdeTrigger_name1(L"occurrence-1");
        std::wstring sdeTrigger_name2(L"occurrence-2");
        std::wstring sdeTrigger_name3(L"occurrence-3");
        
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_list_size() == 0 );

        WHEN( "Several new Triggers are created" ) 
        {
            REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name1.c_str(), 
                NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name2.c_str(),
                NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name3.c_str(), 
                NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                // TODO complete verification after addition of Iterator API
                REQUIRE( dsl_sde_trigger_list_size() == 3 );

                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_sde_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Triggers container is updated correctly on SDE Trigger deletion", 
    "[sde-trigger-api]" )
{
    GIVEN( "A list of Triggers" ) 
    {
        std::wstring sdeTrigger_name1(L"occurrence-1");
        std::wstring sdeTrigger_name2(L"occurrence-2");
        std::wstring sdeTrigger_name3(L"occurrence-3");
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name1.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name2.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name3.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "When Triggers are deleted" )         
        {
            REQUIRE( dsl_sde_trigger_list_size() == 3 );
            REQUIRE( dsl_sde_trigger_delete(sdeTrigger_name1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sde_trigger_list_size() == 2 );

            const wchar_t* eventList[] = {sdeTrigger_name2.c_str(), sdeTrigger_name3.c_str(), NULL};
            REQUIRE( dsl_sde_trigger_delete_many(eventList) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_sde_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's Enabled setting can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        boolean ret_enabled(0);
        REQUIRE( dsl_sde_trigger_enabled_get(sdeTrigger_name.c_str(), 
            &ret_enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_enabled == 1 );

        WHEN( "When the SDE Trigger's Enabled setting is disabled" )         
        {
            uint new_enabled(0);
            REQUIRE( dsl_sde_trigger_enabled_set(sdeTrigger_name.c_str(), 
                new_enabled) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_enabled_get(sdeTrigger_name.c_str(), 
                    &ret_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_enabled == new_enabled );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's Mimimum Inference Confidence setting can be set/get", 
    "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        float ret_min_confidence(99.9);
        
        REQUIRE( dsl_sde_trigger_infer_confidence_min_get(sdeTrigger_name.c_str(), 
            &ret_min_confidence) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_min_confidence == 0.0 );

        WHEN( "When the SDE Trigger's minimum confidence setting is updated" )
        {
            float new_min_confidence(0.4);
            
            REQUIRE( dsl_sde_trigger_infer_confidence_min_set(sdeTrigger_name.c_str(), 
                new_min_confidence) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_infer_confidence_min_get(sdeTrigger_name.c_str(), 
                    &ret_min_confidence) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_confidence == new_min_confidence );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's Maximum Inference Confidence setting can be set/get", 
    "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        float ret_max_confidence(99.9);
        
        REQUIRE( dsl_sde_trigger_infer_confidence_max_get(sdeTrigger_name.c_str(), 
            &ret_max_confidence) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_max_confidence == 0.0 );

        WHEN( "When the SDE Trigger's maximum confidence setting is updated" )
        {
            float new_max_confidence(0.4);
            
            REQUIRE( dsl_sde_trigger_infer_confidence_max_set(sdeTrigger_name.c_str(), 
                new_max_confidence) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_infer_confidence_max_get(sdeTrigger_name.c_str(), 
                    &ret_max_confidence) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_confidence == new_max_confidence );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's Auto-Reset Timeout setting can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_timeout(99);
        REQUIRE( dsl_sde_trigger_reset_timeout_get(sdeTrigger_name.c_str(), 
            &ret_timeout) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_timeout == 0 );

        WHEN( "When the SDE Trigger's Enabled setting is disabled" )         
        {
            uint new_timeout(44);
            REQUIRE( dsl_sde_trigger_reset_timeout_set(sdeTrigger_name.c_str(), 
                new_timeout) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_reset_timeout_get(sdeTrigger_name.c_str(), 
                    &ret_timeout) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_timeout == new_timeout );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's Inference Name can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        const wchar_t* ret_infer; 
        REQUIRE( dsl_sde_trigger_infer_get(sdeTrigger_name.c_str(), 
            &ret_infer) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_infer == NULL );

        WHEN( "When the Trigger's classId is updated" )         
        {
            std::wstring new_infer(L"pgie");
            REQUIRE( dsl_sde_trigger_infer_set(sdeTrigger_name.c_str(), 
                new_infer.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_infer_get(sdeTrigger_name.c_str(),
                    &ret_infer) == DSL_RESULT_SUCCESS );
                std::wstring ret_infer_str(ret_infer);
                REQUIRE( ret_infer_str == new_infer );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's classId can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_sde_trigger_class_id_get(sdeTrigger_name.c_str(), 
            &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        WHEN( "When the Trigger's classId is updated" )         
        {
            uint new_class_id(4);
            REQUIRE( dsl_sde_trigger_class_id_set(sdeTrigger_name.c_str(), 
                new_class_id) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_class_id_get(sdeTrigger_name.c_str(), 
                    &ret_class_id) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_class_id == new_class_id );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's event limit can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_sde_trigger_class_id_get(sdeTrigger_name.c_str(), 
            &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        uint ret_limit(0);
        REQUIRE( dsl_sde_trigger_limit_event_get(sdeTrigger_name.c_str(), 
            &ret_limit) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_limit == limit );

        WHEN( "When the Trigger's limit is updated" )         
        {
            uint new_limit(44);
            REQUIRE( dsl_sde_trigger_limit_event_set(sdeTrigger_name.c_str(), 
                new_limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_limit_event_get(sdeTrigger_name.c_str(), 
                    &ret_limit) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_limit == new_limit );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger's frame limit can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_sde_trigger_class_id_get(sdeTrigger_name.c_str(), 
            &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        uint ret_limit(99);
        REQUIRE( dsl_sde_trigger_limit_frame_get(sdeTrigger_name.c_str(), 
            &ret_limit) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_limit == 0 );

        WHEN( "When the Trigger's limit is updated" )         
        {
            uint new_limit(44);
            REQUIRE( dsl_sde_trigger_limit_frame_set(sdeTrigger_name.c_str(), 
                new_limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_limit_frame_get(sdeTrigger_name.c_str(), 
                    &ret_limit) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_limit == new_limit );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    
SCENARIO( "An SDE Trigger's interval can be set/get", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_interval(99);
        REQUIRE( dsl_sde_trigger_interval_get(sdeTrigger_name.c_str(), 
            &ret_interval) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_interval == 0 );


        WHEN( "When the Trigger's limit is updated" )         
        {
            uint new_interval(44);
            REQUIRE( dsl_sde_trigger_interval_set(sdeTrigger_name.c_str(), new_interval) ==
                DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_sde_trigger_interval_get(sdeTrigger_name.c_str(), 
                    &ret_interval) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_interval == new_interval );
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

static void limit_event_listener(uint event, uint limit, void* client_data)
{
    std::cout << "limit event listner called with event = " 
        << event << ", and limit = " << limit << std::endl;
}


SCENARIO( "An SDE Trigger can add/remove a limit-state-change-listener", 
    "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "When a limit-state-change-listener is added" )         
        {
            REQUIRE( dsl_sde_trigger_limit_state_change_listener_add(sdeTrigger_name.c_str(),
                limit_event_listener, NULL) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_sde_trigger_limit_state_change_listener_add(sdeTrigger_name.c_str(),
                limit_event_listener, NULL) == 
                DSL_RESULT_SDE_TRIGGER_CALLBACK_ADD_FAILED );
            
            THEN( "The same listener function can be removed" ) 
            {
                REQUIRE( dsl_sde_trigger_limit_state_change_listener_remove(sdeTrigger_name.c_str(),
                    limit_event_listener) == DSL_RESULT_SUCCESS );

                // second call fail
                REQUIRE( dsl_sde_trigger_limit_state_change_listener_remove(sdeTrigger_name.c_str(),
                    limit_event_listener) == DSL_RESULT_SDE_TRIGGER_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger notifies its limit-event-listener on limit change", 
    "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger with a limit-event-listener" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sde_trigger_limit_state_change_listener_add(sdeTrigger_name.c_str(),
            limit_event_listener, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "When the trigger limit is updated" )         
        {
            // second call must fail
            REQUIRE( dsl_sde_trigger_limit_event_set(sdeTrigger_name.c_str(),
                DSL_SDE_TRIGGER_LIMIT_ONE) == DSL_RESULT_SUCCESS );
            
            THEN( "The limit-event-listener is notified" ) 
            {
                // requires manual/visual verification at this time.
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

static void enabled_state_change_listener(boolean enabled, void* client_data)
{
    std::cout << "enabled state change listner called with enabled = " << enabled << std::endl;
}

SCENARIO( "An SDE Trigger can add/remove an enabled-state-change-listener", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "When an enabled-state-change-listener is added" )         
        {
            REQUIRE( dsl_sde_trigger_enabled_state_change_listener_add(sdeTrigger_name.c_str(),
                enabled_state_change_listener, NULL) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_sde_trigger_enabled_state_change_listener_add(sdeTrigger_name.c_str(),
                enabled_state_change_listener, NULL) == 
                DSL_RESULT_SDE_TRIGGER_CALLBACK_ADD_FAILED );
            
            THEN( "The same listener function can be removed" ) 
            {
                REQUIRE( dsl_sde_trigger_enabled_state_change_listener_remove(sdeTrigger_name.c_str(),
                    enabled_state_change_listener) == DSL_RESULT_SUCCESS );

                // second call must fail
                REQUIRE( dsl_sde_trigger_enabled_state_change_listener_remove(sdeTrigger_name.c_str(),
                    enabled_state_change_listener) == DSL_RESULT_SDE_TRIGGER_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An SDE Trigger notifies its enabled-state-change-listener on change", "[sde-trigger-api]" )
{
    GIVEN( "An SDE Trigger" ) 
    {
        std::wstring sdeTrigger_name(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_sde_trigger_occurrence_new(sdeTrigger_name.c_str(), 
            NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sde_trigger_enabled_state_change_listener_add(sdeTrigger_name.c_str(),
            enabled_state_change_listener, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "The Trigger's enabled state is changed" )         
        {
            REQUIRE( dsl_sde_trigger_enabled_set(sdeTrigger_name.c_str(), false) == DSL_RESULT_SUCCESS );
            
            THEN( "The client listener function is called" ) 
            {
                // requires manual/visual verification at this time.
                REQUIRE( dsl_sde_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The SDE Trigger API checks for NULL input parameters", "[sde-trigger-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring trigger_name(L"test-trigger");
        std::wstring otherName(L"other");

        uint class_id(0);
        const wchar_t* source(NULL);
        const wchar_t* infer(NULL);
        boolean enabled(0);
        float confidence(0);
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_sde_trigger_occurrence_new(NULL, 
                    NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_trigger_reset(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_state_change_listener_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_state_change_listener_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_trigger_limit_event_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_event_get(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_event_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_frame_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_frame_get(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_limit_frame_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_sde_trigger_enabled_get(NULL, 
                    &enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_enabled_set(NULL, 
                    enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_enabled_state_change_listener_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_enabled_state_change_listener_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_trigger_class_id_get(NULL, 
                    &class_id) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_class_id_set(NULL, 
                    class_id) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_source_get(NULL, 
                    &source) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_source_get(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_source_set(NULL, 
                    source) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_infer_get(NULL, 
                    &infer) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_infer_get(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_infer_set(NULL, 
                    infer) == DSL_RESULT_INVALID_INPUT_PARAM );


                REQUIRE( dsl_sde_trigger_infer_confidence_min_get(NULL, 
                    &confidence) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_infer_confidence_min_set(NULL, 
                    confidence) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_infer_confidence_max_get(NULL, 
                    &confidence) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_infer_confidence_max_set(NULL, 
                    confidence) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_trigger_action_add(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_add(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_add_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_add_many(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_remove(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_remove_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_remove_many(trigger_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_action_remove_all(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_trigger_delete(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_trigger_delete_many(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
                    