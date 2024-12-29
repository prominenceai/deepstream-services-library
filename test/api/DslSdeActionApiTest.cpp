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
#include "Dsl.h"
#include "DslApi.h"

SCENARIO( "A new Print SDE Action can be created and deleted", "[sde-action-api]" )
{
    GIVEN( "Attributes for a new Print SDE Action" ) 
    {
        std::wstring action_name(L"print-action");

        WHEN( "A new Print Action is created" ) 
        {
            REQUIRE( dsl_sde_action_print_new(action_name.c_str(), 
                false) == DSL_RESULT_SUCCESS );
            
            THEN( "The Print Action can be deleted" ) 
            {
                REQUIRE( dsl_sde_action_delete(action_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_sde_action_list_size() == 0 );
            }
        }
        WHEN( "A new Print Action is created" ) 
        {
            REQUIRE( dsl_sde_action_print_new(action_name.c_str(), 
                false) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Print Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_sde_action_print_new(action_name.c_str(), 
                    false) == DSL_RESULT_SDE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_sde_action_delete(action_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_sde_action_list_size() == 0 );
            }
        }
    }
}

static void enabled_state_change_listener(boolean enabled, void* client_data)
{
    
}

SCENARIO( "An SDE Action can add/remove an enabled-state-change-listener", 
    "[sde-action-api]" )
{
    GIVEN( "A new Capture Action and client listener callback" )
    {
        std::wstring action_name(L"capture-action");
        std::wstring outdir(L"./");
        
        REQUIRE( dsl_sde_action_print_new(action_name.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        WHEN( "When an enabled-state-change-listener is added" )         
        {
            REQUIRE( dsl_sde_action_enabled_state_change_listener_add(
                action_name.c_str(), enabled_state_change_listener, 
                NULL) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_sde_action_enabled_state_change_listener_add(
                action_name.c_str(),
                enabled_state_change_listener, NULL) == 
                DSL_RESULT_SDE_ACTION_CALLBACK_ADD_FAILED );
            
            THEN( "The same listener function can be removed" ) 
            {
                REQUIRE( dsl_sde_action_enabled_state_change_listener_remove(
                    action_name.c_str(), enabled_state_change_listener) 
                        == DSL_RESULT_SUCCESS );

                // second call must fail
                REQUIRE( dsl_sde_action_enabled_state_change_listener_remove(
                    action_name.c_str(), enabled_state_change_listener) 
                        == DSL_RESULT_SDE_ACTION_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_sde_action_delete(action_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

static void sde_occurrence_monitor_cb(dsl_sde_occurrence_info* pInfo, 
    void* client_data)
{
}

SCENARIO( "A new Monitor SDE Action can be created and deleted", "[sde-action-api]" )
{
    GIVEN( "Attributes for a new Monitor SDE Action" ) 
    {
        std::wstring action_name(L"monitor-action");

        WHEN( "A new Monitor Action is created" ) 
        {
            REQUIRE( dsl_sde_action_monitor_new(action_name.c_str(), 
                sde_occurrence_monitor_cb, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Monitor Action can be deleted" ) 
            {
                REQUIRE( dsl_sde_action_delete(action_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_sde_action_list_size() == 0 );
            }
        }
        WHEN( "A new Monitor Action is created" ) 
        {
            REQUIRE( dsl_sde_action_monitor_new(action_name.c_str(), 
                sde_occurrence_monitor_cb, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Monitor Action of the same name fails to create" ) 
            {
                REQUIRE( dsl_sde_action_monitor_new(action_name.c_str(), 
                    sde_occurrence_monitor_cb, NULL) 
                        == DSL_RESULT_SDE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_sde_action_delete(action_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_sde_action_list_size() == 0 );
            }
        }
    }
}


SCENARIO( "The SDE Action API checks for NULL input parameters", 
    "[sde-action-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring action_name  = L"test-action";
        boolean enabled(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_sde_action_print_new(NULL, 
                    false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_action_monitor_new(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_action_monitor_new(action_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_action_enabled_get(NULL, 
                    &enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_action_enabled_get(action_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_action_enabled_set(NULL, 
                    false) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sde_action_delete(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sde_action_delete_many(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

            }
        }
    }
}
