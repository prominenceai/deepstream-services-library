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
#include "DslApi.h"

SCENARIO( "The PPH container is updated correctly on new PPH", "[pph-api]" )
{
    GIVEN( "An empty list of Pad Probe Handlers" ) 
    {
        std::wstring odePphName(L"pph");

        REQUIRE( dsl_pph_list_size() == 0 );

        WHEN( "A new ODE PPH is created" ) 
        {
            REQUIRE( dsl_pph_ode_new(odePphName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pph_list_size() == 1 );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The PPH container is updated correctly on PPH delete", "[pph-api]" )
{
    GIVEN( "A new PPH in memory" ) 
    {
        std::wstring odePphName(L"pph");

        REQUIRE( dsl_pph_list_size() == 0 );
        REQUIRE( dsl_pph_ode_new(odePphName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_list_size() == 1 );

        WHEN( "The PPH is deleted" ) 
        {
            REQUIRE( dsl_pph_delete(odePphName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A ODE Handler's Enabled Setting can be disabled and re-enabled", "[pph-api]" )
{
    GIVEN( "A new ODE Handler with Enabled Setting set to true by default" ) 
    {
        std::wstring odePphName(L"pph");

        REQUIRE( dsl_pph_ode_new(odePphName.c_str()) == DSL_RESULT_SUCCESS );

        boolean preDisabled(false);
        REQUIRE(dsl_pph_enabled_get(odePphName.c_str(), &preDisabled) == DSL_RESULT_SUCCESS );
        REQUIRE( preDisabled == true );

        // test negative case first - can't enable when already enabled
        REQUIRE( dsl_pph_enabled_set(odePphName.c_str(), true) == DSL_RESULT_PPH_SET_FAILED );
        
        WHEN( "The PPH is Disabled" ) 
        {
            boolean enabled(false);
            REQUIRE( dsl_pph_enabled_set(odePphName.c_str(), enabled) == DSL_RESULT_SUCCESS );
            enabled = true;
            REQUIRE( dsl_pph_enabled_get(odePphName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
            REQUIRE( enabled == false );
            
            THEN( "The PPH can be Re-enabled" ) 
            {
                // test negative case first - can't disable when already disabled
                REQUIRE( dsl_pph_enabled_set(odePphName.c_str(), false) == DSL_RESULT_PPH_SET_FAILED );
                
                enabled = true;
                REQUIRE( dsl_pph_enabled_set(odePphName.c_str(), enabled) == DSL_RESULT_SUCCESS );
                enabled = false;
                REQUIRE( dsl_pph_enabled_get(odePphName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == true );

                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new ODE Handler can Add and Remove a ODE Trigger", "[pph-api]" )
{
    GIVEN( "A new ODE Handler and new ODE Trigger" ) 
    {
        std::wstring odePphName(L"pph");

        std::wstring triggerName(L"first-occurrence");
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_pph_ode_new(odePphName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(triggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "The ODE Trigger is added to the ODE Handler" ) 
        {
            REQUIRE( dsl_pph_ode_trigger_add(odePphName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            // Adding the same Event twice must fail
            REQUIRE( dsl_pph_ode_trigger_add(odePphName.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_IN_USE );
            
            THEN( "The same ODE Trigger can be removed correctly" ) 
            {
                REQUIRE( dsl_pph_ode_trigger_remove(odePphName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );

                // Adding the same Event twice must fail
                REQUIRE( dsl_pph_ode_trigger_remove(odePphName.c_str(), triggerName.c_str()) == DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new ODE Handler can Add and Remove multiple ODE Triggers", "[pph-api]" )
{
    GIVEN( "A new ODE Handler and multiple new ODE Triggers" ) 
    {
        std::wstring odePphName(L"pph");

        std::wstring odeTriggerName1(L"occurrence-1");
        std::wstring odeTriggerName2(L"occurrence-2");
        std::wstring odeTriggerName3(L"occurrence-3");
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_pph_ode_new(odePphName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName1.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName2.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName3.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "The ODE Triggers are added to the ODE Handler" ) 
        {
            const wchar_t* odeTypes[] = {L"occurrence-1", L"occurrence-2", L"occurrence-3", NULL};

            REQUIRE( dsl_pph_ode_trigger_add_many(odePphName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );
            
            THEN( "The same ODE Trigger can be removed correctly" ) 
            {
                REQUIRE( dsl_pph_ode_trigger_remove_many(odePphName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

void buffer_timeout_handler_cb(uint timeout, void* client_data)
{
    
}

SCENARIO( "A Buffer Timeout Pad Probe Handler can be created and deleted", "[pph-api]" )
{
    GIVEN( "Atributes for a new Buffer Timeout Pad Probe Handler" ) 
    {
        std::wstring buffer_timeout_pph(L"pph");

        REQUIRE( dsl_pph_list_size() == 0 );

        WHEN( "The PPH is created" ) 
        {
            REQUIRE( dsl_pph_buffer_timeout_new(buffer_timeout_pph.c_str(),
                1, buffer_timeout_handler_cb, NULL) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pph_list_size() == 1 );

            // second call must fail
            REQUIRE( dsl_pph_buffer_timeout_new(buffer_timeout_pph.c_str(),
                1, buffer_timeout_handler_cb, NULL) == DSL_RESULT_PPH_NAME_NOT_UNIQUE );
            
            boolean enabled(0);
            REQUIRE( dsl_pph_enabled_get(buffer_timeout_pph.c_str(), &enabled) 
                == DSL_RESULT_SUCCESS);
            REQUIRE( enabled == true );
            
            THEN( "The PPH can then be deleted" )
            {
                REQUIRE( dsl_pph_delete(buffer_timeout_pph.c_str()) == DSL_RESULT_SUCCESS );
                
                // second call must fail
                REQUIRE( dsl_pph_delete(buffer_timeout_pph.c_str()) == DSL_RESULT_PPH_NAME_NOT_FOUND );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

uint eos_handler_cb(void* client_data)
{
    return DSL_PAD_PROBE_DROP;
}

SCENARIO( "A EOS Pad Probe Handler can be created and deleted", "[pph-api]" )
{
    GIVEN( "Atributes for a new EOS Pad Probe Handler" ) 
    {
        std::wstring eos_pph(L"pph");

        REQUIRE( dsl_pph_list_size() == 0 );

        WHEN( "The PPH is created" ) 
        {
            REQUIRE( dsl_pph_eos_new(eos_pph.c_str(),
                eos_handler_cb, NULL) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pph_list_size() == 1 );

            // second call must fail
            REQUIRE( dsl_pph_eos_new(eos_pph.c_str(),
                eos_handler_cb, NULL) == DSL_RESULT_PPH_NAME_NOT_UNIQUE );
            
            boolean enabled(0);
            REQUIRE( dsl_pph_enabled_get(eos_pph.c_str(), &enabled) 
                == DSL_RESULT_SUCCESS);
            REQUIRE( enabled == true );
            
            THEN( "The PPH can then be deleted" )
            {
                REQUIRE( dsl_pph_delete(eos_pph.c_str()) == DSL_RESULT_SUCCESS );
                
                // second call must fail
                REQUIRE( dsl_pph_delete(eos_pph.c_str()) == DSL_RESULT_PPH_NAME_NOT_FOUND );
                REQUIRE( dsl_pph_list_size() == 0 );
            }
        }
    }
}

static void bt_handler_cb(uint timeout, void* client_data)
{
}

SCENARIO( "A Buffer Timeout Handler's Enabled Setting can be disabled and re-enabled", "[pph-api]" )
{
    GIVEN( "A new ODE Handler with Enabled Setting set to true by default" ) 
    {
        std::wstring bufferTimeoutPphName(L"pph");

        REQUIRE( dsl_pph_buffer_timeout_new(bufferTimeoutPphName.c_str(),
            2, bt_handler_cb, NULL) == DSL_RESULT_SUCCESS );

        boolean preDisabled(false);
        REQUIRE(dsl_pph_enabled_get(bufferTimeoutPphName.c_str(), &preDisabled) == DSL_RESULT_SUCCESS );
        REQUIRE( preDisabled == true );

        // test negative case first - can't enable when already enabled
        REQUIRE( dsl_pph_enabled_set(bufferTimeoutPphName.c_str(), true) == DSL_RESULT_PPH_SET_FAILED );
        
        WHEN( "The PPH is Disabled" ) 
        {
            boolean enabled(false);
            REQUIRE( dsl_pph_enabled_set(bufferTimeoutPphName.c_str(), enabled) == DSL_RESULT_SUCCESS );
            enabled = true;
            REQUIRE( dsl_pph_enabled_get(bufferTimeoutPphName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
            REQUIRE( enabled == false );
            
            THEN( "The PPH can be Re-enabled" ) 
            {
                // test negative case first - can't disable when already disabled
                REQUIRE( dsl_pph_enabled_set(bufferTimeoutPphName.c_str(), false) == DSL_RESULT_PPH_SET_FAILED );
                
                enabled = true;
                REQUIRE( dsl_pph_enabled_set(bufferTimeoutPphName.c_str(), enabled) == DSL_RESULT_SUCCESS );
                enabled = false;
                REQUIRE( dsl_pph_enabled_get(bufferTimeoutPphName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == true );

                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Pad Probe Handler API checks for NULL input parameters", "[pph-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring pphName  = L"test-pph";
        std::wstring otherName  = L"other";
        
        uint interval(0);
        boolean enabled(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_pph_ode_new(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_ode_trigger_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_add(pphName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_add_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_add_many(pphName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_remove(pphName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_remove_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_remove_many(pphName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_ode_trigger_remove_all(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_custom_new(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_custom_new(pphName.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_meter_new(NULL, 0, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_meter_new(pphName.c_str(), 0, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_meter_interval_get(NULL, &interval) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_meter_interval_set(NULL, interval) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_buffer_timeout_new(NULL, 1, NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_buffer_timeout_new(pphName.c_str(), 1, NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_eos_new(NULL, NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_eos_new(pphName.c_str(), NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_enabled_get(NULL, &enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_enabled_set(NULL, enabled) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pph_delete(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pph_delete_many(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );


                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
