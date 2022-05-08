/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

SCENARIO( "The ODE Accumulators container is updated correctly on multiple new ODE Accumulators", 
    "[ode-accumulator-api]" )
{
    GIVEN( "An empty list of Accumulators" ) 
    {
        std::wstring odeAccumulatorName1(L"accumulator-1");
        std::wstring odeAccumulatorName2(L"accumulator-2");
        std::wstring odeAccumulatorName3(L"accumulator-3");
        
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_accumulator_list_size() == 0 );

        WHEN( "Several new Accumulators are created" ) 
        {
            REQUIRE( dsl_ode_accumulator_new(
                odeAccumulatorName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_accumulator_new(
                odeAccumulatorName2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_accumulator_new(
                odeAccumulatorName3.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                // TODO complete verification after addition of Iterator API
                REQUIRE( dsl_ode_accumulator_list_size() == 3 );

                REQUIRE( dsl_ode_accumulator_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_accumulator_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Accumulators container is updated correctly on ODE Accumulator deletion", 
    "[ode-accumulator-api]" )
{
    GIVEN( "A list of Accumulators" ) 
    {
        std::wstring odeAccumulatorName1(L"accumulator-1");
        std::wstring odeAccumulatorName2(L"accumulator-2");
        std::wstring odeAccumulatorName3(L"accumulator-3");

        REQUIRE( dsl_ode_accumulator_new(
            odeAccumulatorName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_accumulator_new(
            odeAccumulatorName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_accumulator_new(
            odeAccumulatorName3.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When Accumulators are deleted" )         
        {
            REQUIRE( dsl_ode_accumulator_list_size() == 3 );
            REQUIRE( dsl_ode_accumulator_delete(odeAccumulatorName1.c_str()) == 
                DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_accumulator_list_size() == 2 );

            const wchar_t* eventList[] = 
                {odeAccumulatorName2.c_str(), odeAccumulatorName3.c_str(), NULL};
            REQUIRE( dsl_ode_accumulator_delete_many(eventList) == 
                DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_accumulator_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new ODE Accumulator can Add and Remove an ODE Action", 
    "[ode-accumulator-api]" )
{
    GIVEN( "A new ODE Accumulator and new ODE Action" ) 
    {
        std::wstring odeAccumulatorName(L"accumulator");
        std::wstring actionName(L"action");

        REQUIRE( dsl_ode_accumulator_new(
            odeAccumulatorName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_print_new(actionName.c_str(), false) == 
            DSL_RESULT_SUCCESS );

        WHEN( "The ODE Action is added to the ODE Accumulator" ) 
        {
            REQUIRE( dsl_ode_accumulator_action_add(odeAccumulatorName.c_str(), 
                actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            // Adding the same Event twice must fail
            REQUIRE( dsl_ode_accumulator_action_add(odeAccumulatorName.c_str(), 
                actionName.c_str()) == DSL_RESULT_ODE_ACCUMULATOR_ACTION_ADD_FAILED );
            
            THEN( "The same ODE Action can be removed correctly" ) 
            {
                REQUIRE( dsl_ode_accumulator_action_remove(odeAccumulatorName.c_str(), 
                    actionName.c_str()) == DSL_RESULT_SUCCESS );

                // Removing the same Event twice must fail
                REQUIRE( dsl_ode_accumulator_action_remove(odeAccumulatorName.c_str(), 
                    actionName.c_str()) == 
                        DSL_RESULT_ODE_ACCUMULATOR_ACTION_NOT_IN_USE );
                
                REQUIRE( dsl_ode_accumulator_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new ODE Accumulator can Add and Remove multiple ODE Actions", 
    "[ode-accumulator-api]" )
{
    GIVEN( "A new ODE Accumulator and multiple new ODE Actions" ) 
    {
        std::wstring odeAccumulatorName(L"accumulator");

        std::wstring odeActionName1(L"action-1");
        std::wstring odeActionName2(L"action-2");
        std::wstring odeActionName3(L"action-3");

        REQUIRE( dsl_ode_accumulator_new(
            odeAccumulatorName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_action_print_new(odeActionName1.c_str(),
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_print_new(odeActionName2.c_str(),
            false) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_print_new(odeActionName3.c_str(),
            false) == DSL_RESULT_SUCCESS );

        WHEN( "The ODE Actions are added to the ODE Accumulator" ) 
        {
            const wchar_t* odeActions[] = {L"action-1", L"action-2", L"action-3", NULL};

            REQUIRE( dsl_ode_accumulator_action_add_many(odeAccumulatorName.c_str(), 
                odeActions) == DSL_RESULT_SUCCESS );
            
            THEN( "The same ODE Action can be removed correctly" ) 
            {
                REQUIRE( dsl_ode_accumulator_action_remove_many(
                    odeAccumulatorName.c_str(), odeActions) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_ode_accumulator_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The ODE Accumulator API checks for NULL input parameters", "[ode-accumulator-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring ode_accumulator_name(L"accumulator");
        std::wstring action_name(L"other");
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_ode_accumulator_new(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_ode_accumulator_action_add(NULL, 
                    action_name.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_accumulator_action_add(
                    ode_accumulator_name.c_str(), NULL) == 
                        DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_accumulator_action_add_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_accumulator_action_add_many(
                    ode_accumulator_name.c_str(), NULL) == 
                        DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_accumulator_action_remove(NULL, 
                    action_name.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_accumulator_action_remove(
                    ode_accumulator_name.c_str(), NULL) == 
                        DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_accumulator_action_remove_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_accumulator_action_remove_many(
                    ode_accumulator_name.c_str(), NULL) == 
                        DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_accumulator_delete(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_accumulator_delete_many(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}