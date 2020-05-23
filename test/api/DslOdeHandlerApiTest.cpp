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
#include "DslApi.h"

SCENARIO( "The Components container is updated correctly on new ODE Handler", "[ode-handler-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring odeHandlerName(L"ode-handler");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new ODE Handler is created" ) 
        {

            REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on ODE Handler delete", "[ode-handler-api]" )
{
    GIVEN( "A new ODE Handler in memory" ) 
    {
        std::wstring odeHandlerName(L"ode-handler");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new ODE Handler is created" ) 
        {
            REQUIRE( dsl_component_delete(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A ODE Handler's Enabled Setting can be disabled and re-enabled", "[ode-handler-api]" )
{
    GIVEN( "A new ODE Handler with Enabled Setting set to true by default" ) 
    {
        std::wstring odeHandlerName(L"ode-handler");

        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );

        boolean preDisabled(false);
        REQUIRE( dsl_ode_handler_enabled_get(odeHandlerName.c_str(), &preDisabled) == DSL_RESULT_SUCCESS );
        REQUIRE( preDisabled == true );

        // test negative case first - can't enable when already enabled
        REQUIRE( dsl_ode_handler_enabled_set(odeHandlerName.c_str(), true) == DSL_RESULT_ODE_HANDLER_SET_FAILED );
        
        WHEN( "The ODE Handler's reporting is Disabled" ) 
        {
            boolean enabled(false);
            REQUIRE( dsl_ode_handler_enabled_set(odeHandlerName.c_str(), enabled) == DSL_RESULT_SUCCESS );
            enabled = true;
            REQUIRE( dsl_ode_handler_enabled_get(odeHandlerName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
            REQUIRE( enabled == false );
            
            THEN( "The ODE Handler's reporting can be Re-enabled" ) 
            {
                // test negative case first - can't disable when already disabled
                REQUIRE( dsl_ode_handler_enabled_set(odeHandlerName.c_str(), false) == DSL_RESULT_ODE_HANDLER_SET_FAILED );
                
                enabled = true;
                REQUIRE( dsl_ode_handler_enabled_set(odeHandlerName.c_str(), enabled) == DSL_RESULT_SUCCESS );
                enabled = false;
                REQUIRE( dsl_ode_handler_enabled_get(odeHandlerName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == true );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new ODE Handler can Add and Remove a Detection Event", "[ode-handler-api]" )
{
    GIVEN( "A new ODE Handler and new Detection Event" ) 
    {
        std::wstring odeHandlerName(L"ode-handler");

        std::wstring eventName(L"first-occurrence");
        uint evtype(DSL_ODE_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_type_new(eventName.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        WHEN( "The Detection Event is added to the ODE Handler" ) 
        {
            REQUIRE( dsl_ode_handler_type_add(odeHandlerName.c_str(), eventName.c_str()) == DSL_RESULT_SUCCESS );
            
            // Adding the same Event twice must fail
            REQUIRE( dsl_ode_handler_type_add(odeHandlerName.c_str(), eventName.c_str()) == DSL_RESULT_ODE_TYPE_IN_USE );
            
            THEN( "The same Detection Event can be removed correctly" ) 
            {
                REQUIRE( dsl_ode_handler_type_remove(odeHandlerName.c_str(), eventName.c_str()) == DSL_RESULT_SUCCESS );

                // Adding the same Event twice must fail
                REQUIRE( dsl_ode_handler_type_remove(odeHandlerName.c_str(), eventName.c_str()) == DSL_RESULT_ODE_HANDLER_TYPE_NOT_IN_USE );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_type_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new ODE Handler can Add and Remove multiple Detection Events", "[ode-handler-api]" )
{
    GIVEN( "A new ODE Handler and multiple new Detection Events" ) 
    {
        std::wstring odeHandlerName(L"ode-handler");

        std::wstring odeTypeName1(L"first-occurrence-1");
        std::wstring odeTypeName2(L"first-occurrence-2");
        std::wstring odeTypeName3(L"first-occurrence-3");
        uint evtype(DSL_ODE_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_ode_handler_new(odeHandlerName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_ode_type_new(odeTypeName1.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_type_new(odeTypeName2.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_type_new(odeTypeName3.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        WHEN( "The Detection Events are added to the ODE Handler" ) 
        {
            const wchar_t* odeTypes[] = {L"first-occurrence-1", L"first-occurrence-2", L"first-occurrence-3", NULL};

            REQUIRE( dsl_ode_handler_type_add_many(odeHandlerName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Detection Event can be removed correctly" ) 
            {
                REQUIRE( dsl_ode_handler_type_remove_many(odeHandlerName.c_str(), odeTypes) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_type_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

