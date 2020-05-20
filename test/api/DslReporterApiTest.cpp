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

SCENARIO( "The Components container is updated correctly on new Reporter", "[reporter-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring reporterName(L"reporter");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Reporter is created" ) 
        {

            REQUIRE( dsl_reporter_new(reporterName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on Reporter delete", "[reporter-api]" )
{
    GIVEN( "A new Reporter in memory" ) 
    {
        std::wstring reporterName(L"reporter");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_reporter_new(reporterName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Reporter is created" ) 
        {
            REQUIRE( dsl_component_delete(reporterName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Reporter's Enabled Setting can be disabled and re-enabled", "[reporter-api]" )
{
    GIVEN( "A new Reporter with Enabled Setting set to true by default" ) 
    {
        std::wstring reporterName(L"reporter");

        REQUIRE( dsl_reporter_new(reporterName.c_str()) == DSL_RESULT_SUCCESS );

        boolean preDisabled(false);
        REQUIRE( dsl_reporter_enabled_get(reporterName.c_str(), &preDisabled) == DSL_RESULT_SUCCESS );
        REQUIRE( preDisabled == true );

        // test negative case first - can't enable when already enabled
        REQUIRE( dsl_reporter_enabled_set(reporterName.c_str(), true) == DSL_RESULT_REPORTER_SET_FAILED );
        
        WHEN( "The Reporter's reporting is Disabled" ) 
        {
            boolean enabled(false);
            REQUIRE( dsl_reporter_enabled_set(reporterName.c_str(), enabled) == DSL_RESULT_SUCCESS );
            enabled = true;
            REQUIRE( dsl_reporter_enabled_get(reporterName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
            REQUIRE( enabled == false );
            
            THEN( "The Reporter's reporting can be Re-enabled" ) 
            {
                // test negative case first - can't disable when already disabled
                REQUIRE( dsl_reporter_enabled_set(reporterName.c_str(), false) == DSL_RESULT_REPORTER_SET_FAILED );
                
                enabled = true;
                REQUIRE( dsl_reporter_enabled_set(reporterName.c_str(), enabled) == DSL_RESULT_SUCCESS );
                enabled = false;
                REQUIRE( dsl_reporter_enabled_get(reporterName.c_str(), &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == true );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Reporter can Add and Remove a Detection Event", "[reporter-api]" )
{
    GIVEN( "A new Reporter and new Detection Event" ) 
    {
        std::wstring reporterName(L"reporter");

        std::wstring eventName(L"first-occurrence");
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_reporter_new(reporterName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_detection_event_new(eventName.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        WHEN( "The Detection Event is added to the Reporter" ) 
        {
            REQUIRE( dsl_reporter_detection_event_add(reporterName.c_str(), eventName.c_str()) == DSL_RESULT_SUCCESS );
            
            // Adding the same Event twice must fail
            REQUIRE( dsl_reporter_detection_event_add(reporterName.c_str(), eventName.c_str()) == DSL_RESULT_EVENT_IN_USE );
            
            THEN( "The same Detection Event can be removed correctly" ) 
            {
                REQUIRE( dsl_reporter_detection_event_remove(reporterName.c_str(), eventName.c_str()) == DSL_RESULT_SUCCESS );

                // Adding the same Event twice must fail
                REQUIRE( dsl_reporter_detection_event_remove(reporterName.c_str(), eventName.c_str()) == DSL_RESULT_REPORTER_EVENT_NOT_IN_USE );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_event_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Reporter can Add and Remove multiple Detection Events", "[reporter-api]" )
{
    GIVEN( "A new Reporter and multiple new Detection Events" ) 
    {
        std::wstring reporterName(L"reporter");

        std::wstring eventName1(L"first-occurrence-1");
        std::wstring eventName2(L"first-occurrence-2");
        std::wstring eventName3(L"first-occurrence-3");
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_reporter_new(reporterName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_detection_event_new(eventName1.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_detection_event_new(eventName2.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_detection_event_new(eventName3.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        WHEN( "The Detection Events are added to the Reporter" ) 
        {
            const wchar_t* events[] = {L"first-occurrence-1", L"first-occurrence-2", L"first-occurrence-3", NULL};

            REQUIRE( dsl_reporter_detection_event_add_many(reporterName.c_str(), events) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Detection Event can be removed correctly" ) 
            {
                REQUIRE( dsl_reporter_detection_event_remove_many(reporterName.c_str(), events) == DSL_RESULT_SUCCESS );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_event_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

