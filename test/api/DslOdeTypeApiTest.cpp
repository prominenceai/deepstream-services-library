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
#include "Dsl.h"
#include "DslApi.h"

SCENARIO( "The ODE Triggers container is updated correctly on multiple new ODE Triggers", "[ode-trigger-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring odeTypeName1(L"occurrence-1");
        std::wstring odeTypeName2(L"occurrence-2");
        std::wstring odeTypeName3(L"occurrence-3");
        
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_list_size() == 0 );

        WHEN( "Several new Events are created" ) 
        {
            REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName1.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName2.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName3.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                // TODO complete verification after addition of Iterator API
                REQUIRE( dsl_ode_trigger_list_size() == 3 );

                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Events container is updated correctly on Detection Event deletion", "[ode-trigger-api]" )
{
    GIVEN( "A list of Events" ) 
    {
        std::wstring odeTypeName1(L"occurrence-1");
        std::wstring odeTypeName2(L"occurrence-2");
        std::wstring odeTypeName3(L"occurrence-3");
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName1.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName2.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName3.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "When Events are deleted" )         
        {
            REQUIRE( dsl_ode_trigger_list_size() == 3 );
            REQUIRE( dsl_ode_trigger_delete(odeTypeName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_trigger_list_size() == 2 );

            const wchar_t* eventList[] = {odeTypeName2.c_str(), odeTypeName3.c_str(), NULL};
            REQUIRE( dsl_ode_trigger_delete_many(eventList) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A Detection Event's Enabled setting can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring odeTypeName(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );

        boolean ret_enabled(0);
        REQUIRE( dsl_ode_trigger_enabled_get(odeTypeName.c_str(), &ret_enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_enabled == 1 );

        WHEN( "When the ODE Type's Enabled setting is disabled" )         
        {
            uint new_enabled(0);
            REQUIRE( dsl_ode_trigger_enabled_set(odeTypeName.c_str(), new_enabled) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_enabled_get(odeTypeName.c_str(), &ret_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_enabled == new_enabled );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Detection Event's classId can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring odeTypeName(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_ode_trigger_class_id_get(odeTypeName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        WHEN( "When the Event's classId is updated" )         
        {
            uint new_class_id(4);
            REQUIRE( dsl_ode_trigger_class_id_set(odeTypeName.c_str(), new_class_id) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_class_id_get(odeTypeName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_class_id == new_class_id );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Detection Event's minimum dimensions can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring odeTypeName(L"occurrence");
        uint limit(0);
        uint class_id(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );

        uint min_width(1), min_height(1);
        REQUIRE( dsl_ode_trigger_dimensions_min_get(odeTypeName.c_str(), &min_width, &min_height) == DSL_RESULT_SUCCESS );
        REQUIRE( min_width == 0 );
        REQUIRE( min_height == 0 );

        WHEN( "When the Event's min dimensions are updated" )         
        {
            uint new_min_width(300), new_min_height(200);
            REQUIRE( dsl_ode_trigger_dimensions_min_set(odeTypeName.c_str(), new_min_width, new_min_height) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_dimensions_min_get(odeTypeName.c_str(), &min_width, &min_height) == DSL_RESULT_SUCCESS );
                REQUIRE( min_width == new_min_width );
                REQUIRE( min_height == new_min_height );
                
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Detection Event's minimum frame count can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring odeTypeName(L"first-occurrence");
        
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTypeName.c_str(), class_id, limit) == DSL_RESULT_SUCCESS );

        uint min_count_n(0), min_count_d(0);
        REQUIRE( dsl_ode_trigger_frame_count_min_get(odeTypeName.c_str(), &min_count_n, &min_count_d) == DSL_RESULT_SUCCESS );
        REQUIRE( min_count_n == 1 );
        REQUIRE( min_count_d == 1 );

        WHEN( "When the Event's min frame count properties are updated" )         
        {
            uint new_min_count_n(300), new_min_count_d(200);
            REQUIRE( dsl_ode_trigger_frame_count_min_set(odeTypeName.c_str(), new_min_count_n, new_min_count_d) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_frame_count_min_get(odeTypeName.c_str(), &min_count_n, &min_count_d) == DSL_RESULT_SUCCESS );
                REQUIRE( min_count_n == new_min_count_n );
                REQUIRE( min_count_d == new_min_count_d );
                
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

