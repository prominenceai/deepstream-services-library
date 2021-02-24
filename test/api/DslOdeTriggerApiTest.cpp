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
    GIVEN( "An empty list of Triggers" ) 
    {
        std::wstring odeTriggerName1(L"occurrence-1");
        std::wstring odeTriggerName2(L"occurrence-2");
        std::wstring odeTriggerName3(L"occurrence-3");
        
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_list_size() == 0 );

        WHEN( "Several new Triggers are created" ) 
        {
            REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName1.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName2.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName3.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
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

SCENARIO( "The Triggers container is updated correctly on ODE Trigger deletion", "[ode-trigger-api]" )
{
    GIVEN( "A list of Triggers" ) 
    {
        std::wstring odeTriggerName1(L"occurrence-1");
        std::wstring odeTriggerName2(L"occurrence-2");
        std::wstring odeTriggerName3(L"occurrence-3");
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName1.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName2.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName3.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        WHEN( "When Triggers are deleted" )         
        {
            REQUIRE( dsl_ode_trigger_list_size() == 3 );
            REQUIRE( dsl_ode_trigger_delete(odeTriggerName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_trigger_list_size() == 2 );

            const wchar_t* eventList[] = {odeTriggerName2.c_str(), odeTriggerName3.c_str(), NULL};
            REQUIRE( dsl_ode_trigger_delete_many(eventList) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "An ODE Trigger's Enabled setting can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "An ODE Trigger" ) 
    {
        std::wstring odeTriggerName(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        boolean ret_enabled(0);
        REQUIRE( dsl_ode_trigger_enabled_get(odeTriggerName.c_str(), &ret_enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_enabled == 1 );

        WHEN( "When the ODE Type's Enabled setting is disabled" )         
        {
            uint new_enabled(0);
            REQUIRE( dsl_ode_trigger_enabled_set(odeTriggerName.c_str(), new_enabled) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_enabled_get(odeTriggerName.c_str(), &ret_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_enabled == new_enabled );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An ODE Trigger's classId can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "An ODE Trigger" ) 
    {
        std::wstring odeTriggerName(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_ode_trigger_class_id_get(odeTriggerName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        WHEN( "When the Trigger's classId is updated" )         
        {
            uint new_class_id(4);
            REQUIRE( dsl_ode_trigger_class_id_set(odeTriggerName.c_str(), new_class_id) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_class_id_get(odeTriggerName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_class_id == new_class_id );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An ODE Trigger's limit can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "An ODE Trigger" ) 
    {
        std::wstring odeTriggerName(L"occurrence");
        
        uint class_id(9);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_ode_trigger_class_id_get(odeTriggerName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        uint ret_limit(0);
        REQUIRE( dsl_ode_trigger_limit_get(odeTriggerName.c_str(), &ret_limit) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_limit == limit );

        WHEN( "When the Trigger's limit is updated" )         
        {
            uint new_limit(44);
            REQUIRE( dsl_ode_trigger_limit_set(odeTriggerName.c_str(), new_limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_limit_get(odeTriggerName.c_str(), &ret_limit) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_limit == new_limit );
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An ODE Trigger's minimum dimensions can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "An ODE Trigger" ) 
    {
        std::wstring odeTriggerName(L"occurrence");
        uint limit(0);
        uint class_id(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        float min_width(1), min_height(1);
        REQUIRE( dsl_ode_trigger_dimensions_min_get(odeTriggerName.c_str(), &min_width, &min_height) == DSL_RESULT_SUCCESS );
        REQUIRE( min_width == 0 );
        REQUIRE( min_height == 0 );

        WHEN( "When the Trigger's min dimensions are updated" )         
        {
            float new_min_width(300), new_min_height(200);
            REQUIRE( dsl_ode_trigger_dimensions_min_set(odeTriggerName.c_str(), new_min_width, new_min_height) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_dimensions_min_get(odeTriggerName.c_str(), &min_width, &min_height) == DSL_RESULT_SUCCESS );
                REQUIRE( min_width == new_min_width );
                REQUIRE( min_height == new_min_height );
                
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An ODE Trigger's maximum dimensions can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "An ODE Trigger" ) 
    {
        std::wstring odeTriggerName(L"occurrence");
        uint limit(0);
        uint class_id(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        float max_width(1), max_height(1);
        REQUIRE( dsl_ode_trigger_dimensions_max_get(odeTriggerName.c_str(), &max_width, &max_height) == DSL_RESULT_SUCCESS );
        REQUIRE( max_width == 0 );
        REQUIRE( max_height == 0 );

        WHEN( "When the Trigger's max dimensions are updated" )         
        {
            uint new_max_width(300), new_max_height(200);
            REQUIRE( dsl_ode_trigger_dimensions_max_set(odeTriggerName.c_str(), new_max_width, new_max_height) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_dimensions_max_get(odeTriggerName.c_str(), &max_width, &max_height) == DSL_RESULT_SUCCESS );
                REQUIRE( max_width == new_max_width );
                REQUIRE( max_height == new_max_height );
                
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "An ODE Trigger's minimum frame count can be set/get", "[ode-trigger-api]" )
{
    GIVEN( "An ODE Trigger" ) 
    {
        std::wstring odeTriggerName(L"first-occurrence");
        
        uint class_id(0);
        uint limit(0);

        REQUIRE( dsl_ode_trigger_occurrence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );

        uint min_count_n(0), min_count_d(0);
        REQUIRE( dsl_ode_trigger_frame_count_min_get(odeTriggerName.c_str(), &min_count_n, &min_count_d) == DSL_RESULT_SUCCESS );
        REQUIRE( min_count_n == 1 );
        REQUIRE( min_count_d == 1 );

        WHEN( "When the Trigger's min frame count properties are updated" )         
        {
            uint new_min_count_n(300), new_min_count_d(200);
            REQUIRE( dsl_ode_trigger_frame_count_min_set(odeTriggerName.c_str(), new_min_count_n, new_min_count_d) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_ode_trigger_frame_count_min_get(odeTriggerName.c_str(), &min_count_n, &min_count_d) == DSL_RESULT_SUCCESS );
                REQUIRE( min_count_n == new_min_count_n );
                REQUIRE( min_count_d == new_min_count_d );
                
                REQUIRE( dsl_ode_trigger_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new Absence Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Absence Trigger" ) 
    {
        std::wstring odeTriggerName(L"absence");
        uint class_id(0);
        uint limit(0);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_absence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_absence_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_absence_new(odeTriggerName.c_str(), NULL, class_id, limit) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Always Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Always Trigger" ) 
    {
        std::wstring odeTriggerName(L"always");
        uint when(DSL_ODE_POST_OCCURRENCE_CHECK);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_always_new(odeTriggerName.c_str(), NULL, when) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_always_new(odeTriggerName.c_str(), NULL, when) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_always_new(odeTriggerName.c_str(), NULL, when) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
        WHEN( "When an Invalid 'when' parameter is provided" )         
        {
            when += 100;
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_always_new(odeTriggerName.c_str(), NULL, when+100) 
                    == DSL_RESULT_ODE_TRIGGER_ALWAYS_WHEN_PARAMETER_INVALID );
                    
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Summation Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Summation Trigger" ) 
    {
        std::wstring odeTriggerName(L"summation");
        uint class_id(0);
        uint limit(0);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_summation_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_summation_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_summation_new(odeTriggerName.c_str(), NULL, class_id, limit) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Intersection Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Intersection Trigger" ) 
    {
        std::wstring odeTriggerName(L"intersection");
        uint class_id(0);
        uint limit(0);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_intersection_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_intersection_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_intersection_new(odeTriggerName.c_str(), NULL, class_id, limit) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Minimum Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Minimum Trigger" ) 
    {
        std::wstring odeTriggerName(L"minimum");
        uint class_id(0);
        uint limit(0);
        uint minimum(2);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_minimum_new(odeTriggerName.c_str(), NULL, class_id, limit, minimum) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_minimum_new(odeTriggerName.c_str(), NULL, class_id, limit, minimum) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_minimum_new(odeTriggerName.c_str(), NULL, class_id, limit, minimum) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Maximum Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Maximum Trigger" ) 
    {
        std::wstring odeTriggerName(L"maximum");
        uint class_id(0);
        uint limit(0);
        uint maximum(5);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_maximum_new(odeTriggerName.c_str(), NULL, class_id, limit, maximum) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_maximum_new(odeTriggerName.c_str(), NULL, class_id, limit, maximum) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_maximum_new(odeTriggerName.c_str(), NULL, class_id, limit, maximum) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Smallest Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Smallest Trigger" ) 
    {
        std::wstring odeTriggerName(L"smallest");
        uint class_id(0);
        uint limit(0);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_smallest_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_smallest_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_smallest_new(odeTriggerName.c_str(), NULL, class_id, limit) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A new Largest Trigger can be created and deleted correctly", "[ode-trigger-api]" )
{
    GIVEN( "Attributes for a new Largest Trigger" ) 
    {
        std::wstring odeTriggerName(L"largest");
        uint class_id(0);
        uint limit(0);

        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_largest_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "The Trigger can be deleted only once" ) 
            {
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND );
            }
        }
        WHEN( "When the Trigger is created" )         
        {
            REQUIRE( dsl_ode_trigger_largest_new(odeTriggerName.c_str(), NULL, class_id, limit) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Trigger with the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_trigger_largest_new(odeTriggerName.c_str(), NULL, class_id, limit) 
                    == DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_trigger_delete(odeTriggerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_trigger_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The ODE Trigger API checks for NULL input parameters", "[ode-trigger-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring triggerName  = L"test-trigger";
        std::wstring otherName  = L"other";
        
        uint claass_id(0);
        const wchar_t* source(NULL);
        boolean enabled(0), infer(0);
        float confidence(0), min_height(0), min_width(0), max_height(0), max_width(0);
        dsl_ode_check_for_occurrence_cb callback;
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_ode_trigger_always_new(NULL, NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_occurrence_new(NULL, NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_absence_new(NULL, NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_intersection_new(NULL, NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_summation_new(NULL, NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_custom_new(NULL, NULL, 0, 0, NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_custom_new(triggerName.c_str(), NULL, 0, 0, NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_custom_new(triggerName.c_str(), NULL, 0, 0, callback, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_minimum_new(NULL, NULL, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_maximum_new(NULL, NULL, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_range_new(NULL, NULL, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_smallest_new(NULL, NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_largest_new(NULL, NULL, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_reset(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_enabled_get(NULL, &enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_enabled_set(NULL, enabled) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_class_id_get(NULL, &claass_id) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_class_id_set(NULL, claass_id) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_source_get(NULL, &source) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_source_set(NULL, source) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_confidence_min_get(NULL, &confidence) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_confidence_min_set(NULL, confidence) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_dimensions_min_get(NULL, &min_width, &min_height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_dimensions_min_set(NULL, min_width, min_height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_dimensions_max_get(NULL, &max_width, &max_height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_dimensions_max_set(NULL, max_width, max_height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_infer_done_only_get(NULL, &infer) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_infer_done_only_set(NULL, infer) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_action_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_add(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_add_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_add_many(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_remove(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_remove_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_remove_many(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_action_remove_all(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_area_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_add(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_add_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_add_many(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_remove(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_remove_many(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_remove_many(triggerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_area_remove_all(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_ode_trigger_delete(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_ode_trigger_delete_many(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );



                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
