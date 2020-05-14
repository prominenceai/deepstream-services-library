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

SCENARIO( "The Events container is updated correctly on multiple new Detection Events", "[event-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring eventName1(L"first-occurrence-1");
        std::wstring eventName2(L"first-occurrence-2");
        std::wstring eventName3(L"first-occurrence-3");
        
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_event_list_size() == 0 );

        WHEN( "Several new Events are created" ) 
        {
            REQUIRE( dsl_detection_event_new(eventName1.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_detection_event_new(eventName2.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_detection_event_new(eventName3.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                // TODO complete verification after addition of Iterator API
                REQUIRE( dsl_event_list_size() == 3 );

                REQUIRE( dsl_event_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_event_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Events container is updated correctly on Detection Event deletion", "[event-api]" )
{
    GIVEN( "A list of Events" ) 
    {
        std::wstring eventName1(L"first-occurrence-1");
        std::wstring eventName2(L"first-occurrence-2");
        std::wstring eventName3(L"first-occurrence-3");
        
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_detection_event_new(eventName1.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_detection_event_new(eventName2.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_detection_event_new(eventName3.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        WHEN( "When Events are deleted" )         
        {
            REQUIRE( dsl_event_list_size() == 3 );
            REQUIRE( dsl_event_delete(eventName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_event_list_size() == 2 );

            const wchar_t* eventList[] = {eventName2.c_str(), eventName3.c_str(), NULL};
            REQUIRE( dsl_event_delete_many(eventList) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_event_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A Detection Event's classId can be set/get", "[event-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring eventName(L"first-occurrence");
        
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(9);

        REQUIRE( dsl_detection_event_new(eventName.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        uint ret_class_id(0);
        REQUIRE( dsl_detection_event_class_id_get(eventName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_class_id == class_id );

        WHEN( "When the Event's classId is updated" )         
        {
            uint new_class_id(4);
            REQUIRE( dsl_detection_event_class_id_set(eventName.c_str(), new_class_id) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_detection_event_class_id_get(eventName.c_str(), &ret_class_id) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_class_id == new_class_id );
                REQUIRE( dsl_event_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Detection Event's minimum dimensions can be set/get", "[event-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring eventName(L"first-occurrence");
        
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_detection_event_new(eventName.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        uint min_width(1), min_height(1);
        REQUIRE( dsl_detection_event_dimensions_min_get(eventName.c_str(), &min_width, &min_height) == DSL_RESULT_SUCCESS );
        REQUIRE( min_width == 0 );
        REQUIRE( min_height == 0 );

        WHEN( "When the Event's min dimensions are updated" )         
        {
            uint new_min_width(300), new_min_height(200);
            REQUIRE( dsl_detection_event_dimensions_min_set(eventName.c_str(), new_min_width, new_min_height) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_detection_event_dimensions_min_get(eventName.c_str(), &min_width, &min_height) == DSL_RESULT_SUCCESS );
                REQUIRE( min_width == new_min_width );
                REQUIRE( min_height == new_min_height );
                
                REQUIRE( dsl_event_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Detection Event's minimum frame count can be set/get", "[event-api]" )
{
    GIVEN( "A Detection Event" ) 
    {
        std::wstring eventName(L"first-occurrence");
        
        uint evtype(DSL_EVENT_TYPE_FIRST_OCCURRENCE);
        uint class_id(0);

        REQUIRE( dsl_detection_event_new(eventName.c_str(), evtype, class_id) == DSL_RESULT_SUCCESS );

        uint min_count_n(1), min_count_d(1);
        REQUIRE( dsl_detection_event_frame_count_min_get(eventName.c_str(), &min_count_n, &min_count_d) == DSL_RESULT_SUCCESS );
        REQUIRE( min_count_n == 0 );
        REQUIRE( min_count_d == 0 );

        WHEN( "When the Event's min frame count properties are updated" )         
        {
            uint new_min_count_n(300), new_min_count_d(200);
            REQUIRE( dsl_detection_event_frame_count_min_set(eventName.c_str(), new_min_count_n, new_min_count_d) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_detection_event_frame_count_min_get(eventName.c_str(), &min_count_n, &min_count_d) == DSL_RESULT_SUCCESS );
                REQUIRE( min_count_n == new_min_count_n );
                REQUIRE( min_count_d == new_min_count_d );
                
                REQUIRE( dsl_event_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

