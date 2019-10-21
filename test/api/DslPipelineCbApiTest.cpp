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

#ifndef _DSL_PIPELINE_DBG_API_TEST_H
#define _DSL_PIPELINE_DBG_API_TEST_H

#include "catch.hpp"
#include "DslApi.h"


SCENARIO( "A state-change-listener can be added and removed", "[pipeline]" )
{
    std::string pipelineName = "test-pipeline";
    dsl_state_change_listener_cb listener;

    GIVEN( "A Pipeline in memory" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
    }
    WHEN( "A state-change-listner is added" )
    {
        REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
            listener, (void*)0x12345678) == DSL_RESULT_SUCCESS );

        THEN( "The same listner can't be added again" ) 
        {
            REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
                listener, NULL) == DSL_RESULT_PIPELINE_LISTENER_NOT_UNIQUE );
        }
    }
    WHEN( "A state-change-listner is removed" )
    {
        REQUIRE( dsl_pipeline_state_change_listener_remove(pipelineName.c_str(),
            listener) == DSL_RESULT_SUCCESS );

        THEN( "The same handler can't be removed again" ) 
        {
            REQUIRE( dsl_pipeline_state_change_listener_remove(pipelineName.c_str(),
                listener) == DSL_RESULT_PIPELINE_LISTENER_NOT_FOUND );
        }
    }
    WHEN( "The Pipeline is deleted")
    {
        REQUIRE( dsl_pipeline_delete(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        THEN( "the container is updated correctly")
        {
            REQUIRE( dsl_pipeline_list_size() == 0 );
            REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        }
    }
}

SCENARIO( "An event handler can be added and removed", "[pipeline]" )
{
    std::string pipelineName = "test-pipeline";
    dsl_display_event_handler_cb handler;

    GIVEN( "A Pipeline in memory" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
    }
    WHEN( "An event handler is added" )
    {
        REQUIRE( dsl_pipeline_display_event_handler_add(pipelineName.c_str(),
            handler, (void*)0x12345678) == DSL_RESULT_SUCCESS );

        THEN( "The same handler can't be added again" ) 
        {
            REQUIRE( dsl_pipeline_display_event_handler_add(pipelineName.c_str(),
                handler, NULL) == DSL_RESULT_PIPELINE_HANDLER_NOT_UNIQUE );
        }
    }
    WHEN( "An event handler is removed" )
    {
        REQUIRE( dsl_pipeline_display_event_handler_remove(pipelineName.c_str(),
            handler) == DSL_RESULT_SUCCESS );

        THEN( "The same handler can't be removed again" ) 
        {
            REQUIRE( dsl_pipeline_display_event_handler_remove(pipelineName.c_str(),
                handler) == DSL_RESULT_PIPELINE_HANDLER_NOT_FOUND );
        }
    }
    WHEN( "The Pipeline is deleted")
    {
        REQUIRE( dsl_pipeline_delete(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        THEN( "the container is updated correctly")
        {
            REQUIRE( dsl_pipeline_list_size() == 0 );
            REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        }
    }
}

#endif // _DSL_PIPELINE_DBG_API_TEST_H
