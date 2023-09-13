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

static std::wstring inter_pipe_sink_name(L"inter-pipe-sink");
static bool forward_eos(false); 
static bool forward_events(false);


SCENARIO( "The Components container is updated correctly on new Inter-Pipe Sink",
    "[inter-pipe-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Inter-Pipe Sink is created" ) 
        {
            REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name.c_str(), 
                forward_eos, forward_events) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on Inter-Pipe Sink delete",
    "[inter-pipe-sink-api]" )
{
    GIVEN( "A new Inter-Pipe Sink in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name.c_str(), 
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Inter-Pipe Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(inter_pipe_sink_name.c_str()) == 
                DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Interpipe Sink can update it's common properties correctly", 
    "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sink_name = L"interpipe-sink";

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_interpipe_new(sink_name.c_str(), 
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );

        WHEN( "The Interpipe Sink's sync property is updated from its default" ) 
        {
            boolean newSync(false); // default == true
            REQUIRE( dsl_sink_sync_enabled_set(sink_name.c_str(), 
                newSync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retSync(true);
                REQUIRE( dsl_sink_sync_enabled_get(sink_name.c_str(), 
                    &retSync) == DSL_RESULT_SUCCESS );
                REQUIRE( retSync == newSync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Interpipe Sink's async property is updated from its default" ) 
        {
            boolean newAsync(false);  // default == true
            REQUIRE( dsl_sink_async_enabled_set(sink_name.c_str(), 
                newAsync) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retAsync(true);
                REQUIRE( dsl_sink_async_enabled_get(sink_name.c_str(), 
                    &retAsync) == DSL_RESULT_SUCCESS );
                REQUIRE( retAsync == newAsync );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Interpipe Sink's max-lateness property is updated from its default" ) 
        {
            int64_t newMaxLateness(1);  // default == -1
            REQUIRE( dsl_sink_max_lateness_set(sink_name.c_str(), 
                newMaxLateness) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                int64_t retMaxLateness(12345678);
                REQUIRE( dsl_sink_max_lateness_get(sink_name.c_str(), 
                    &retMaxLateness) == DSL_RESULT_SUCCESS );
                REQUIRE( retMaxLateness == newMaxLateness );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "The Interpipe Sink's qos property is updated from its default" ) 
        {
            boolean newQos(false);  // default == true
            REQUIRE( dsl_sink_qos_enabled_set(sink_name.c_str(), 
                newQos) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                boolean retQos(true);
                REQUIRE( dsl_sink_qos_enabled_get(sink_name.c_str(), 
                    &retQos) == DSL_RESULT_SUCCESS );
                REQUIRE( retQos == newQos );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Inter-Pipe Sink can update its forward settings correctly",
    "[inter-pipe-sink-api]" )
{
    GIVEN( "A new Inter-Pipe Sink in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_sink_interpipe_new(inter_pipe_sink_name.c_str(), 
            forward_eos, forward_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The Inter-Pipe Sink's forward settings are updated" ) 
        {
            boolean new_forward_eos(true), new_forward_events(true);
            
            REQUIRE( dsl_sink_interpipe_forward_settings_set(
                inter_pipe_sink_name.c_str(), new_forward_eos,
                new_forward_events) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct settings are returned on get" )
            {
                boolean ret_forward_eos(false), ret_forward_events(false);
                REQUIRE( dsl_sink_interpipe_forward_settings_get(
                    inter_pipe_sink_name.c_str(), &ret_forward_eos,
                    &ret_forward_events) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( ret_forward_eos == new_forward_eos );
                REQUIRE( ret_forward_events == new_forward_events );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Inter-Pipe Sink API checks for NULL input parameters", 
    "[inter-pipe-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            uint num_listeners;
            boolean ret_forward_eos(false), ret_forward_events(false);
            
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_sink_interpipe_new(NULL, 0, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_sink_interpipe_num_listeners_get(
                    NULL, &num_listeners) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_interpipe_num_listeners_get(
                    inter_pipe_sink_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_interpipe_forward_settings_get(
                    NULL, &ret_forward_eos, &ret_forward_events) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_interpipe_forward_settings_get(
                    inter_pipe_sink_name.c_str(), NULL,
                    &ret_forward_events) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_sink_interpipe_forward_settings_get(
                    inter_pipe_sink_name.c_str(), &ret_forward_eos,
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
            }
        }
    }
}