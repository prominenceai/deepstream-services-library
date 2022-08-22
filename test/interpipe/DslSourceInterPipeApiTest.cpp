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
#include "DslSinkBintr.h"
#include "DslSourceBintr.h"
#include "DslPipelineSourcesBintr.h"

static std::wstring inter_pipe_source_name(L"inter-pipe-source");
static std::wstring inter_pipe_sink_name(L"inter-pipe-sink");
static bool is_live(true); 
static bool accept_eos(false); 
static bool accept_events(false);

using namespace DSL;


SCENARIO( "The Components container is updated correctly on new Inter-Pipe Source",
    "[inter-pipe-source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Inter-Pipe Source is created" ) 
        {
            REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name.c_str(), 
                inter_pipe_sink_name.c_str(), is_live, accept_eos, 
                accept_events) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on Inter-Pipe Source delete",
    "[inter-pipe-source-api]" )
{
    GIVEN( "A new Inter-Pipe Source in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name.c_str(), 
            inter_pipe_sink_name.c_str(), is_live, accept_eos, 
            accept_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Inter-Pipe Source is deleted" ) 
        {
            REQUIRE( dsl_component_delete(inter_pipe_source_name.c_str()) == 
                DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Inter-Pipe Source can update its listen-to setting correctly",
    "[inter-pipe-source-api]" )
{
    GIVEN( "A new Inter-Pipe Source in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name.c_str(), 
            inter_pipe_sink_name.c_str(), is_live, accept_eos, 
            accept_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The Inter-Pipe Source's listen-to setting is updated" ) 
        {
            std::wstring new_listen_to_name(L"new-inter-pipe-sink");
            
            REQUIRE( dsl_source_interpipe_listen_to_set(inter_pipe_source_name.c_str(),
                new_listen_to_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct setting is returned on get" )
            {
                const wchar_t* c_listen_to_name;
                REQUIRE( dsl_source_interpipe_listen_to_get(
                    inter_pipe_source_name.c_str(), &c_listen_to_name) == 
                    DSL_RESULT_SUCCESS );
                    
                std::wstring ret_listen_to_name(c_listen_to_name);
                REQUIRE( ret_listen_to_name == new_listen_to_name );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Inter-Pipe Source can update its accept settings correctly",
    "[inter-pipe-source-api]" )
{
    GIVEN( "A new Inter-Pipe Source in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name.c_str(), 
            inter_pipe_sink_name.c_str(), is_live, accept_eos, 
            accept_events) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The Inter-Pipe Source's accept settings are updated" ) 
        {
            boolean new_accept_eos(true), new_accept_events(true);
            
            REQUIRE( dsl_source_interpipe_accept_settings_set(
                inter_pipe_source_name.c_str(), new_accept_eos,
                new_accept_events) == DSL_RESULT_SUCCESS );
            
            THEN( "The correct settings are returned on get" )
            {
                boolean ret_accept_eos(false), ret_accept_events(false);
                REQUIRE( dsl_source_interpipe_accept_settings_get(
                    inter_pipe_source_name.c_str(), &ret_accept_eos,
                    &ret_accept_events) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( ret_accept_eos == new_accept_eos );
                REQUIRE( ret_accept_events == new_accept_events );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Inter-Pipe Source API checks for NULL input parameters", 
    "[inter-pipe-source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            const wchar_t* c_listen_to_name;
            boolean ret_accept_eos(false), ret_accept_events(false);
            
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_source_interpipe_new(NULL, 
                    inter_pipe_sink_name.c_str(), 0, 0, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_interpipe_new(inter_pipe_source_name.c_str(), 
                    NULL, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_source_interpipe_listen_to_get(
                    NULL, &c_listen_to_name) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_interpipe_listen_to_get(
                    inter_pipe_source_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_interpipe_listen_to_set(
                    NULL, inter_pipe_sink_name.c_str()) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_interpipe_listen_to_get(
                    inter_pipe_source_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_source_interpipe_accept_settings_get(
                    NULL, &ret_accept_eos,
                    &ret_accept_events) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_interpipe_accept_settings_get(
                    inter_pipe_source_name.c_str(), NULL,
                    &ret_accept_events) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_interpipe_accept_settings_get(
                    inter_pipe_source_name.c_str(), &ret_accept_eos,
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
            }
        }
    }
}