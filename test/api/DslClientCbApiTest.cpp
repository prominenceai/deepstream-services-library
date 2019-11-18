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

#define LISTENER_1_DATA 0x11111111
#define LISTENER_2_DATA 0x22222222

void StateChangeListener1(uint oldstate, uint newstate, void* userdata)
{
    REQUIRE(userdata == (void*)LISTENER_1_DATA);
}

void StateChangeListener2(uint oldstate, uint newstate, void* userdata)
{
    REQUIRE(userdata == (void*)LISTENER_2_DATA);
}

SCENARIO( "All state-change-listeners are called on change of state", "[client-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";
    std::wstring sourceName  = L"csi-source";

    GIVEN( "A Pipeline with two state-change-listeners" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
            (dsl_state_change_listener_cb)StateChangeListener1, (void*)LISTENER_1_DATA) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
            (dsl_state_change_listener_cb)StateChangeListener2, (void*)LISTENER_2_DATA) 
            == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is requested to change state" )
        {
    //        REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            // ** NOTE ** This test is incomplete... the Pipeline must transition into 
            // a newstate successfully
            // TODO - need to sync with callbacks before proceeding
        }

        REQUIRE( dsl_pipeline_state_change_listener_remove(pipelineName.c_str(),
            (dsl_state_change_listener_cb)StateChangeListener1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_state_change_listener_remove(pipelineName.c_str(),
            (dsl_state_change_listener_cb)StateChangeListener2) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_delete(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( *(dsl_pipeline_list_all()) == NULL );
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( *(dsl_component_list_all()) == NULL );
    }
}
