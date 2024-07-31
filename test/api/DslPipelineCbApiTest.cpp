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

static void state_change_listener_cb(uint prev_state, 
    uint curr_state, void* client_data)
{
}

static void eos_listener_cb(void* client_data)
{
}

static void error_message_handler(const wchar_t* source, 
    const wchar_t* message, void* client_data)
{
}

static void buffering_message_handler(const wchar_t* source, 
    uint percentage, void* client_data)
{
}

SCENARIO( "A state-change-listener must be unique", "[pipeline-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A Pipeline in memory" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A state-change-listener is added" )
        {
            REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
                state_change_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            THEN( "The same listener can't be added again" ) 
            {
                REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
                    state_change_listener_cb, NULL) == DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A state-change-listener can be removed", "[pipeline-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A Pipeline with one state-change-listener" )
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_state_change_listener_add(pipelineName.c_str(),
            state_change_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

        WHEN( "A state-change-listener is removed" )
        {
            REQUIRE( dsl_pipeline_state_change_listener_remove(pipelineName.c_str(),
                state_change_listener_cb) == DSL_RESULT_SUCCESS );

            THEN( "The same handler can't be removed again" ) 
            {
                REQUIRE( dsl_pipeline_state_change_listener_remove(pipelineName.c_str(),
                    state_change_listener_cb) == DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A EOS-listener must be unique", "[pipeline-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A Pipeline in memory" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A EOS-listener is added" )
        {
            REQUIRE( dsl_pipeline_eos_listener_add(pipelineName.c_str(),
                eos_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            THEN( "The same listener can't be added again" ) 
            {
                REQUIRE( dsl_pipeline_eos_listener_add(pipelineName.c_str(),
                    eos_listener_cb, NULL) == DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A EOS-listener can be removed", "[pipeline-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A Pipeline with one EOS-listener" )
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_eos_listener_add(pipelineName.c_str(),
            eos_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

        WHEN( "A EOS-listener is removed" )
        {
            REQUIRE( dsl_pipeline_eos_listener_remove(pipelineName.c_str(),
                eos_listener_cb) == DSL_RESULT_SUCCESS );

            THEN( "The same handler can't be removed again" ) 
            {
                REQUIRE( dsl_pipeline_eos_listener_remove(pipelineName.c_str(),
                    eos_listener_cb) == DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An error-message-handler can be added and removed", "[pipeline-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";
    
    GIVEN( "A Pipeline in memory" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "An error message handler is added" )
        {
            REQUIRE( dsl_pipeline_error_message_handler_add(pipelineName.c_str(),
                error_message_handler, NULL) == DSL_RESULT_SUCCESS );

            // calling a second time must fail
            REQUIRE( dsl_pipeline_error_message_handler_add(pipelineName.c_str(),
                error_message_handler, NULL) == DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED );

            THEN( "The same handler can be removed" ) 
            {
                REQUIRE( dsl_pipeline_error_message_handler_remove(pipelineName.c_str(),
                    error_message_handler) == DSL_RESULT_SUCCESS );
                
                // second call must fail
                REQUIRE( dsl_pipeline_error_message_handler_remove(pipelineName.c_str(),
                    error_message_handler) == DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "An buffering-message-handler can be added and removed", "[pipeline-cb-api]" )
{
    std::wstring pipelineName = L"test-pipeline";
    
    GIVEN( "A Pipeline in memory" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "An buffering message handler is added" )
        {
            REQUIRE( dsl_pipeline_buffering_message_handler_add(pipelineName.c_str(),
                buffering_message_handler, NULL) == DSL_RESULT_SUCCESS );

            // calling a second time must fail
            REQUIRE( dsl_pipeline_buffering_message_handler_add(pipelineName.c_str(),
                buffering_message_handler, NULL) == DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED );

            THEN( "The same handler can be removed" ) 
            {
                REQUIRE( dsl_pipeline_buffering_message_handler_remove(pipelineName.c_str(),
                    buffering_message_handler) == DSL_RESULT_SUCCESS );
                
                // second call must fail
                REQUIRE( dsl_pipeline_buffering_message_handler_remove(pipelineName.c_str(),
                    buffering_message_handler) == DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Pipeline Callback API checks for NULL input parameters", "[pipeline-cb-api]" )
{
    GIVEN( "An empty list of Pipelines" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_pipeline_state_change_listener_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_state_change_listener_add(pipeline_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_state_change_listener_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_eos_listener_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_eos_listener_add(pipeline_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_eos_listener_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_pipeline_error_message_handler_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_error_message_handler_add(pipeline_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_error_message_handler_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_pipeline_buffering_message_handler_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_buffering_message_handler_add(pipeline_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_buffering_message_handler_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
            }
        }
    }
}