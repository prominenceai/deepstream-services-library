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

SCENARIO( "The ODE Actions container is updated correctly on multiple new ODE Action", "[ode-action-api]" )
{
    GIVEN( "An empty list of Events" ) 
    {
        std::wstring actionName1(L"log-action-1");
        std::wstring actionName2(L"log-action-2");
        std::wstring actionName3(L"log-action-3");
        
        REQUIRE( dsl_ode_action_list_size() == 0 );

        WHEN( "Several new Actions are created" ) 
        {
            REQUIRE( dsl_ode_action_log_new(actionName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_action_log_new(actionName2.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_ode_action_log_new(actionName3.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_action_list_size() == 3 );

                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The ODE Actions container is updated correctly on Delete ODE Action", "[ode-action-api]" )
{
    GIVEN( "A list of several ODE Actions" ) 
    {
        std::wstring actionName1(L"action-1");
        std::wstring actionName2(L"action-2");
        std::wstring actionName3(L"action-3");
        uint left(0), top(0), width(100), height(100);
        boolean display(true);
        
        REQUIRE( dsl_ode_action_list_size() == 0 );

        REQUIRE( dsl_ode_action_log_new(actionName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_log_new(actionName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_ode_action_log_new(actionName3.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A single Action is deleted" ) 
        {
            REQUIRE( dsl_ode_action_delete(actionName1.c_str()) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_action_list_size() == 2 );

                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "Multiple Actions are deleted" ) 
        {
            const wchar_t* actions[] = {L"action-2", L"action-3", NULL};
            
            REQUIRE( dsl_ode_action_delete_many(actions) == DSL_RESULT_SUCCESS );
            THEN( "The list size and events are updated correctly" ) 
            {
                REQUIRE( dsl_ode_action_list_size() == 1 );

                REQUIRE( dsl_ode_action_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Callback ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Callback ODE Action" ) 
    {
        std::wstring actionName(L"callback-action");
        dsl_ode_handle_occurrence_cb client_handler;

        WHEN( "A new Callback ODE Action is created" ) 
        {
            REQUIRE( dsl_ode_action_callback_new(actionName.c_str(), client_handler, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Callback ODE Action is created" ) 
        {
            REQUIRE( dsl_ode_action_callback_new(actionName.c_str(), client_handler, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "A second callback of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_callback_new(actionName.c_str(), 
                    client_handler, NULL) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Frame Capture ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Frame Capture ODE Action" ) 
    {
        std::wstring actionName(L"capture-action");
        std::wstring outdir(L"./");

        WHEN( "A new Frame Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_frame_new(actionName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Frame Capture Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Frame Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_frame_new(actionName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Frame Capture Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_frame_new(actionName.c_str(), 
                    outdir.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "An invalid Output Directory is specified" ) 
        {
            std::wstring invalidOutDir(L"/invalid/output/directory");
            
            THEN( "A new Frame Capture Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_frame_new(actionName.c_str(), 
                    invalidOutDir.c_str()) == DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Object Capture ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Object Capture ODE Action" ) 
    {
        std::wstring actionName(L"capture-action");
        std::wstring outdir(L"./");

        WHEN( "A new Object Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_object_new(actionName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Object Capture Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Object Capture Action is created" ) 
        {
            REQUIRE( dsl_ode_action_capture_object_new(actionName.c_str(), outdir.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Object Capture Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_object_new(actionName.c_str(), 
                    outdir.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "An invalid Output Directory is specified" ) 
        {
            std::wstring invalidOutDir(L"/invalid/output/directory");
            
            THEN( "A new Object Capture Action fails to create" ) 
            {
                REQUIRE( dsl_ode_action_capture_object_new(actionName.c_str(), 
                    invalidOutDir.c_str()) == DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Display ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Display ODE Action" ) 
    {
        std::wstring actionName(L"display-action");
        boolean offsetY_with_classId(true);

        WHEN( "A new Display Action is created" ) 
        {
            REQUIRE( dsl_ode_action_display_new(actionName.c_str(), 
                10, 10, offsetY_with_classId) == DSL_RESULT_SUCCESS );
            
            THEN( "The Display Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Display Action is created" ) 
        {
            REQUIRE( dsl_ode_action_display_new(actionName.c_str(), 
                10, 10, offsetY_with_classId) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Display Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_display_new(actionName.c_str(), 
                    10, 10, offsetY_with_classId) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Fill Frame ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Fill Frame ODE Action" ) 
    {
        std::wstring actionName(L"fill-frame-action");

        WHEN( "A new Fill Frame Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_frame_new(actionName.c_str(), 0.0, 0.0, 0.0, 0.0) == DSL_RESULT_SUCCESS );
            
            THEN( "The Fill Frame Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Fill Frame Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_frame_new(actionName.c_str(), 0.0, 0.0, 0.0, 0.0) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Fill Frame Action of the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_action_fill_frame_new(actionName.c_str(), 0.0, 0.0, 0.0, 0.0) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Fill Object ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Fill Object ODE Action" ) 
    {
        std::wstring actionName(L"fill-object-action");

        WHEN( "A new Fill Object Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_object_new(actionName.c_str(), 0.0, 0.0, 0.0, 0.0) == DSL_RESULT_SUCCESS );
            
            THEN( "The Fill Object Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Fill Object Action is created" ) 
        {
            REQUIRE( dsl_ode_action_fill_object_new(actionName.c_str(), 0.0, 0.0, 0.0, 0.0) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Fill Object Action of the same name fails to create" ) 
            {
                REQUIRE( dsl_ode_action_fill_object_new(actionName.c_str(), 0.0, 0.0, 0.0, 0.0) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Handler Disable ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Handler Disable ODE Action" ) 
    {
        std::wstring actionName(L"handler-disable-action");
        std::wstring handlerName(L"handler");

        WHEN( "A new Handler Disable Action is created" ) 
        {
            REQUIRE( dsl_ode_action_handler_disable_new(actionName.c_str(), handlerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Handler Disable Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Handler Disable Action is created" ) 
        {
            REQUIRE( dsl_ode_action_handler_disable_new(actionName.c_str(), handlerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Handler Disable Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_handler_disable_new(actionName.c_str(), handlerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Log ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Log ODE Action" ) 
    {
        std::wstring actionName(L"log-action");
        boolean offsetY_with_classId(true);

        WHEN( "A new Log Action is created" ) 
        {
            REQUIRE( dsl_ode_action_log_new(actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Log Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Log Action is created" ) 
        {
            REQUIRE( dsl_ode_action_log_new(actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Log Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_log_new(actionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pause ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Pause ODE Action" ) 
    {
        std::wstring actionName(L"pause-action");
        std::wstring pipelineName(L"pipeline");
        boolean offsetY_with_classId(true);

        WHEN( "A new Pause Action is created" ) 
        {
            REQUIRE( dsl_ode_action_pause_new(actionName.c_str(), pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Pause Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Pause Action is created" ) 
        {
            REQUIRE( dsl_ode_action_pause_new(actionName.c_str(), pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Pause Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_pause_new(actionName.c_str(), pipelineName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Print ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Print ODE Action" ) 
    {
        std::wstring actionName(L"print-action");

        WHEN( "A new Print Action is created" ) 
        {
            REQUIRE( dsl_ode_action_print_new(actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Print Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Print Action is created" ) 
        {
            REQUIRE( dsl_ode_action_print_new(actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Print Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_print_new(actionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Redact ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Redact ODE Action" ) 
    {
        std::wstring actionName(L"redact-action");

        WHEN( "A new Redact Action is created" ) 
        {
            REQUIRE( dsl_ode_action_redact_new(actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Redact Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Redact Action is created" ) 
        {
            REQUIRE( dsl_ode_action_redact_new(actionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Redact Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_redact_new(actionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Sink ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Sink ODE Action" ) 
    {
        std::wstring actionName(L"sink_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sinkName(L"sink");

        WHEN( "A new Add Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_add_new(actionName.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Sink Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Add Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_add_new(actionName.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Add Sink Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_sink_add_new(actionName.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Remove Sink ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Remove Sink ODE Action" ) 
    {
        std::wstring actionName(L"sink_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sinkName(L"sink");

        WHEN( "A new Remove Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_remove_new(actionName.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Remove Sink Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Remove Sink Action is created" ) 
        {
            REQUIRE( dsl_ode_action_sink_remove_new(actionName.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Remove Sink Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_sink_remove_new(actionName.c_str(), pipelineName.c_str(), sinkName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Source ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Source ODE Action" ) 
    {
        std::wstring actionName(L"source_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sourceName(L"source");

        WHEN( "A new Add Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_add_new(actionName.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Source Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Add Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_add_new(actionName.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Add Source Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_source_add_new(actionName.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Remove Source ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Remove Source ODE Action" ) 
    {
        std::wstring actionName(L"source_add-action");
        std::wstring pipelineName(L"pipeline");
        std::wstring sourceName(L"source");

        WHEN( "A new Remove Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_remove_new(actionName.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Remove Source Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Remove Source Action is created" ) 
        {
            REQUIRE( dsl_ode_action_source_remove_new(actionName.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Remove Source Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_source_remove_new(actionName.c_str(), pipelineName.c_str(), sourceName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Reset Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Reset Trigger ODE Action" ) 
    {
        std::wstring actionName(L"trigger-reset-action");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Reset Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_reset_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Reset Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Reset Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_reset_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Reset Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_reset_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Trigger ODE Action" ) 
    {
        std::wstring actionName(L"trigger_add-action");
        std::wstring handlerName(L"handler");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Add Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_add_new(actionName.c_str(), handlerName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Add Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_add_new(actionName.c_str(), handlerName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Add Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_add_new(actionName.c_str(), handlerName.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Disable Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Disable Trigger ODE Action" ) 
    {
        std::wstring actionName(L"trigger_disable-action");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Disable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_disable_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Disable Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Disable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_disable_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Disable Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_disable_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Enable Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Enable Trigger ODE Action" ) 
    {
        std::wstring actionName(L"trigger-enable-action");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Enable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_enable_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Enable Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Enable Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_enable_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Enable Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_enable_new(actionName.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Remove Trigger ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Remove Trigger ODE Action" ) 
    {
        std::wstring actionName(L"trigger_remove-action");
        std::wstring handlerName(L"handler");
        std::wstring triggerName(L"trigger");

        WHEN( "A new Remove Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_remove_new(actionName.c_str(), handlerName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Remove Trigger Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Remove Trigger Action is created" ) 
        {
            REQUIRE( dsl_ode_action_trigger_remove_new(actionName.c_str(), handlerName.c_str(), triggerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Remove Trigger Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_trigger_remove_new(actionName.c_str(), handlerName.c_str(), triggerName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Add Action ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Add Action ODE Action" ) 
    {
        std::wstring actionName(L"action_add-action");
        std::wstring triggerName(L"trigger");
        std::wstring slaveActionName(L"action");

        WHEN( "A new Add Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_add_new(actionName.c_str(), triggerName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Add Action Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Add Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_add_new(actionName.c_str(), triggerName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Add Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_action_add_new(actionName.c_str(), triggerName.c_str(), slaveActionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Disable Action ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Disable Action ODE Action" ) 
    {
        std::wstring actionName(L"action_disable-action");
        std::wstring slaveActionName(L"action");

        WHEN( "A new Disable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_disable_new(actionName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Disable Action Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Disable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_disable_new(actionName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Disable Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_action_disable_new(actionName.c_str(), slaveActionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Enable Action ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Enable Action ODE Action" ) 
    {
        std::wstring actionName(L"action-enable-action");
        std::wstring slaveActionName(L"action");

        WHEN( "A new Enable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_enable_new(actionName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Enable Action Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Enable Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_enable_new(actionName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Enable Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_action_enable_new(actionName.c_str(), slaveActionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Remove Action ODE Action can be created and deleted", "[ode-action-api]" )
{
    GIVEN( "Attributes for a new Remove Action ODE Action" ) 
    {
        std::wstring actionName(L"action_remove-action");
        std::wstring triggerName(L"trigger");
        std::wstring slaveActionName(L"action");

        WHEN( "A new Remove Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_remove_new(actionName.c_str(), triggerName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Remove Action Action can be deleted" ) 
            {
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
        WHEN( "A new Remove Action Action is created" ) 
        {
            REQUIRE( dsl_ode_action_action_remove_new(actionName.c_str(), triggerName.c_str(), slaveActionName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "A second Remove Action Action of the same names fails to create" ) 
            {
                REQUIRE( dsl_ode_action_action_remove_new(actionName.c_str(), triggerName.c_str(), slaveActionName.c_str()) == DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE );
                    
                REQUIRE( dsl_ode_action_delete(actionName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_ode_action_list_size() == 0 );
            }
        }
    }
}
