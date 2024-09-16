/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

/*##############################################################################
#
# This example demonstrates how to run multple Pipelines, each in their own 
# thread, and each with their own main-context and main-loop.
#
# After creating and starting each Pipelines, the script joins each of the 
# threads waiting for them to complete - either by EOS message, 'Q' key, or 
# Delete Window.
#
##############################################################################*/

#include <iostream> 
#include <glib.h>

#include "DslApi.h"

// File path for the single File Source
static const std::wstring file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// Source file dimensions are 960 × 540 
int source_width = 960;
int source_height = 540;

// Window Sink dimensions same as Source dimensions - no scaling.
int sink_width = source_width;
int sink_height = source_height;

GThread* main_loop_thread_1(NULL);
GThread* main_loop_thread_2(NULL);
GThread* main_loop_thread_3(NULL);

uint g_num_active_pipelines = 0;

//     
// Objects of this class will be used as "client_data" for all callback notifications.
// Defines a class of all component names associated with a single Pipeline.     
// The names are derived from the provided unique id     
//    
struct ClientData
{
    ClientData(uint id){
        pipeline = L"pipeline-" + std::to_wstring(id);
        source = L"source-" + std::to_wstring(id);    
        window_sink = L"egl-sink-" + std::to_wstring(id);    
    }

    std::wstring pipeline;
    std::wstring source;
    std::wstring window_sink;
};

// Prototypes
DslReturnType create_pipeline(ClientData* client_data);
DslReturnType delete_pipeline(ClientData* client_data);

// 
// Function to be called on XWindow KeyRelease event
// 
static void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wcout << L"key released = " << in_key << std::endl;

    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());

    ClientData* c_data = (ClientData*) client_data;

    key = std::toupper(key[0]);
    if(key == "P"){
        dsl_pipeline_pause(c_data->pipeline.c_str());
    } else if(key == "S"){
        dsl_pipeline_stop(c_data->pipeline.c_str());
    } else if (key == "R"){
        dsl_pipeline_play(c_data->pipeline.c_str());
    } else if (key == "Q"){
        std::wcout << L"Pipeline Quit" << std::endl;

        // quiting the main loop will allow the pipeline thread to 
        // stop and delete the pipeline and its components
        dsl_pipeline_stop(c_data->pipeline.c_str());
        dsl_pipeline_main_loop_quit(c_data->pipeline.c_str());
    }
}

// 
// Function to be called on XWindow Delete event
//
static void xwindow_delete_event_handler(void* client_data)
{
    ClientData* c_data = (ClientData*)client_data; 
    std::wcout << L"delete window event for Pipeline " 
        << c_data->pipeline.c_str() << std::endl;

    // quiting the main loop will allow the pipeline thread to 
    // stop and delete the pipeline and its components
    dsl_pipeline_stop(c_data->pipeline.c_str());
    dsl_pipeline_main_loop_quit(c_data->pipeline.c_str());
}

// 
// Function to be called on End-of-Stream (EOS) event
// 
static void eos_event_listener(void* client_data)
{
    ClientData* c_data = (ClientData*)client_data; 
    std::wcout << L"EOS event for Pipeline " 
        << c_data->pipeline.c_str() << std::endl;

    // quiting the main loop will allow the pipeline thread to 
    // stop and delete the pipeline and its components
    dsl_pipeline_stop(c_data->pipeline.c_str());
    dsl_pipeline_main_loop_quit(c_data->pipeline.c_str());
}    

// 
// Function to be called on every change of Pipeline state
// 
static void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::wcout << L"previous state = " << dsl_state_value_to_string(old_state) 
        << L", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

DslReturnType create_pipeline(ClientData* client_data)
{
    DslReturnType retval(DSL_RESULT_SUCCESS);

    // New File Source using the same URI for all Piplines
    retval = dsl_source_file_new(client_data->source.c_str(),
        file_path.c_str(), false);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // New Window Sink using the global dimensions
    retval = dsl_sink_window_egl_new(client_data->window_sink.c_str(),
        0, 0, sink_width, sink_height);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    
    // Disable the sync property - which will disable QOS as well.
    retval = dsl_sink_sync_enabled_set(client_data->window_sink.c_str(), false);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the XWindow event handler functions defined above
    retval = dsl_sink_window_key_event_handler_add(client_data->window_sink.c_str(), 
        xwindow_key_event_handler, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    retval = dsl_sink_window_delete_event_handler_add(client_data->window_sink.c_str(), 
        xwindow_delete_event_handler, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    const wchar_t* component_names[] = 
        {client_data->source.c_str(), client_data->window_sink.c_str(), NULL};

    retval = dsl_pipeline_new_component_add_many(client_data->pipeline.c_str(),
        component_names);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the listener callback functions defined above
    retval = dsl_pipeline_state_change_listener_add(client_data->pipeline.c_str(), 
        state_change_listener, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    retval = dsl_pipeline_eos_listener_add(client_data->pipeline.c_str(), 
        eos_event_listener, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    
    // Tell the Pipeline to create its own main-context and main-loop that
    // will be set as the default main-context for the main_loop_thread_func
    // defined below once it is run. 
    retval = dsl_pipeline_main_loop_new(client_data->pipeline.c_str());

    return retval;    
}

DslReturnType delete_pipeline(ClientData* client_data)
{
    DslReturnType retval(DSL_RESULT_SUCCESS);

    std::wcout << L"stoping and deleting Pipeline " 
        << client_data->pipeline.c_str() << std::endl;
        
    // Stop the pipeline
    retval = dsl_pipeline_stop(client_data->pipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Delete the Pipeline first, then the components. 
    retval = dsl_pipeline_delete(client_data->pipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    const wchar_t* component_names[] = 
        {client_data->source.c_str(), client_data->window_sink.c_str(), NULL};

    // Now safe to delete all components for this Pipeline
    retval = dsl_component_delete_many(component_names);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    std::wcout << L"Pipeline " 
        << client_data->pipeline.c_str() << " deleted successfully" << std::endl;
        
    g_num_active_pipelines--;
    
    if (!g_num_active_pipelines)
    {
        dsl_main_loop_quit();
    }
        
    return retval;    
}

//
// Thread function to start and wait on the main-loop
//
void* main_loop_thread_func(void *data)
{
    ClientData* client_data = (ClientData*)data;

    // Play the pipeline
    DslReturnType retval = dsl_pipeline_play(client_data->pipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS) return NULL;

    g_num_active_pipelines++;

    // blocking call
    dsl_pipeline_main_loop_run(client_data->pipeline.c_str());
    
    delete_pipeline(client_data);

    return NULL;
}

int main(int argc, char** argv)
{  
    DslReturnType retval(DSL_RESULT_SUCCESS);

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // Client data for 3 Pipelines
        ClientData client_data_1(1);
        ClientData client_data_2(2);
        ClientData client_data_3(3);

        // Create the first Pipeline and sleep for a second to seperate 
        // the start time with the next Pipeline.
        retval = create_pipeline(&client_data_1);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Start the Pipeline with its own main-context and main-loop in a 
        // seperate thread. 
        main_loop_thread_1 = g_thread_new("main-loop-1", 
            main_loop_thread_func, &client_data_1);

        g_usleep(1000000);

        // Create the second Pipeline. 
        retval = create_pipeline(&client_data_2);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        main_loop_thread_2 = g_thread_new("main-loop-2", 
            main_loop_thread_func, &client_data_2);

        g_usleep(1000000);
        
        // Create the third Pipeline. 
        retval = create_pipeline(&client_data_3);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        main_loop_thread_3 = g_thread_new("main-loop-3", 
            main_loop_thread_func, &client_data_3);

        // All pipelines have been created and set to a state of playing
        // noting more to do so start and join the main-loop
        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        
        break;
    }

    // Print out the final result
    std::wcout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout << "Goodbye!" << std::endl;  
    return 0;
}
