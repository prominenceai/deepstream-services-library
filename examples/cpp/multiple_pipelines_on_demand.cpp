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

#include <iostream> 
#include <gst/gst.h>

#include "catch.hpp"
#include "DslApi.h"

// File path for the single File Source
static const std::wstring file_path(L"/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_qHD.mp4");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// Window Sink Dimensions
int sink_width = 1280;
int sink_height = 720;

GMutex mutex;
GCond cond;

GThread* main_loop_thread(NULL);

//
// Thread function to start and wait on the main-loop
//
void* main_loop_thread_func(void *data)
{
    // Blocking call - until dsl_main_loop_quit() is called.
    dsl_main_loop_run();

    // Signal the waiting thread that the main-loop has quit
    g_cond_signal(&cond);

    return NULL;
}

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
        pgie = L"pgie-" + std::to_wstring(id);	
        osd = L"osd-" + std::to_wstring(id);	
        window_sink = L"window-sink-" + std::to_wstring(id);	
    }

    std::wstring pipeline;
    std::wstring source;
    std::wstring pgie;
    std::wstring osd;    
    std::wstring window_sink;
};

// Prototypes
DslReturnType create_pipeline(ClientData* client_data);
DslReturnType delete_pipeline(ClientData* client_data);

// 
// Function to be called on XWindow KeyRelease event
// 
void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());
    std::cout << "key released = " << key << std::endl;

    ClientData* c_data = (ClientData*) client_data;

    key = std::toupper(key[0]);
    if(key == "P"){
        dsl_pipeline_pause(c_data->pipeline.c_str());
    } else if (key == "R"){
        dsl_pipeline_play(c_data->pipeline.c_str());
    } else if (key == "Q"){
        std::cout << "Main Loop Quit" << std::endl;
        delete_pipeline(c_data);
    }
}

// 
// Function to be called on XWindow Delete event
//
void xwindow_delete_event_handler(void* client_data)
{
    ClientData* c_data = (ClientData*)client_data; 
    std::cout << "delete window event for Pipeline " 
        << c_data->pipeline.c_str() << std::endl;

    delete_pipeline((ClientData*)client_data);
}

// 
// Function to be called on End-of-Stream (EOS) event
// 
void eos_event_listener(void* client_data)
{
    std::cout << "Pipeline EOS event" << std::endl;

    delete_pipeline((ClientData*)client_data);
}	

// 
// Function to be called on every change of Pipeline state
// 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout<<"previous state = " << dsl_state_value_to_string(old_state) 
        << ", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

DslReturnType create_pipeline(ClientData* client_data)
{
    DslReturnType retval(DSL_RESULT_SUCCESS);

    // New File Source using the same URI for all Piplines
    retval = dsl_source_file_new(client_data->source.c_str(),
        file_path.c_str(), false);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // New Window Sink using the global dimensions
    retval = dsl_sink_window_new(client_data->window_sink.c_str(),
        0, 0, sink_width, sink_height);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    const wchar_t* component_names[] = 
        {client_data->source.c_str(), client_data->window_sink.c_str(), NULL};

    retval = dsl_pipeline_new_component_add_many(client_data->pipeline.c_str(),
        component_names);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the XWindow event handler functions defined above
    retval = dsl_pipeline_xwindow_key_event_handler_add(client_data->pipeline.c_str(), 
        xwindow_key_event_handler, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    retval = dsl_pipeline_xwindow_delete_event_handler_add(client_data->pipeline.c_str(), 
        xwindow_delete_event_handler, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the listener callback functions defined above
    retval = dsl_pipeline_state_change_listener_add(client_data->pipeline.c_str(), 
        state_change_listener, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    retval = dsl_pipeline_eos_listener_add(client_data->pipeline.c_str(), 
        eos_event_listener, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Play the pipeline
    retval = dsl_pipeline_play(client_data->pipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    return retval;    
}

DslReturnType delete_pipeline(ClientData* client_data)
{
    DslReturnType retval(DSL_RESULT_SUCCESS);

    // Stop the pipeline
    retval = dsl_pipeline_stop(client_data->pipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_pipeline_xwindow_clear(client_data->pipeline.c_str());

    // Delete the Pipeline first, then the components. 
    retval = dsl_pipeline_delete(client_data->pipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    const wchar_t* component_names[] = 
        {client_data->source.c_str(), client_data->window_sink.c_str(), NULL};

    // Now safe to delete all components for this Pipeline
    retval = dsl_component_delete_many(component_names);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    return retval;    
}

int main(int argc, char** argv)
{  
    g_mutex_init(&mutex);
	g_cond_init(&cond);

    DslReturnType retval(DSL_RESULT_SUCCESS);

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // We can start the main loop in a seperate thread first
        main_loop_thread = g_thread_new("main-loop", main_loop_thread_func, NULL);

        // Client data for 3 Pipelines
        ClientData client_data_1(1);
        ClientData client_data_2(2);
        ClientData client_data_3(3);

        retval = create_pipeline(&client_data_1);
        if (retval != DSL_RESULT_SUCCESS) break;

        g_usleep(2000000);

        retval = create_pipeline(&client_data_2);
        if (retval != DSL_RESULT_SUCCESS) break;

        g_usleep(2000000);

        dsl_main_loop_quit();

        // Lock the mutex and wait for the main-loop to exit
	    g_mutex_lock(&mutex);
        g_cond_wait(&cond, &mutex);
        g_mutex_unlock(&mutex);

        break;
    }

    // Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

	g_mutex_clear(&mutex);
	g_cond_clear(&cond);

    std::cout << "Goodbye!" << std::endl;  
    return 0;
}
