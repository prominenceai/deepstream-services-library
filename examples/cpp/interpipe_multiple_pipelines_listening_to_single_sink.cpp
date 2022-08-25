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

//-------------------------------------------------------------------------------------
//
// This script demonstrates how to run multple Pipelines, each with an Interpipe
// Source, both listening to the same Interpipe Sink.
//
// A single Player is created with a File Source and Interpipe Sink. Two Inference
// Pipelines are created to listen to the single Player. 
//
// The two Pipelines can be created with different configs, models, and/or Trackers
// for side-by-side comparison. Both Pipelines run in their own main-loop with their 
// own main-context and have their own Window Sink for viewing and external control.

#include <iostream> 
#include <glib.h>

#include "DslApi.h"

// File path for the single File Source
static const std::wstring file_path(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4");

// Filespecs for the Primary GIE
static const std::wstring primary_infer_config_file_1(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static const std::wstring primary_infer_config_file_2(
    L"../../test/configs/config_infer_primary_nano_nms_test.txt");
    
    
static const std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// Window Sink Dimensions
uint sink_width = 1280;
uint sink_height = 720;

GThread* main_loop_thread_1(NULL);
GThread* main_loop_thread_2(NULL);

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
        pgie = L"pgie-" + std::to_wstring(id);
        tracker = L"tracker-" + std::to_wstring(id);
        osd = L"osd-" + std::to_wstring(id);
        window_sink = L"window-sink-" + std::to_wstring(id);    
    }

    std::wstring pipeline;
    std::wstring source;
    std::wstring pgie;
    std::wstring tracker;
    std::wstring osd;
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

    // New OSD with text, clock and bbox display all enabled. 
    retval = dsl_osd_new(client_data->osd.c_str(), true, true, true, false);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // New Window Sink using the global dimensions
    retval = dsl_sink_window_new(client_data->window_sink.c_str(),
        0, 0, sink_width, sink_height);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    const wchar_t* component_names[] = 
    {
        client_data->source.c_str(), client_data->osd.c_str(), 
        client_data->window_sink.c_str(), NULL
    };

    retval = dsl_pipeline_new_component_add_many(client_data->pipeline.c_str(),
        component_names);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the XWindow event handler functions defined above
    retval = dsl_pipeline_xwindow_key_event_handler_add(client_data->pipeline.c_str(), 
        xwindow_key_event_handler, (void*)client_data);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    
    retval = dsl_pipeline_xwindow_delete_event_handler_add(
        client_data->pipeline.c_str(), xwindow_delete_event_handler, 
        (void*)client_data);
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
        // New file source using the filespath defined above
        retval = dsl_source_file_new(L"file-source", file_path.c_str(), false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New interpipe sink to broadcast to all listeners (interpipe sources).
        retval = dsl_sink_interpipe_new(L"interpipe-sink", true, true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Player to play the file source with interpipe sink
        retval = dsl_player_new(L"player", L"file-source", L"interpipe-sink");
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Client data for two Pipelines
        ClientData client_data_1(1);
        ClientData client_data_2(2);

        // --------------------------------------------------------------------------
        // Create the first Pipeline with common components 
        // - interpipe source, OSD, and window sink,
        // - then add PGIE and Tracker using first set of configs
        
        retval = create_pipeline(&client_data_1);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the first config file. 
        retval = dsl_infer_gie_primary_new(client_data_1.pgie.c_str(),
            primary_infer_config_file_1.c_str(), primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new(client_data_1.tracker.c_str(), 
            tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        const wchar_t* additional_components_1[] = 
            {client_data_1.pgie.c_str(), client_data_1.tracker.c_str(), NULL};
        
        // Add the new components to the first Pipeline
        retval = dsl_pipeline_component_add_many(client_data_1.pipeline.c_str(),
            additional_components_1);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Start the Pipeline with its own main-context and main-loop in a 
        // seperate thread. 
        main_loop_thread_1 = g_thread_new("main-loop-1", 
            main_loop_thread_func, &client_data_1);

        // --------------------------------------------------------------------------
        // Create the Second Pipeline with common components 
        // - interpipe source, OSD, and window sink,
        // - then add PGIE and Tracker using second set of configs
        
        retval = create_pipeline(&client_data_2);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the second config file. 
        retval = dsl_infer_gie_primary_new(client_data_2.pgie.c_str(),
            primary_infer_config_file_2.c_str(), primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new(client_data_2.tracker.c_str(), 
            tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        const wchar_t* additional_components_2[] = 
            {client_data_2.pgie.c_str(), client_data_2.tracker.c_str(), NULL};
        
        // Add the new components to the second Pipeline
        retval = dsl_pipeline_component_add_many(client_data_2.pipeline.c_str(),
            additional_components_2);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // Start the Pipeline with its own main-context and main-loop in a 
        // seperate thread. 
        main_loop_thread_2 = g_thread_new("main-loop-2", 
            main_loop_thread_func, &client_data_2);

        // --------------------------------------------------------------------------
        // Once the Pipelines are running we can start the player - i.e common stream
        retval = dsl_player_play(L"player");
        
        // Join both threads - in any order.
        g_thread_join(main_loop_thread_1);
        g_thread_join(main_loop_thread_2);
        
        break;
    }

    // Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout << "Goodbye!" << std::endl;  
    return 0;
}
