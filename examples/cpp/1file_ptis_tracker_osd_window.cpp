
/*
The MIT License

Copyright (c) 2021-2022, Prominence AI, Inc.

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
#include <glib.h>

#include "DslApi.h"

// File path for the single File Source
static const std::wstring file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4");

// Filespecs for the Primary Triton Inference Server (PTIS)
static const std::wstring primary_infer_config_file = 
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt";

// IOU Tracker config file    
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// Window Sink Dimensions
int sink_width = 1280;
int sink_height = 720;

//
// Function to be called on XWindow KeyRelease event
//
void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());
    std::cout << "key released = " << key << std::endl;
    key = std::toupper(key[0]);
    if(key == "P"){
        dsl_pipeline_pause(L"pipeline");
    } else if (key == "R"){
        dsl_pipeline_play(L"pipeline");
    } else if (key == "Q"){
        dsl_pipeline_stop(L"pipeline");
        dsl_main_loop_quit();
    }
}
 
// ## 
// # Function to be called on XWindow Delete event
// ##
void xwindow_delete_event_handler(void* client_data)
{
    std::cout<<"delete window event"<<std::endl;

    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}
    

// # Function to be called on End-of-Stream (EOS) event
void eos_event_listener(void* client_data)
{
    std::cout<<"Pipeline EOS event"<<std::endl;
    
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}
    

// 
// Function to be called on every change of Pipeline state
// 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout<<"previous state = " << dsl_state_value_to_string(old_state) 
        << ", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}


int main(int argc, char** argv)
{  
    DslReturnType retval;

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // # New File Source using the file path specified above, repeat diabled.
        retval = dsl_source_file_new(L"file-source", file_path.c_str(), false);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // # New Primary TIS using the filespec specified above, with interval = 0
        retval = dsl_infer_tis_primary_new(L"primary-tis", primary_infer_config_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New KTL Tracker, setting output width and height of tracked objects
        retval = dsl_tracker_new(L"iou-tracker", tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New OSD with text, clock, bboxs enabled, mask display disabled
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new(L"window-sink", 0, 0, sink_width, sink_height);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add all the components to a new pipeline
        const wchar_t* components[] = { L"file-source",L"primary-tis",
            L"iou-tracker",L"on-screen-display",L"window-sink",nullptr};
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);            
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", 
            xwindow_key_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", 
            xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", 
            state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_pipeline_eos_listener_add(L"pipeline", 
            eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Join with main loop until released - blocking call
        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        break;

    }

    // # Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}


