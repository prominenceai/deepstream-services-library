/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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

/*################################################################################
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary GST Inference Engine (PGIE)
#   - DCF Tracker
#   - 2 Secondary GST Inference Engines (SGIEs)
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#   
##############################################################################*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdspreprocess_meta.h>
#include "DslApi.h"

// URI for the File Source
std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Config and model-engine files 
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

std::wstring secondary_infer_config_file_1(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
std::wstring secondary_model_engine_file_1(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet.etlt_b8_gpu0_int8.engine");
        
std::wstring secondary_infer_config_file_2(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehiclemake.txt");

std::wstring secondary_model_engine_file_2(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleMake/resnet18_vehiclemakenet.etlt_b8_gpu0_int8.engine");

// Filespec for the NvDCF Tracker config file   
std::wstring dcf_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml");

// EGL Window Sink Dimensions 
uint WINDOW_WIDTH = DSL_1K_HD_WIDTH / 2;
uint WINDOW_HEIGHT = DSL_1K_HD_HEIGHT / 2;

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
    } else if (key == "Q" or key == "" or key == ""){
        std::cout << "Main Loop Quit" << std::endl;
        dsl_pipeline_stop(L"pipeline");
        dsl_main_loop_quit();
    }
}

// 
// Function to be called on XWindow Delete event
//
void xwindow_delete_event_handler(void* client_data)
{
    std::cout<<"delete window event"<<std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}
    
// 
// Function to be called on End-of-Stream (EOS) event
// 
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
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // New File Source
        retval = dsl_source_file_new(L"file-source-1", uri_h265.c_str(), false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval and Id
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // First new Secondary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_secondary_new(L"secondary-gie-1", 
            secondary_infer_config_file_1.c_str(), secondary_model_engine_file_1.c_str(), 
            L"primary-gie", 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Second new Secondary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_secondary_new(L"secondary-gie-2", 
            secondary_infer_config_file_2.c_str(), secondary_model_engine_file_2.c_str(), 
            L"primary-gie", 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New NvDCF Tracker, setting operation width and height
        // NOTE: width and height paramaters must be multiples of 32 for dcf
        retval = dsl_tracker_new(L"dcf-tracker", 
            dcf_tracker_config_file.c_str(), 640, 384);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Window Sink, 0 x/y offsets.
        retval = dsl_sink_window_egl_new(L"egl-sink", 0, 0, 
            WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add(L"egl-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"egl-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"file-source-1",  L"primary-gie", 
            L"dcf-tracker", L"secondary-gie-1", L"secondary-gie-2",
            L"on-screen-display", L"egl-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the EOS listener function defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the State Change listener function defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", 
            state_change_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Start and join the main-loop
        dsl_main_loop_run();
        break;

    }
    
    // Print out the final result
    std::wcout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
            