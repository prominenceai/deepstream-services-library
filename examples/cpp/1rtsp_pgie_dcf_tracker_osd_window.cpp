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
#   - RTSP Source
#   - Primary GST Inference Engine (PGIE)
#   - DCF Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - error-message events
#   - Pipeline change-of-state events
#   - RTSP Source change-of-state events.
#  
# IMPORTANT! The error-message-handler callback fucntion will stop the Pipeline 
# and main-loop, and then exit. If the error condition is due to a camera
# connection failure, the application could choose to let the RTSP Source's
# connection manager periodically reattempt connection for some length of time.
#
##############################################################################*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdspreprocess_meta.h>
#include "DslApi.h"


// RTSP Source URI for AMCREST Camera    
std::wstring amcrest_rtsp_uri = 
    L"rtsp://username:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";

// RTSP Source URI for HIKVISION Camera    
std::wstring hikvision_rtsp_uri = 
    L"rtsp://username:password@192.168.1.64:554/Streaming/Channels/101";

// Config and model-engine files 
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring dcf_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml");

// IMPORTANT! "DCF Tracker width and height paramaters must be multiples of 32
uint tracker_width = 640;
uint tracker_height = 384;

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
    std::cout << "Pipeline EOS event" << std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}    

//
// Function to be called with every error message received
// by the Pipeline bus manager
//
void error_message_handler(const wchar_t* source, 
    const wchar_t* message, void* client_data)
{    
    std::wcout << L"Error: source = " << source
        << L" message = " << message << std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}

// 
// Function to be called on every change of Pipeline state
// 
void pipeline_state_change_listener(uint old_state, 
    uint new_state, void* client_data)
{
    std::wcout << L"previous state = " << dsl_state_value_to_string(old_state) 
        << L", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

// 
// Function to be called on every change of RTSP Source state
// 
void rtsp_state_change_listener(uint old_state, 
    uint new_state, void* client_data)
{
    std::wcout << L"previous state = " << dsl_state_value_to_string(old_state) 
        << L", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // New RTSP Source for the specific RTSP URI with a timeout of 10s
        // IMPORTANT! a timeout > 0 enables the source's connection management.   
        retval = dsl_source_rtsp_new(L"rtsp-source",     
            hikvision_rtsp_uri.c_str(), // using hikvision URI defined above   
            DSL_RTP_ALL,                // use RTP ALL protocol
            0,                          // skip-frames = 0, decode every frame
            0,                          // drop-frame-interval = 0, decode every frame  
            1000,                       // 1000 ms of jitter buffer
            10);                        // 10 second new buffer timeout   
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval and Id
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New NvDCF Tracker, setting operation width and height
        retval = dsl_tracker_new(L"dcf-tracker", 
            dcf_tracker_config_file.c_str(),
            tracker_width, tracker_height);
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
        const wchar_t* components[] = {L"rtsp-source",  L"primary-gie", 
            L"dcf-tracker", L"on-screen-display", L"egl-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the error-message handler defined above
        retval = dsl_pipeline_error_message_handler_add(L"pipeline", 
            error_message_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the EOS listener function defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the State Change listener function defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", 
            pipeline_state_change_listener, NULL);
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
            
