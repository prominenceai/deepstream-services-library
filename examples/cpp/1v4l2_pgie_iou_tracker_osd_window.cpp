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

/* ##############################################################################
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - V4L2 Source - Web Camera
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#
# The key-release handler function will update the V4L2 device picture settings
# based on the key value as follows during runtime.
#   * brightness - or more correctly the black level. 
#                  enter 'B' to increase, 'b' to decrease
#   * contrast   - color contrast setting or luma gain.
#                  enter 'C' to increase, 'c' to decrease
#   * hue        - color hue or color balence.
#                  enter 'H' to increase, 'h' to decrease
#
# The Picture Settings are all integer values, range 
############################################################################## */

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdspreprocess_meta.h>

#include "DslApi.h"

// Picture settings, read after device negotiation.
int brightness=0;
int contrast=0;
int hue=0;

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Config and model-engine files 
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

uint WINDOW_WIDTH = 1280;
uint WINDOW_HEIGHT = 720;

// 
// Function to be called on XWindow KeyRelease event
// 
void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());
    std::cout << "key released = " << key << std::endl;
    
    bool update(false);
    
    // Upper case 'B' - increase the picture brighness by 10
    if (key == "B")
    {
        brightness += 10;
        update = true;
    }    
    // Lower case 'b' - decrease the picture brighness by 10
    else if (key == "b")
    {
        brightness -= 10;
        update = true;
    }    
    // Upper case 'C' - increase the picture contrast (luma gain) by 10
    else if (key == "C")
    {
        contrast += 10;
        update = true;
    }   
    // Lower case 'c' - decrease the picture contrast (luma gain) by 10
    else if (key == "c")
    {
        contrast -= 10;
        update = true;
    }
    // Upper case 'H' - increase the picture hue (color balence) by 10
    else if (key == "H")
    {
        hue += 10;
        update = true;
    }
    // Lower case 'h' - decrease the picture hue (color balence) by 10
    else if (key == "h")
    {
        hue -= 10;
        update = true;
    }
    if (update)
    {
        dsl_source_v4l2_picture_settings_set(L"v4l2-source",
            brightness, contrast, hue);
        return;    
    }
    
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

        // New V4L2 Live Web Camera Source
        retval = dsl_source_v4l2_new(L"v4l2-source", L"/dev/video0");
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval and Id
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting operational width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New 3D Window Sink with 0 x/y offsets, and same dimensions as Camera Source
        // EGL Sink runs on both platforms. 3D Sink is Jetson only.
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            retval = dsl_sink_window_3d_new(L"window-sink", 0, 0, 
                WINDOW_WIDTH, WINDOW_HEIGHT);            
        }
        else
        {
            retval = dsl_sink_window_egl_new(L"window-sink", 0, 0, 
                WINDOW_WIDTH, WINDOW_HEIGHT);            
        }
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add(L"window-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"window-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"v4l2-source",  L"primary-gie", 
            L"iou-tracker", L"on-screen-display", L"window-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the EOS listener function defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // Once playing, we can retrieve the device information: 
        // name, file-descriptor, and device-flags
        const wchar_t* device_name;
        retval = dsl_source_v4l2_device_name_get(L"v4l2-source", &device_name);
        if (retval != DSL_RESULT_SUCCESS) break;

        int device_fd;
        retval = dsl_source_v4l2_device_fd_get(L"v4l2-source", &device_fd);
        if (retval != DSL_RESULT_SUCCESS) break;

        uint device_flags;
        retval = dsl_source_v4l2_device_flags_get(L"v4l2-source", &device_flags);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Get the initial values updated after the Pipeline is playing.
        retval = dsl_source_v4l2_picture_settings_get(L"v4l2-source",
            &brightness, &contrast, &hue);
        if (retval != DSL_RESULT_SUCCESS) break;

        std::wcout << L"V4L2 Device Propertes" << std::endl;
        std::wcout << L"   Name       : " << device_name << std::endl;
        std::wcout << L"   File Desc  : " << device_fd << std::endl;
        std::wcout << L"   Flags      : " << device_flags << std::endl;
        std::wcout << L"   Brightness : " << brightness << std::endl;
        std::wcout << L"   Contrast   : " << contrast << std::endl;
        std::wcout << L"   Hue        : " << hue << std::endl;

        // Start and join the main-loop
        dsl_main_loop_run();
        break;

    }
    
    // Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
            