/*
The MIT License

Copyright (c) 2023, Prominence AI, Inc.

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

/*
#
# This example demonstrates the use of a Frame-Capture Sink to encode and
# save video frames to JPEG files on client/viewer demand.
#
# An ODE Frame-Capture Action is provided to The Frame-Capture Sink on creation.
# A client "capture_complete_listener" is added to the the Action to be notified
# when each new file is saved (the ODE Action performs the actual frame-capture).
#
# Child Players (to play the captured image) and Mailers (to mail the image) can
# be added to the ODE Frame-Capture action as well (not shown).
#
# The "invocation" of a new Frame-Capture is done by pressing the "C" key while 
# the Window Sink has user focus... i.e. the xwindow_key_event_handler will call
# the "dsl_sink_frame_capture_initiate" service on key-event.
#
# IMPORT All captured frames are copied and buffered in the Sink's processing
# thread. The encoding and saving of each buffered frame is done in the 
# g-idle-thread, therefore, the capture-complete notification is asynchronous.
#
*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "DslApi.h"

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Config and model-engine files - Jetson and dGPU
std::wstring primary_infer_config_file_jetson(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
std::wstring primary_model_engine_file_jetson(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_fp16.engine");
std::wstring primary_infer_config_file_dgpu(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file_dgpu(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");


uint PGIE_CLASS_ID_VEHICLE = 0;
uint PGIE_CLASS_ID_BICYCLE = 1;
uint PGIE_CLASS_ID_PERSON = 2;
uint PGIE_CLASS_ID_ROADSIGN = 3;

uint WINDOW_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH;
uint WINDOW_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT;

// 
// Function to be called on XWindow KeyRelease event
// 
void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());
    std::cout << "key released = " << key << std::endl;
    key = std::toupper(key[0]);
    if(key == "C"){
        std::wcout << L"Initiate capture returned" << dsl_return_value_to_string(
            dsl_sink_frame_capture_initiate(L"frame-capture-sink")) << std::endl;
    }
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
    std::cout << "delete window event" <<std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}
    
// 
// Function to be called on End-of-Stream (EOS) event
// 
void eos_event_listener(void* client_data)
{
    std::cout <<"Pipeline EOS event" <<std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}    

// 
// Function to be called on every change of Pipeline state
// 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout << "previous state = " << dsl_state_value_to_string(old_state) 
        << ", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

// 
// Function to be called on Object Capture (and file-save) complete
// 
void capture_complete_listener(dsl_capture_info* info_ptr, 
    void* client_data)
{
    std::cout << "***  Object Capture Complete  ***" << std::endl;
    
    std::wcout << L"capture_id: " << info_ptr->capture_id << std::endl;
    std::wcout << L"filename:   " << info_ptr->filename << std::endl;
    std::wcout << L"dirpath:    " << info_ptr->dirpath << std::endl;
    std::wcout << L"width:      " << info_ptr->width << std::endl;
    std::wcout << L"height:     " << info_ptr->height << std::endl;
}    

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // New File Source
        retval = dsl_source_file_new(L"file-source", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval = 4
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            retval = dsl_infer_gie_primary_new(L"primary-gie", 
                primary_infer_config_file_jetson.c_str(), 
                primary_model_engine_file_jetson.c_str(), 4);
        }
        else
        {
            retval = dsl_infer_gie_primary_new(L"primary-gie", 
                primary_infer_config_file_dgpu.c_str(), 
                primary_model_engine_file_dgpu.c_str(), 4);
        }
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New IOU Tracker, setting operational width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Window Sink, 0 x/y offsets.
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add(L"window-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"window-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Create a new Capture Action to capture and encode frame to jpeg image, 
        // and save to file. Encoding and saving is done in the g-idle-thread.
        // Saving to current directory. File names will be generated as
        //    <action-name>_<unique_capture_id>_<%Y%m%d-%H%M%S>.jpeg
        retval = dsl_ode_action_capture_frame_new(L"frame-capture-action", L"./");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the capture complete listener function to the action
        retval = dsl_ode_action_capture_complete_listener_add(L"frame-capture-action",
            capture_complete_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Frame-Capture Sink created with the new Capture Action.
        retval = dsl_sink_frame_capture_new(L"frame-capture-sink", 
            L"frame-capture-action");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"file-source",  L"primary-gie", 
            L"iou-tracker", L"on-screen-display", L"window-sink", L"frame-capture-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the EOS listener function defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

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
            