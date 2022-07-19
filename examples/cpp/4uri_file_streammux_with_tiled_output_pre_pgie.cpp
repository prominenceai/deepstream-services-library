
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

/**
 * This example demonstrates how a Pipeline's Stream-Muxer output can be
 * tiled prior to Primary GIE. The Tiler can be set to show an individual source
 * button press events -- the 2D Tiler's output stream to: 
 *   - show a specific source on key input (source No.) or mouse click on tile.
 *   - to return to showing all sources on 'A' key input, mouse click, or timeout.
 *   - to cycle through all sources on 'C' input showing each for timeout.
 * 
 * Note: timeout is controled with the global variable SHOW_SOURCE_TIMEOUT 
 */

#include <iostream>
#include <glib.h>
#include <X11/Xlib.h>

#include "DslApi.h"

// File path for the single File Source
std::wstring file_path1(
    L"/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_1080p_h265.mp4");
std::wstring file_path2(
    L"/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_qHD.mp4");
std::wstring file_path3(
    L"/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_ride_bike.mov");
std::wstring file_path4(
    L"/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_walk.mov");

std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

int TILER_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH;
int TILER_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT;


// Window Sink Dimensions - used to create the sink, however, in this
// example the Pipeline XWindow service is called to enabled full-sreen
int WINDOW_WIDTH = 1280;
int WINDOW_HEIGHT = 720;

int SHOW_SOURCE_TIMEOUT = 3;

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
        dsl_pipeline_stop(L"pipeline");
        dsl_main_loop_quit();
    } else if (key >= "0" and key <= "3"){
        const wchar_t* source;
        
        if (dsl_source_name_get(std::stoi(key), &source) == DSL_RESULT_SUCCESS)
            dsl_tiler_source_show_set(L"tiler", 
                source, SHOW_SOURCE_TIMEOUT, true);
    
    } else if (key == "C"){
        dsl_tiler_source_show_cycle(L"tiler", SHOW_SOURCE_TIMEOUT);

    } else if (key == "A"){
        dsl_tiler_source_show_all(L"tiler");
    }
}
 
//
// Function to be called on XWindow Button Press event
// 
void xwindow_button_event_handler(uint button, 
    int xpos, int ypos, void* client_data)
{
    std::cout << "button = ", button, " pressed at x = ", xpos, " y = ", ypos;
    
    if (button == Button1){
        // get the current XWindow dimensions - the XWindow was overlayed with our Window Sink
        uint width(0), height(0);
        
        if (dsl_pipeline_xwindow_dimensions_get(L"pipeline", 
            &width, &height) == DSL_RESULT_SUCCESS)
            
            // call the Tiler to show the source based on the x and y button cooridantes
            //and the current window dimensions obtained from the XWindow
            dsl_tiler_source_show_select(L"tiler", 
                xpos, ypos, width, height, SHOW_SOURCE_TIMEOUT);
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
        // 4 New File Sources
        retval = dsl_source_file_new(L"uri-source-1", file_path1.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_file_new(L"uri-source-2", file_path2.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_file_new(L"uri-source-3", file_path3.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_file_new(L"uri-source-4", file_path4.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new(L"iou-tracker", 
            tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Tiler, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new(L"tiler", TILER_WIDTH, TILER_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text and bbox display enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, false, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"uri-source-1", L"uri-source-2", 
            L"uri-source-3", L"uri-source-4", L"primary-gie", L"iou-tracker", 
            L"on-screen-display", L"window-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // IMPORTANT! in this example we add the Tiler to the Stream-Muxer's output.
        // The tiled stream is provided as input to the Pirmary GIE
        retval = dsl_pipeline_streammux_tiler_add(L"pipeline", L"tiler");
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // IMPORTANT! explicity set the PGIE batch-size to 1 otherwise the Pipeline will set
        // it to the number of Sources added to the Pipeline.
        retval = dsl_infer_batch_size_set(L"primary-gie", 1);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Enabled the XWindow for full-screen-mode
        retval = dsl_pipeline_xwindow_fullscreen_enabled_set(L"pipeline", true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the EOS listener and XWindow event handler functions defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_xwindow_button_event_handler_add(L"pipeline", 
            xwindow_button_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Start and join the main-loop
        dsl_main_loop_run();
        break;

    }

    // # Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
        