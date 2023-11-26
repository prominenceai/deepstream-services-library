
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
 * This example demonstrates how to manually control -- using key release and 
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
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
std::wstring file_path2(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4");
std::wstring file_path3(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov");
std::wstring file_path4(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov");

std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine");
std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

int TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH;
int TILER_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT;


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
        
        if (dsl_sink_render_dimensions_get(L"egl-sink", 
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
        // Create two predefined RGBA colors, white and black, that will be
        // used to create text to display the source number on each stream. 
        retval = dsl_display_type_rgba_color_predefined_new(L"full-white", 
            DSL_COLOR_PREDEFINED_WHITE, 1.0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_display_type_rgba_color_predefined_new(L"full-black", 
            DSL_COLOR_PREDEFINED_BLACK, 1.0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_display_type_rgba_font_new(L"arial-18-white", 
            L"arial", 18, L"full-white");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Create a new "source-stream-id" display-type using the new RGBA
        // colors and font created above.
        retval = dsl_display_type_source_stream_id_new(L"source-stream-id", 
            15, 20, L"arial-18-white", true, L"full-black");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Create a new ODE Action to add the display-type's metadata
        // to a frame's meta on invocation.
        retval = dsl_ode_action_display_meta_add_new(L"add-souce-stream-id", 
            L"source-stream-id");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create an ODE Always triger to call the "add-meta" Action to display
        // the source stream-id on every frame for each source. 
        retval = dsl_ode_trigger_always_new(L"always-trigger", 
            DSL_ODE_ANY_SOURCE, DSL_ODE_PRE_OCCURRENCE_CHECK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the Action to new Trigger
        retval = dsl_ode_trigger_action_add(L"always-trigger", 
            L"add-souce-stream-id");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a new ODE Pad Probe Handler (PPH) to add to the Tiler's Src Pad
        retval = dsl_pph_ode_new(L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the Trigger to the ODE PPH which will be added to the Tiler below.
        retval = dsl_pph_ode_trigger_add(L"ode-handler", L"always-trigger");
        if (retval != DSL_RESULT_SUCCESS) break;

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
        retval = dsl_tracker_new(L"iou-tracker", 
            tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Tiler, setting width and height
        retval = dsl_tiler_new(L"tiler", TILER_WIDTH, TILER_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // IMPORTANT! we must explicitly set the number of cols/rows in order to
        // use/call dsl_tiler_source_show_set (see xwindow_key_event_handler)
        retval = dsl_tiler_tiles_set(L"tiler", 2, 2);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_tiler_pph_add(L"tiler", L"ode-handler", DSL_PAD_SINK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text and bbox display enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, false, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_egl_new(L"egl-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_button_event_handler_add(L"egl-sink", 
            xwindow_button_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Enabled the XWindow for full-screen-mode
        retval = dsl_sink_window_fullscreen_enabled_set(L"egl-sink", true);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_key_event_handler_add(L"egl-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"egl-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"uri-source-1", L"uri-source-2", 
            L"uri-source-3", L"uri-source-4", L"primary-gie", L"iou-tracker", 
            L"tiler", L"on-screen-display", L"egl-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the EOS listener and XWindow event handler functions defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
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
        