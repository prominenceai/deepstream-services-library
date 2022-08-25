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

// ------------------------------------------------------------------------------------
// This example demonstrates interpipe dynamic switching. Four DSL Players
// are created, each with a File Source and Interpipe Sink. A single
// inference Pipeline with an Interpipe Source is created as the single listener
// 
// The Interpipe Source's "listen_to" setting is updated based on keyboard input.
// The xwindow_key_event_handler (see below) is added to the Pipeline's Window Sink.
// The handler, on key release, sets the "listen_to" setting to the Interpipe Sink
// name that corresponds to the key value - 1 through 4.

#include <iostream> 
#include <glib.h>

#include "DslApi.h"

// File path for the single File Source
static const std::wstring uri1(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov");
static const std::wstring uri2(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_push.mov");
static const std::wstring uri3(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov");
static const std::wstring uri4(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov");

// Filespecs for the Primary GIE
static const std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static const std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");


// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// Window Sink Dimensions
uint WINDOW_WIDTH = 1280;
uint WINDOW_HEIGHT = 720;

// 
// Function to be called on XWindow KeyRelease event
// 
static void xwindow_key_event_handler(const wchar_t* in_key, void* client_data)
{   
    std::wcout << L"key released = " << in_key << std::endl;

    std::wstring wkey(in_key); 
    std::string key(wkey.begin(), wkey.end());

    key = std::toupper(key[0]);
    if(key >= "1" and key <= "4")
    {
        std::wstring listen_to = L"inter-pipe-sink-" + wkey;
        dsl_source_interpipe_listen_to_set(L"inter-pipe-source",
            listen_to.c_str());
    } 
    else if (key == "Q")
    {
        std::wcout << L"Pipeline Quit" << std::endl;
        dsl_pipeline_stop(L"pipeline");
        dsl_main_loop_quit();
    }
}

// 
// Function to be called on XWindow Delete event
//
static void xwindow_delete_event_handler(void* client_data)
{
    std::wcout << L"delete window event received " << std::endl;

    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}

// 
// Function to be called on End-of-Stream (EOS) event
// 
static void eos_event_listener(void* client_data)
{
    std::wcout << L"EOS event received" << std::endl;

    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}    

// 
// Function to be called on every change of Pipeline state
// 
static void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::wcout << L"previous state = " << dsl_state_value_to_string(old_state) 
        << L", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

int main(int argc, char** argv)
{  
    DslReturnType retval(DSL_RESULT_SUCCESS);

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // Four new file sources using the filespecs defined above
        retval = dsl_source_file_new(L"file-source-1", uri1.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_file_new(L"file-source-2", uri2.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_file_new(L"file-source-3", uri3.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_file_new(L"file-source-4", uri4.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Four new inter-pipe sinks
        retval = dsl_sink_interpipe_new(L"inter-pipe-sink-1", false, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_interpipe_new(L"inter-pipe-sink-2", false, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_interpipe_new(L"inter-pipe-sink-3", false, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_interpipe_new(L"inter-pipe-sink-4", false, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Four new Players, each with a file source and inter-pipe sink
        retval = dsl_player_new(L"player-1", L"file-source-1", L"inter-pipe-sink-1");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_new(L"player-2", L"file-source-2", L"inter-pipe-sink-2");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_new(L"player-3", L"file-source-3", L"inter-pipe-sink-3");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_new(L"player-4", L"file-source-4", L"inter-pipe-sink-4");
        if (retval != DSL_RESULT_SUCCESS) break;

        //-----------------------------------------------------------------------------
        // Create the Inference Pipeline with an inter-pipe source that can
        // dynamically switch between the four players and their inter-pipe sink
        
        // New inter-pipe source - listen to inter-pipe-sink-1 to start with.
        retval = dsl_source_interpipe_new(L"inter-pipe-source",
            L"inter-pipe-sink-1", false, false, false);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New Primary GIE's using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker setting max width and height of input frame
        retval = dsl_tracker_iou_new(L"iou-tracker", tracker_config_file.c_str(),
            480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"osd", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"inter-pipe-source", 
            L"primary-gie", L"iou-tracker", L"osd", L"window-sink", NULL};
        
        // Create a new Pipeline and add the above components.
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ---------------------------------------------------------------------------
        // Add all Client callback functions to the Pipeline.
        
        // Add the XWindow key event handler callback function.
        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", 
            xwindow_key_event_handler, nullptr);    
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow delete window event handler function.
        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", 
            xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the state-change listener callback function.
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", 
            state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the end of stream (EOS) event listener callback function.
        retval = dsl_pipeline_eos_listener_add(L"pipeline", 
            eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;


        // ---------------------------------------------------------------------------
        // Start the four players first (although, it's safe to start the Pipeline first)

        retval = dsl_player_play(L"player-1");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_play(L"player-2");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_play(L"player-3");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_player_play(L"player-4");
        if (retval != DSL_RESULT_SUCCESS) break;

        // ---------------------------------------------------------------------------
        // Start the Pipeline and join the g-main-loop

        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;
        
        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        
        break;
    }

    // Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout << "Goodbye!" << std::endl;  
    return 0;
}
        