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
################################################################################
#
# This example shows how to dynamically add and remove Source components
# while the Pipeline is playing. The Pipeline must have at least once source
# while playing. The Pipeline consists of:
#   - A variable number of File Sources. The Source are created/added and 
#       removed/deleted on user key-input.
#   - The Pipeline's built-in streammuxer muxes the streams into a
#       batched stream as input to the Inference Engine.
#   - Primary GST Inference Engine (PGIE).
#   - IOU Tracker.
#   - Multi-stream 2D Tiler - created with rows/cols to support max-sources.
#   - On-Screen Display (OSD)
#   - Window Sink - with window-delete and key-release event handlers.
# 
# A Source component is created and added to the Pipeline by pressing the 
#  "+" key which calls the following services:
#
#    dsl_source_uri_new(source_name, uri_h265, True, 0, 0)
#    dsl_pipeline_component_add('pipeline', source_name)
#
# A Source component (last added) is removed from the Pipeline and deleted by 
# pressing the "-" key which calls the following services
#
#    dsl_pipeline_component_remove('pipeline', source_name)
#    dsl_component_delete(source_name)
#  
################################################################################
*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "DslApi.h"

std::wstring uri_h265(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

std::wstring streammux_config_file = L"../../test/config/all_sources_30fps.txt";

// Config and model-engine files - Jetson and dGPU
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

//
// Maximum number of sources that can be added to the Pipeline
//
uint MAX_SOURCE_COUNT = 8;

//
// Current number of sources added to the Pipeline
//
uint cur_source_count = 0;

//
// Number of rows and columns for the Multi-stream 2D Tiler
//
uint TILER_COLS = 4;
uint TILER_ROWS = 2;

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

    // Add a new source
    } else if (key == "+"){
        if (cur_source_count < MAX_SOURCE_COUNT) {
            cur_source_count += 1;
            std::wstring source_name = L"source-" + std::to_wstring(cur_source_count);
            std::wcout << "adding source '" << source_name << "'" << std::endl;
            dsl_source_uri_new(source_name.c_str(), uri_h265.c_str(), FALSE, 0, 0);
            dsl_pipeline_component_add(L"pipeline", source_name.c_str());
        }
    // Remove the last source added
    } else if (key == "-"){
        if (cur_source_count > 1){
            std::wstring source_name = L"source-" + std::to_wstring(cur_source_count);
            std::wcout << "removing source '" << source_name << "'" << std::endl;
            dsl_pipeline_component_remove(L"pipeline", source_name.c_str());
            dsl_component_delete(source_name.c_str());
            cur_source_count -= 1;
        }
    }
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
//  Client handler for our Stream-Event Pad Probe Handler
// 
uint stream_event_handler(uint stream_event, 
    uint stream_id, void* client_data)
{
    if (stream_event == DSL_PPH_EVENT_STREAM_ADDED) {
        std::wcout << L"Stream Id = " << stream_id 
            << L" added to Pipeline" << std::endl;
    }
    else if (stream_event == DSL_PPH_EVENT_STREAM_ENDED) {
        std::wcout << L"Stream Id = " << stream_id 
            << L" ended" << std::endl;
    }
    else if (stream_event == DSL_PPH_EVENT_STREAM_DELETED) {
        std::wcout << L"Stream Id = " << stream_id 
            << L" deleted from Pipeline" << std::endl;
    }
        
    return DSL_PAD_PROBE_OK;
}

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // First new URI File Source
        retval = dsl_source_uri_new(L"source-1", uri_h265.c_str(), FALSE, 0, 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval = 4
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), 
            primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting operational width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New Tiled Display, setting width and height, 
        retval = dsl_tiler_new(L"tiler", 1280, 720);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Set the Tiled Displays tiles to accommodate max sources.
        retval = dsl_tiler_tiles_set(L"tiler", TILER_COLS, TILER_ROWS);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", TRUE, TRUE, TRUE, FALSE);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Window Sink, 0 x/y offsets.
        retval = dsl_sink_window_egl_new(L"egl-sink", 
            300, 300, 1280, 720);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // IMPORTANT! the default Window-Sink (and Overlay-Sink) "sync" settings must
        // be set to false to support dynamic Pipeline updates.ties.
        retval = dsl_sink_sync_enabled_set(L"egl-sink", FALSE);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add(L"egl-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"egl-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* pipeline_components[] = {
            L"source-1", L"primary-gie", L"iou-tracker", L"tiler", 
            L"on-screen-display", L"egl-sink", NULL};

        // Update the current source count
        cur_source_count = 1;

        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline",
            pipeline_components);
        if (retval != DSL_RESULT_SUCCESS) break;

        // IMPORTANT: we need to explicitely set the Streammuxer batch-size,
        // otherwise the Pipeline will use the current number of Sources when set to 
        // Playing, which would be 1 and too small

        // New Streammux uses config file with overall-min-fps
        if (dsl_info_use_new_nvstreammux_get())
        {
            retval = dsl_pipeline_streammux_config_file_set(L"pipeline",
                streammux_config_file.c_str());
            if (retval != DSL_RESULT_SUCCESS) break;
                
            retval = dsl_pipeline_streammux_batch_size_set(L"pipeline",
                MAX_SOURCE_COUNT);
        }
        // Old Streammux we set the batch-timeout along with the batch-size
        else
        {
            retval = dsl_pipeline_streammux_batch_properties_set(L"pipeline", 
                MAX_SOURCE_COUNT, 40000);
        }
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // Create a Stream-Event Pad Probe Handler (PPH) to manage new Streammuxer 
        // stream-events: stream-added, stream-ended, and stream-deleted.
        retval = dsl_pph_stream_event_new(L"stream-event-pph",
            stream_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the PPH to the source (output) pad of the Pipeline's Streammuxer
        retval = dsl_pipeline_streammux_pph_add(L"pipeline",
            L"stream-event-pph");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_eos_listener_add(L"pipeline", 
            eos_event_listener, NULL);
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
