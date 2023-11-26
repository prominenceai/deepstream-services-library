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
# This example shows how to use a single dynamic demuxer branch with a 
# multi-source Pipeline. The Pipeline trunk consists of:
#   - 5 Streaming Images Sources - each streams a single image at a given 
#       frame-rate with a number overlayed representing the stream-id.
#   - The Pipeline's built-in streammuxer muxes the streams into a
#       batched stream as input to the Inference Engine.
#   - Primary GST Inference Engine (PGIE).
#   - IOU Tracker.
#
# The dynamic branch will consist of:
#   - On-Screen Display (OSD)
#   - Window Sink - with window-delete and key-release event handlers.
# 
# The branch is added to one of the Streams when the Pipeline is constructed
# by calling:
#
#    dsl_tee_demuxer_branch_add_to('demuxer', 'branch-0', stream_id)
#
# Once the Pipeline is playing, the example uses a simple periodic timer to 
# call a callback function which advances/cycles the current stream_id 
# variable and moves the branch by calling.
#
#    dsl_tee_demuxer_branch_move_to('demuxer', 'branch-0', stream_id)
#
################################################################################
*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "DslApi.h"

// Each png file has a unique id [0..4] overlayed on the picture
// This makes it easy to see which stream the branch is connected to.
std::wstring image_0(L"../../test/streams/sample_720p.0.png");
std::wstring image_1(L"../../test/streams/sample_720p.1.png");
std::wstring image_2(L"../../test/streams/sample_720p.2.png");
std::wstring image_3(L"../../test/streams/sample_720p.3.png");
std::wstring image_4(L"../../test/streams/sample_720p.4.png");

// Config and model-engine files - Jetson and dGPU
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

/*
# Global variable - indicates the current stream_id for the 
# single branch (branch-0). The variable will be updated as
#     stream_id = (stream_id+1)%num_streams
# to cycle through each of the source 4 streams. The other branch,
# which is a sole window sink, is connected to the 5th stream
# (stream_id=4) at all times.
*/
static uint stream_id(0);

/*
# Number of Source streams for the Pipeline
*/
static uint num_streams(5);

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
    }
}
//
// Callback function to move the dynamic branch (branch-0) from
// its current stream to next stream-id in the cycle.
//
static int move_branch(void* user_data)
{
    // set the stream-id to the next stream in the cycle of 5.
    stream_id = (stream_id+1)%num_streams;
    
    // we then call the Demuxer service to add it back at the specified stream-id
    std::wcout << L"dsl_tee_demuxer_branch_move_to() returned "
    << dsl_return_value_to_string(
        dsl_tee_demuxer_branch_move_to(L"demuxer", L"branch-0", stream_id)) 
        << std::endl;
        
    return true;
}

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // ----------------------------------------------------------------------------
        // Create the five streaming-image sources that will provide the streams for 
        // the single dynamic branch.
        // IMPORTANT! see Demuxer blocking-timeout notes below if setting a 
        // frame-rate that is less than 2 fps.
        
        retval = dsl_source_image_stream_new(L"source-0", 
            image_0.c_str(), TRUE, 10, 1, 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_image_stream_new(L"source-1", 
            image_1.c_str(), TRUE, 10, 1, 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_image_stream_new(L"source-2", 
            image_2.c_str(), TRUE, 10, 1, 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_image_stream_new(L"source-3", 
            image_3.c_str(), TRUE, 10, 1, 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_image_stream_new(L"source-4", 
            image_4.c_str(), TRUE, 15, 1, 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------
        // Create the PGIE and Tracker components that will become the
        // fixed pipeline-trunk to process all batched sources. The Demuxer will
        // demux/split the batched streams back to individual source streams.
        
        // New Primary GIE using the filespecs defined above, with interval = 4
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), 
            primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting operational width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------
        // Next, create the OSD and Overlay-Sink. These two components will make up
        // the dynamic branch. The dynamic branch will be moved from stream to stream
        // i.e. from demuxer pad to pad.

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

        // Create a list of dynamic branch components.
       const wchar_t* branch_components[] = {
            L"on-screen-display", L"egl-sink", NULL};
            
        // Add the branch components to a new branch
        retval = dsl_branch_new_component_add_many(L"branch-0",
            branch_components);
        if (retval != DSL_RESULT_SUCCESS) break;

        // IMPORTANT! when creating the Demuxer, we need to set the maximum number
        // of branches equal to the number of Source Streams, even though we will 
        // only be using one branch. The Demuxer needs to allocate a source-pad
        // for each stream prior to playing so that the dynamic Branch can be 
        // moved from stream to stream while the Pipeline is in a state of PLAYING.
        retval = dsl_tee_demuxer_new(L"demuxer", num_streams);
        if (retval != DSL_RESULT_SUCCESS) break;

        // IMPORTANT! all tees use a blocking-timeout to manage the process of
        // dynamically adding, removing, or moving (demuxer only) a branch.
        // A blocking pad-probe is used block data at the Tee's source pad while 
        // the branch is added or removed. The process requires the stream to be 
        // in a state of Playing. If not, the pad-probe-handler (callback) to 
        // complete this process would block indefinitely if not for the timeout. 
        // This can occur if the source has been paused or dynamically removed.
        // The default timeout is set to DSL_TEE_DEFAULT_BLOCKING_TIMEOUT_IN_SEC(1)
        // and must be increased if running at slow frame rates < 2fps or the 
        // process may timeout before the blocking-pph-handler can be called.

        // Update the default timeout if the sources are running at a slower fps.
//        retval = dsl_tee_blocking_timeout_set(L"demuxer", 3);
//        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the branch to the Demuxer at stream_id=0
        retval = dsl_tee_demuxer_branch_add_to(L"demuxer", L"branch-0", stream_id);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* pipeline_components[] = {
            L"source-0", L"source-1", L"source-2", L"source-3", L"source-4",
            L"primary-gie", L"iou-tracker", L"demuxer", NULL};

        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline",
            pipeline_components);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline"); 
        if (retval != DSL_RESULT_SUCCESS) break;

        g_timeout_add(6000, move_branch, NULL);

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
            