
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

/*
This example illustrates how to push raw video buffers to a DSL Pipeline
using an App Source component. The example application adds the following
client handlers to control the input of raw buffers to the App Source
  * need_data_handler   - called when the App Source needs data to process
  * enough_data_handler - called when the App Source has enough data to process

The client handlers add/remove a callback function to read, map, and push data
to the App Source called "read_and_push_data". 

The raw video file used with this example is created by executing the following 
gst-launch-1.0 command.

gst-launch-1.0 uridecodebin \
      uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
      ! nvvideoconvert ! 'video/x-raw, format=I420, width=1280, height=720' \
      ! filesink location=./sample_720p.i420
*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>

#include "DslApi.h"

// Local raw file generated with the command ablove
static const std::string raw_file("./sample_720p.i420");

// Filespecs for the Primary Triton Inference Server (PTIS)
static const std::wstring primary_infer_config_file = 
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt";

// IOU Tracker config file    
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// Source dimensions for the raw video
uint source_width = 1280;
uint source_height = 720;

// Rate to push frames into the DSL Pipeline
uint fps_n = 30;
uint fps_d = 1;

// Window Sink Dimensions
uint sink_width = 1280;
uint sink_height = 720;

// To enable presentation timestamp
#define CUSTOM_PTS TRUE

//
// Structure to contain all our information for appsrc, so we can pass it to callbacks
//
struct AppSrcData
{
    FILE* file;         // Pointer to the raw video file
    long  frame_size;
    gint  frame_num;
    guint fps;          // To set the FPS value */
    guint sourceid;     // To control the GSource */
};

//
// This callback is called by the idle GSource in the mainloop 
// to feed one raw video frame into appsrc. The function will
// be rescheduled for execution if TRUE is returned.
// The idle handler is added to the mainloop when appsrc requests us
// to start sending data (need-data signal) and is removed when appsrc 
// has enough data (enough-data signal).
//
static boolean read_and_push_data(void* client_data)
{
    AppSrcData* data = (AppSrcData*)client_data;
    
    // Allocate a new buffer and map it.
    GstBuffer* buffer = gst_buffer_new_allocate(NULL, data->frame_size, NULL);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);

    // Read a frame from the raw data file
    size_t size = fread(map.data, 1, data->frame_size, data->file);
    map.size = size;

    // Safe to unmap the buffer now before pushing to the App Source
    gst_buffer_unmap (buffer, &map);
    
    // if we have data to push to the App Source
    if (size > 0)
    {
        
// To enable presentation timestamp
#if CUSTOM_PTS
        GST_BUFFER_PTS(buffer) =
            gst_util_uint64_scale(data->frame_num, GST_SECOND, data->fps);
#endif

        DslReturnType retval = dsl_source_app_buffer_push(L"app-source", buffer);
            
        if (retval != DSL_RESULT_SUCCESS)
        {
            std::cout<< "'dsl_source_app_buffer_push' returned "
                << dsl_return_value_to_string(retval) << std::endl;
            return FALSE;
        }
    } 
    
    // else, all frames have been read from the file so end-of-stream the App Source
    else if (size == 0) 
    {
        DslReturnType retval = dsl_source_app_eos(L"app-source");
            
        if (retval != DSL_RESULT_SUCCESS)
        {
            std::cout<< "'dsl_source_app_eos' returned "
                << dsl_return_value_to_string(retval) << std::endl;
        }
        
        // Return false to unschedule this callback.
        return FALSE;
    }
    else 
    {
        std::cout<< "ERROR failed to read from raw data file" << std::endl;
        return FALSE;
    }
    data->frame_num++;

  return TRUE;
}


//
// This signal callback triggers when appsrc needs data. Here,
// we add an idle handler to the mainloop to start pushing
// data into the appsrc
//
void need_data_handler(uint length, void* client_data)
{
    AppSrcData* data = (AppSrcData*)client_data;

    std::cout << "'need-data' handler called with length = "
        << length << std::endl;
        
    if (data->sourceid == 0)
    {
        data->sourceid = g_idle_add((GSourceFunc)read_and_push_data, data);
    }
}

//
// This callback triggers when appsrc has enough data and we can stop sending.
// We remove the idle handler from the mainloop.
//
void enough_data_handler(void* client_data)
{
    AppSrcData* data = (AppSrcData*)client_data;

    std::cout << "'enough-data' handler called" << std::endl;
    
    if (data->sourceid != 0)
    {
        g_source_remove(data->sourceid);
        data->sourceid = 0;
    }
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
    } else if (key == "Q"){
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

// # Function to be called on End-of-Stream (EOS) event
void eos_event_listener(void* client_data)
{
    std::cout << "Pipeline EOS event" <<std::endl;
    
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

int main(int argc, char** argv)
{  
    DslReturnType retval;

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // Initialize the App Source client data. 
        AppSrcData data;
        data.frame_size = source_width * source_height * 1.5; // for I420
        data.file = fopen(raw_file.c_str(), "r");
        data.frame_num = 0;
        data.fps = fps_n/fps_d;
        data.sourceid = 0;
        
        // New App Source with is_live = false, format = I420
        retval = dsl_source_app_new(L"app-source", false, 
            DSL_STREAM_FORMAT_I420, source_width, source_height, fps_n, fps_d);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // 
        retval = dsl_source_app_data_handlers_add(L"app-source",
            need_data_handler, enough_data_handler, (void*)&data);
        if (retval != DSL_RESULT_SUCCESS) break;

// To enable presentation timestamp
#if CUSTOM_PTS
        retval = dsl_source_do_timestamp_set(L"app-source", TRUE);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_source_app_buffer_format_set(L"app-source",
            DSL_BUFFER_FORMAT_TIME);
        if (retval != DSL_RESULT_SUCCESS) break;
#endif            
        // New Primary TIS using the filespec specified above, with interval = 0
        retval = dsl_infer_tis_primary_new(L"primary-tis", primary_infer_config_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting output width and height of tracked objects
        retval = dsl_tracker_new(L"iou-tracker", tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock, bboxs enabled, mask display disabled
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new(L"window-sink", 0, 0, sink_width, sink_height);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add all the components to a new pipeline
        const wchar_t* components[] = { L"app-source",L"primary-tis",
            L"iou-tracker",L"on-screen-display",L"window-sink",nullptr};
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);            
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", 
            xwindow_key_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", 
            xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", 
            state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_pipeline_eos_listener_add(L"pipeline", 
            eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Join with main loop until released - blocking call
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
