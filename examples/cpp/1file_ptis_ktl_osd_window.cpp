
// 1file_ptis_ktl_osd_window.py 

// #-------------------------------------------------------------------------------------------
// #
// # This script demonstrates the use of a Primary Triton Inference Server (PTIS). The PTIS
// # requires a unique name, TIS inference config file, and inference interval when created.
// #
// # The PTIS is added to a new Pipeline with a single File Source, KTL Tracker, 
// # On-Screen-Display (OSD), and Window Sink with 1280x720 dimensions.


#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <gtkmm.h>
#include <gst/gst.h>

#include "DslApi.h"



// # File path for the single File Source
static const std::wstring file_path(L"/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_qHD.mp4");


// # Filespecs for the Primary Triton Inference Server (PTIS)
static const std::wstring primary_infer_config_file = L"/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt";

// File name for .dot file output
static const std::wstring dot_file = L"state-playing";

// # Window Sink Dimensions
int sink_width = 1280;
int sink_height = 720;

// ## 
// # Function to be called on XWindow KeyRelease event
// ## 
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
        dsl_main_loop_quit();
    }
}
 
// ## 
// # Function to be called on XWindow Delete event
// ##
void xwindow_delete_event_handler(void* client_data)
{
    std::cout<<"delete window event"<<std::endl;
    dsl_main_loop_quit();
}
    

// # Function to be called on End-of-Stream (EOS) event
void eos_event_listener(void* client_data)
{
    std::cout<<"Pipeline EOS event"<<std::endl;
    dsl_main_loop_quit();
}
    
// Function to convert state enum to state string
std::string get_state_name(int state)
{
    // Must match GST_STATE enum values
    // DSL_STATE_NULL                                              1
    // DSL_STATE_READY                                             2
    // DSL_STATE_PAUSED                                            3
    // DSL_STATE_PLAYING                                           4
    // DSL_STATE_CHANGE_ASYNC                                      5
    // DSL_STATE_UNKNOWN                                           UINT32_MAX

    std::string state_str = "UNKNOWN";
    switch(state)
    {
        case DSL_STATE_NULL:
            state_str = "NULL";
        break;
        case DSL_STATE_READY:
            state_str = "READY";
        break;
        case DSL_STATE_PAUSED:
            state_str = "PAUSED";
        break;
        case DSL_STATE_PLAYING:
            state_str = "PLAYING";
        break;
        case DSL_STATE_CHANGE_ASYNC:
            state_str = "CHANGE_ASYNC";
        break;
        case DSL_STATE_UNKNOWN:
            state_str = "UNKNOWN";
        break;
    }

    return state_str;
}

// ## 
// # Function to be called on every change of Pipeline state
// ## 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout<<"previous state = " << get_state_name(old_state) << ", new state = " << get_state_name(new_state) << std::endl;
    if(new_state == DSL_STATE_PLAYING){        
        dsl_pipeline_dump_to_dot(L"pipeline",L"state-playing");
    }
}


int main(int argc, char** argv)
{  
    DslReturnType retval;

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // # New File Source using the file path specified above, repeat diabled.
        retval = dsl_source_file_new(L"uri-source", file_path.c_str(), false);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // # New Primary TIS using the filespec specified above, with interval = 0
        retval = dsl_infer_tis_primary_new(L"primary-tis", primary_infer_config_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New KTL Tracker, setting output width and height of tracked objects
        retval = dsl_tracker_ktl_new(L"ktl-tracker", 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New OSD with clock and text enabled... using default values.
        retval = dsl_osd_new(L"on-screen-display", true, true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new(L"window-sink", 0, 0, sink_width, sink_height);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add all the components to a new pipeline
        const wchar_t* components[] = { L"uri-source",L"primary-tis",L"ktl-tracker",L"on-screen-display",L"window-sind",nullptr};
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);            
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", xwindow_key_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Play the pipeline
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Join with main loop until released - blocking call
        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        break;

    }

    // # Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_pipeline_delete_all();
    dsl_component_delete_all();


    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}


