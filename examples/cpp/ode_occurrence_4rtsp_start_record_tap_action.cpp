#include <iostream>
#include <map>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <gtkmm.h>
#include <gst/gst.h>

#include "DslApi.h"

// Set Camera RTSP URI's - these must be set to valid rtsp uri's for camera's on your network
// RTSP Source URI	
std::wstring src_url_0 = L"rtsp://admin:Password1!@10.0.0.38:554/cam/realmonitor?channel=1&subtype=1";
std::wstring src_url_1 = L"rtsp://admin:Password1!@10.0.0.38:554/cam/realmonitor?channel=2&subtype=1";
std::wstring src_url_2 = L"rtsp://admin:Password1!@10.0.0.38:554/cam/realmonitor?channel=3&subtype=1";
std::wstring src_url_3 = L"rtsp://admin:Password1!@10.0.0.38:554/cam/realmonitor?channel=4&subtype=1";
std::wstring office_url = L"rtsp://admin:Password1!@10.0.0.100";
std::wstring office_2_url = L"rtsp://admin:12345@10.0.0.101/axis-media/media.amp";

   
// These must be set to point to the location of these files on your network.  
//  Examples for your use can often be found in your Deepstream install, i.e. /opt/nvidia/deepstream/deepstream-6.0/samples
// # Filespecs for the Primary GIE	
std::wstring primary_infer_config_file = L"../resources/configs/config_infer_primary.txt";	
std::wstring primary_model_engine_file = L"../resources/models/infer_primary_b9_gpu0_fp16.engine";
std::wstring tracker_config_file = L"../resources/configs/iou_config.txt";

int TILER_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH;
int TILER_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT;
int WINDOW_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH;
int WINDOW_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT;

int PGIE_CLASS_ID_VEHICLE = 0;
int PGIE_CLASS_ID_BICYCLE = 1;	
int PGIE_CLASS_ID_PERSON = 2;	
int PGIE_CLASS_ID_ROADSIGN = 3;


// ## 	
// # Objects of this class will be used as "client_data" for all callback notifications.	
// # defines a class of all component names associated with a single RTSP Source. 	
// # The names are derived from the unique Source name	
// ##	

struct ClientData
{
    ClientData(std::wstring src, std::wstring rtsp_url){
        source = src;	
        occurrence_trigger = source + L"-occurrence-trigger";
        record_tap = source + L"-record-tap";	
        ode_notify = source + L"-ode-notify";
        start_record = source + L"-start-record";
        url = rtsp_url;
    }

    std::wstring source;
    std::wstring occurrence_trigger;
    std::wstring record_tap;
    std::wstring ode_notify;
    std::wstring start_record;
    std::wstring url;
};


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
        std::cout << "Main Loop Quit" << std::endl;
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

// ## 	
// # Function to create all Display Types used in this example	
// ## 	
DslReturnType create_display_types()
{
    DslReturnType retval;

    // # ````````````````````````````````````````````````````````````````````````````````````````````````````````	
    // # Create new RGBA color types	
    retval = dsl_display_type_rgba_color_new(L"full-red", 1.0f, 0.0f, 0.0f, 1.0f);	
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_display_type_rgba_color_new(L"full-white", 1.0f, 1.0f, 1.0f, 1.0f);	
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_display_type_rgba_color_new(L"opaque-black", 0.0f, 0.0f, 0.0f, 0.8f);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_display_type_rgba_font_new(L"impact-20-white", L"impact", 20, L"full-white");	
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # Create a new Text type object that will be used to show the recording in progress	
    retval = dsl_display_type_rgba_text_new(L"rec-text", L"REC    ", 10, 30, L"impact-20-white", true, L"opaque-black");
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # A new RGBA Circle to be used to simulate a red LED light for the recording in progress.	
    return dsl_display_type_rgba_circle_new(L"red-led", 94, 52, 8, L"full-red", true, L"full-red");

}





void RecordingStarted(uint64_t event_id, const wchar_t* trigger, void* buffer, void* frame_meta, void* object_meta, void* client_data)
{
    // # cast the C void* client_data back to a py_object pointer and deref	    
    ClientData* camera = reinterpret_cast<ClientData*>(client_data);

    // # a good place to enabled an Always Trigger that adds `REC` text to the frame which can	
    // # be disabled in the RecordComplete callback below. And/or send notifictions to external clients.	

    // # in this example we will call on the Tiler to show the source that started recording.
    uint timeout_duration = 0;  // time to show the source in units of seconds, before showing all-sources again A value of 0 indicates no timeout.
    dsl_tiler_source_show_set(L"tiler", camera->source.c_str(), timeout_duration, true);
}


// ##	
// # Callback function to process all "record-complete" notifications	
// ##	
void* RecordComplete(dsl_recording_info* session_info, void* client_data)
{
    // # session_info is obtained using the NVIDIA python bindings	

    // # cast the C void* client_data back to a py_object pointer and deref
    ClientData* camera = reinterpret_cast<ClientData*>(client_data);

    // # reset the Trigger that started this recording so that a new session can be started.	
    dsl_ode_trigger_reset(camera->occurrence_trigger.c_str());
}

// ##	
// # Function to create all "1-per-source" components, and add them to the Pipeline	
// # pipeline - unique name of the Pipeline to add the Source components to	
// # clientdata - pointer to instance of custom client data
// # ode_handler - Object Detection Event (ODE) handler to add the new Trigger and Actions to	
// ##	
// DslReturnType CreatePerSourceComponents(const wchar_t* pipeline, const wchar_t* source, 
//                                         const wchar_t* rtsp_uri, const wchar_t* ode_handler, 
//                                         ComponentNames* components)

DslReturnType CreatePerSourceComponents(const wchar_t* pipeline, ClientData* clientdata, const wchar_t* ode_handler)
{
    DslReturnType retval;

    // # New Component names based on unique source name	
    // ComponentNames components(source); 
    void* ptrClientData = reinterpret_cast<void*>(&clientdata);   
    
    
    // # For each camera, create a new RTSP Source for the specific RTSP URI	
    retval = dsl_source_rtsp_new(clientdata->source.c_str(), clientdata->url.c_str(), DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, false, 0, 100, 2);	
    if (retval != DSL_RESULT_SUCCESS) return retval;



    // # New record tap created with our common RecordComplete callback function defined above	    
    retval = dsl_tap_record_new(clientdata->record_tap.c_str(), L"./recordings/", DSL_CONTAINER_MKV, RecordComplete);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # Add the new Tap to the Source directly	
    retval = dsl_source_rtsp_tap_add(clientdata->source.c_str(), clientdata->record_tap.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;



    // # Next, create the Person Occurrence Trigger. We will reset the trigger in the recording complete callback	
    retval = dsl_ode_trigger_occurrence_new(clientdata->occurrence_trigger.c_str(), clientdata->source.c_str(), PGIE_CLASS_ID_PERSON, 1);	
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # New (optional) Custom Action to be notified of ODE Occurrence, and pass component names as client data.	
    retval = dsl_ode_action_custom_new(clientdata->ode_notify.c_str(), RecordingStarted, ptrClientData);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # Create a new Action to start the record session for this Source, with the component names as client data	
    retval = dsl_ode_action_tap_record_start_new(clientdata->start_record.c_str(), clientdata->record_tap.c_str(), 2, 30, ptrClientData);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # Add the Actions to the trigger for this source.
    const wchar_t* actions[] = {clientdata->ode_notify.c_str(), clientdata->start_record.c_str(), nullptr};
    retval = dsl_ode_trigger_action_add_many(clientdata->occurrence_trigger.c_str(), actions);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # Add the new Source with its Record-Tap to the Pipeline	
    retval = dsl_pipeline_component_add(pipeline, clientdata->source.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // # Add the new Trigger to the ODE Pad Probe Handler	
    return dsl_pph_ode_trigger_add(ode_handler, clientdata->occurrence_trigger.c_str());	
}




int main(int argc, char** argv)
{  
    

    DslReturnType retval = DSL_RESULT_FAILURE;

    // # Since we're not using args, we can Let DSL initialize GST on first call	
    while(true) // this construct allows us to use "break" to exit bracketed region below (better than "goto")
    {	

        // # ````````````````````````````````````````````````````````````````````````````````````````````````````````	
        // # This example is used to demonstrate the use of First Occurrence Triggers and Start Record Actions	
        // # to control Record Taps with a multi camera setup	

        retval = create_display_types();
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Create a new Action to display the "recording in-progress" text	
        retval = dsl_ode_action_display_meta_add_new(L"rec-text-overlay", L"rec-text");
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Create a new Action to display the "recording in-progress" LED	
        retval = dsl_ode_action_display_meta_add_new(L"red-led-overlay", L"red-led");	
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New Primary GIE using the filespecs above, with interval and Id	
        retval = dsl_infer_gie_primary_new(L"primary-gie", primary_infer_config_file.c_str(), nullptr, 4); // primary_model_engine_file.c_str(), 2);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New KTL Tracker, setting max width and height of input frame	
        retval = dsl_tracker_iou_new(L"iou-tracker", tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New Tiled Display, setting width and height, use default cols/rows set by source count	
        retval = dsl_tiler_new(L"tiler", TILER_WIDTH, TILER_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Object Detection Event (ODE) Pad Probe Handler (PPH) to manage our ODE Triggers with their ODE Actions	
        retval = dsl_pph_ode_new(L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add the ODE Pad Probe Handler to the Sink Pad of the Tiler	
        retval = dsl_tiler_pph_add(L"tiler", L"ode-handler", DSL_PAD_SINK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New OSD with clock and text enabled... using default values.
        retval = dsl_osd_new(L"on-screen-display", true, true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display	
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Add all the components to our pipeline	
        const wchar_t* cmpts[] = {L"primary-gie", L"iou-tracker", L"tiler", L"on-screen-display", L"window-sink", nullptr};
        retval = dsl_pipeline_new_component_add_many(L"pipeline", cmpts);	
        if (retval != DSL_RESULT_SUCCESS) break;


        // Add the 4 cameras here.  If fewer/more cameras are to be used, comment/add the lines below as appropriate
            // camera 1
            ClientData camera0(L"src-0", src_url_0.c_str());
            retval = CreatePerSourceComponents(L"pipeline", &camera0, L"ode-handler");
            if (retval != DSL_RESULT_SUCCESS) break;

            // camera 2
            ClientData camera1(L"src-1", src_url_1.c_str());
            retval = CreatePerSourceComponents(L"pipeline", &camera1, L"ode-handler");
            if (retval != DSL_RESULT_SUCCESS) break;

            // camera 3
            ClientData camera2(L"src-2", src_url_2.c_str());
            retval = CreatePerSourceComponents(L"pipeline", &camera2, L"ode-handler");
            if (retval != DSL_RESULT_SUCCESS) break;

            // camera 4
            ClientData camera3(L"src-3", office_url.c_str());
            retval = CreatePerSourceComponents(L"pipeline", &camera3, L"ode-handler");
            if (retval != DSL_RESULT_SUCCESS) break;

        // # Add the XWindow event handler functions defined above	
        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", xwindow_key_event_handler, nullptr);	
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # Play the pipeline	
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        break;	        
    }

    // # Print out the final result
    std::cout << "DSL Return: " <<  dsl_return_value_to_string(retval) << std::endl;

    // # Cleanup all DSL/GST resources
    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}

