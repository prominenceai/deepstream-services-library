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

#include <iostream>
#include <glib.h>

#include "DslApi.h"

// Set Camera RTSP URI's - these must be set to valid rtsp uri's for camera's on your network
// RTSP Source URI    
std::wstring src_url_1 = L"rtsp://user:pwd!@192.168.1.64:554/Streaming/Channels/101";
std::wstring src_url_2 = L"rtsp://user:pwd!@192.168.1.65:554/Streaming/Channels/101";
std::wstring src_url_3 = L"rtsp://user:pwd!@192.168.1.66:554/Streaming/Channels/101";
std::wstring src_url_4 = L"rtsp://user:pwd!@192.168.1.67:554/Streaming/Channels/101";

   
// Filespecs for the Primary GIE    
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");


int TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH;
int TILER_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT;
int WINDOW_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH;
int WINDOW_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT;

int PGIE_CLASS_ID_VEHICLE = 0;
int PGIE_CLASS_ID_BICYCLE = 1;    
int PGIE_CLASS_ID_PERSON = 2;    
int PGIE_CLASS_ID_ROADSIGN = 3;

//     
// Objects of this class will be used as "client_data" for all callback notifications.    
// defines a class of all component names associated with a single RTSP Source.     
// The names are derived from the unique Source name    
//    

struct ClientData
{
    ClientData(std::wstring src, std::wstring rtsp_url){
        source = src;    
        instance_trigger = source + L"-instance-trigger";
        always_trigger = source + L"-always-trigger";
        record_tap = source + L"-record-tap";    
        start_record = source + L"-start-record";
        display_meta = source + L"-display-meta";
        url = rtsp_url;
    }

    std::wstring source;
    std::wstring instance_trigger;
    std::wstring always_trigger;
    std::wstring record_tap;
    std::wstring start_record;
    std::wstring display_meta;
    std::wstring url;
};

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
        std::cout << "Main Loop Quit" << std::endl;
        dsl_main_loop_quit();
    }
}

// 
// Function to be called on XWindow Delete event
//
void xwindow_delete_event_handler(void* client_data)
{
    std::cout<<"delete window event"<<std::endl;
    dsl_main_loop_quit();
}
    

// 
// Function to be called on End-of-Stream (EOS) event
// 
void eos_event_listener(void* client_data)
{
    std::cout<<"Pipeline EOS event"<<std::endl;
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

//     
// Function to create all Display Types used in this example    
//     
DslReturnType create_display_types()
{
    DslReturnType retval;

    // ````````````````````````````````````````````````````````````````````````````````````````````````````````    
    // Create new RGBA color types    
    retval = dsl_display_type_rgba_color_custom_new(L"full-red", 1.0f, 0.0f, 0.0f, 1.0f);    
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_display_type_rgba_color_custom_new(L"full-white", 1.0f, 1.0f, 1.0f, 1.0f);    
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_display_type_rgba_color_custom_new(L"opaque-black", 0.0f, 0.0f, 0.0f, 0.8f);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    retval = dsl_display_type_rgba_font_new(L"impact-20-white", L"impact", 20, L"full-white");    
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Create a new Text type object that will be used to show the recording in progress    
    retval = dsl_display_type_rgba_text_new(L"rec-text", L"REC    ", 10, 30, L"impact-20-white", true, L"opaque-black");
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // A new RGBA Circle to be used to simulate a red LED light for the recording in progress.    
    return dsl_display_type_rgba_circle_new(L"red-led", 94, 52, 8, L"full-red", true, L"full-red");

}

//    
// Callback function to handle recording session start and stop events
//    
void* OnRecordingEvent(dsl_recording_info* session_info, void* client_data)
{
    // session_info is obtained using the NVIDIA python bindings    
    // cast the C void* client_data back to a py_object pointer and deref
    ClientData* camera = reinterpret_cast<ClientData*>(client_data);

    DslReturnType retval;

    std::cout << "session_id: " << session_info->session_id << std::endl;
    
    // If we're starting a new recording for this source
    if (session_info->recording_event == DSL_RECORDING_EVENT_START)
    {
        std::cout << "event:      " << "DSL_RECORDING_EVENT_START" << std::endl;

        // enable the always trigger showing the metadata for "recording in session" 
        uint retval = dsl_ode_trigger_enabled_set(camera->always_trigger.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS)
        {
            std::cout << "Enable always trigger failed with error: " 
                << dsl_return_value_to_string(retval) << std::endl;
        }

        // in this example we will call on the Tiler to show the source that started recording.
        // timeout(0) = show indefinitely, and has_precedence(true)
        retval = dsl_tiler_source_show_set(L"tiler", camera->source.c_str(), 0, true);    
        if (retval != DSL_RESULT_SUCCESS)
        {
            std::cout << "Tiler show single source failed with error: "
                << dsl_return_value_to_string(retval) << std::endl;
        }
    }
    // Else, the recording session has ended for this source
    else
    {    
        std::cout << "event:      " << "DSL_RECORDING_EVENT_END" << std::endl;
        std::cout << "filename:   " << session_info->filename << std::endl;
        std::cout << "dirpath:    " << session_info->dirpath << std::endl;
        std::cout << "duration:   " << session_info->duration << std::endl;
        std::cout << "container:  " << session_info->container_type << std::endl;
        std::cout << "width:      " << session_info->width << std::endl;
        std::cout << "height:     " << session_info->height << std::endl;

        // disable the always trigger showing the metadata for "recording in session" 
        retval = dsl_ode_trigger_enabled_set(camera->always_trigger.c_str(), false);
        if (retval != DSL_RESULT_SUCCESS)
        {
            std::cout << "Disable always trigger failed with error: "
                << dsl_return_value_to_string(retval) << std::endl;
        }

        // if we're showing the source that started this recording
        // we can set the tiler back to showing all tiles, otherwise
        // another source has started recording and taken precedence
        const wchar_t* current_source_cstr;
        uint current_timeout;
        
        retval = dsl_tiler_source_show_get(L"tiler", &current_source_cstr, &current_timeout);
        std::wstring current_source(current_source_cstr);
        
        if (retval == DSL_RESULT_SUCCESS and current_source == camera->source)
        {
            dsl_tiler_source_show_all(L"tiler");
        }

        // re-enable the one-shot trigger for the next "New Instance" of a person
        retval = dsl_ode_trigger_reset(camera->instance_trigger.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            std::cout << "Failed to reset instance trigger with error:"
                << dsl_return_value_to_string(retval) << std::endl;
        }
    }
}

//
// Function to create all "1-per-source" components, and add them to the Pipeline    
// pipeline - unique name of the Pipeline to add the Source components to    
// clientdata - pointer to instance of custom client data
// ode_handler - Object Detection Event (ODE) handler to add the new Trigger and Actions to    
// 

DslReturnType CreatePerSourceComponents(const wchar_t* pipeline, 
    ClientData* clientdata, const wchar_t* ode_handler)
{
    DslReturnType retval;

    // New Component names based on unique source name    
    // ComponentNames components(source); 
    void* ptrClientData = reinterpret_cast<void*>(clientdata);   
    
    // For each camera, create a new RTSP Source for the specific RTSP URI    
    retval = dsl_source_rtsp_new(clientdata->source.c_str(), clientdata->url.c_str(), 
        DSL_RTP_ALL, false, 0, 100, 2);    
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // New record tap created with our common RecordComplete callback function defined above        
    retval = dsl_tap_record_new(clientdata->record_tap.c_str(), L"./recordings/", 
        DSL_CONTAINER_MKV, OnRecordingEvent);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the new Tap to the Source directly    
    retval = dsl_source_rtsp_tap_add(clientdata->source.c_str(), clientdata->record_tap.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Next, create the Instance Trigger with Person class id with a limit of 1. 
    // We will reset the trigger in the recording complete callback    
    retval = dsl_ode_trigger_instance_new(clientdata->instance_trigger.c_str(), 
        clientdata->source.c_str(), PGIE_CLASS_ID_PERSON, 1);    
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Create a new Action to start the record session for this Source, with the component names as client data    
    retval = dsl_ode_action_tap_record_start_new(clientdata->start_record.c_str(), 
        clientdata->record_tap.c_str(), 5, 30, ptrClientData);
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the Start Record Action to the trigger for this source.
    retval = dsl_ode_trigger_action_add(clientdata->instance_trigger.c_str(), 
        clientdata->start_record.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the new Source with its Record-Tap to the Pipeline    
    retval = dsl_pipeline_component_add(pipeline, clientdata->source.c_str());
    if (retval != DSL_RESULT_SUCCESS) return retval;
    
    // Create an action to add the metadata for the "recording in session" indicator
    const wchar_t* display_types[] = {L"rec-text", L"red-led", nullptr};
    retval = dsl_ode_action_display_meta_add_many_new(clientdata->display_meta.c_str(),
        display_types);
    if (retval != DSL_RESULT_SUCCESS) return retval;
    
    // Create an Always Trigger that will trigger on every frame when enabled.
    // We use this trigger to display meta data while the recording is in session.
    // POST_OCCURRENCE_CHECK == after all other triggers are processed first.
    retval = dsl_ode_trigger_always_new(clientdata->always_trigger.c_str(),     
        clientdata->source.c_str(), DSL_ODE_POST_OCCURRENCE_CHECK);    
    if (retval != DSL_RESULT_SUCCESS) return retval;
    
    // Add the Display Meta action created above to the Always trigger
    retval = dsl_ode_trigger_action_add(clientdata->always_trigger.c_str(),
        clientdata->display_meta.c_str());

    // Disable the trigger, to be re-enabled in the recording_event listener callback
    retval = dsl_ode_trigger_enabled_set(clientdata->always_trigger.c_str(), false);    
    if (retval != DSL_RESULT_SUCCESS) return retval;

    // Add the new Trigger to the ODE Pad Probe Handler    
    const wchar_t* triggers[] = {clientdata->instance_trigger.c_str(), 
        clientdata->always_trigger.c_str(), nullptr};
    return dsl_pph_ode_trigger_add_many(ode_handler, triggers);    
}

int main(int argc, char** argv)
{  
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) // this construct allows us to use "break" to exit bracketed region below (better than "goto")
    {    

        // ````````````````````````````````````````````````````````````````````````````````````````````````````````    
        // This example is used to demonstrate the use of First Occurrence Triggers and Start Record Actions    
        // to control Record Taps with a multi camera setup    

        retval = create_display_types();
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a new Action to display the "recording in-progress" text    
        retval = dsl_ode_action_display_meta_add_new(L"rec-text-overlay", L"rec-text");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a new Action to display the "recording in-progress" LED    
        retval = dsl_ode_action_display_meta_add_new(L"red-led-overlay", L"red-led");    
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs above, with interval and Id    
        // Set the engine_file param to nullptr to create the model engine. 
//        retval = dsl_infer_gie_primary_new(L"primary-gie", primary_infer_config_file.c_str(), nullptr, 4); 
        retval = dsl_infer_gie_primary_new(L"primary-gie", primary_infer_config_file.c_str(), 
            primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New KTL Tracker, setting max width and height of input frame    
        retval = dsl_tracker_iou_new(L"iou-tracker", tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Tiled Display, setting width and height, use default cols/rows set by source count    
        retval = dsl_tiler_new(L"tiler", TILER_WIDTH, TILER_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Object Detection Event (ODE) Pad Probe Handler (PPH) to manage our ODE Triggers with their ODE Actions    
        retval = dsl_pph_ode_new(L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the ODE Pad Probe Handler to the Sink Pad of the Tiler    
        retval = dsl_tiler_pph_add(L"tiler", L"ode-handler", DSL_PAD_SINK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // # New OSD with text, clock, bboxs enabled, mask display disabled
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display    
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add all the components to our pipeline    
        const wchar_t* cmpts[] = {L"primary-gie", L"iou-tracker", L"tiler", L"on-screen-display", L"window-sink", nullptr};
        retval = dsl_pipeline_new_component_add_many(L"pipeline", cmpts);    
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the 4 cameras here.  If fewer/more cameras are to be used, comment/add the lines below as appropriate
        ClientData camera1(L"src-1", src_url_1.c_str());
        retval = CreatePerSourceComponents(L"pipeline", &camera1, L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;

        ClientData camera2(L"src-2", src_url_2.c_str());
        retval = CreatePerSourceComponents(L"pipeline", &camera2, L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;

        ClientData camera3(L"src-3", src_url_3.c_str());
        retval = CreatePerSourceComponents(L"pipeline", &camera3, L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;

        ClientData camera4(L"src-4", src_url_4.c_str());
        retval = CreatePerSourceComponents(L"pipeline", &camera4, L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;


        // Add the XWindow event handler functions defined above    
        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", xwindow_key_event_handler, nullptr);    
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_xwindow_delete_event_handler_add(L"pipeline", xwindow_delete_event_handler, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add(L"pipeline", state_change_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, nullptr);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Play the pipeline    
        retval = dsl_pipeline_play(L"pipeline");
        if (retval != DSL_RESULT_SUCCESS) break;

        dsl_main_loop_run();
        retval = DSL_RESULT_SUCCESS;
        break;            
    }

    // Print out the final result
    std::cout << "DSL Return: " <<  dsl_return_value_to_string(retval) << std::endl;

    // Cleanup all DSL/GST resources
    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}

