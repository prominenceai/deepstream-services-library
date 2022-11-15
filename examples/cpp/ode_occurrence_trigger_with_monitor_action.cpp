/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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

/* ------------------------------------------------------------------------------------
 This example demonstrates the use of an ODE Monitor Action -- added to an 
 ODE Occurrence Trigger with the below criteria -- to monitor all 
 ODE Occurrences
   - class id            = PGIE_CLASS_ID_PERSON
   - inference-done-only = TRUE
   - minimum confidience = PERSON_MIN_CONFIDENCE
   - minimum width       = PERSON_MIN_WIDTH
   - minimum height      = PERSON_MIN_HEIGHT
*/

#include <iostream>
#include <glib.h>

#include "DslApi.h"

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Filespecs for the Primary GIE
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

uint PGIE_CLASS_ID_VEHICLE = 0;
uint PGIE_CLASS_ID_BICYCLE = 1;
uint PGIE_CLASS_ID_PERSON = 2;
uint PGIE_CLASS_ID_ROADSIGN = 3;

uint WINDOW_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH/4;
uint WINDOW_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT/4;

// Minimum Inference confidence level to Trigger ODE Occurrence
float PERSON_MIN_CONFIDENCE = 0.4; // 40%

uint PERSON_MIN_WIDTH = 120;
uint PERSON_MIN_HEIGHT = 320;

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
// Function to be called on XWindow Delete event
//
void xwindow_delete_event_handler(void* client_data)
{
    std::cout<<"delete window event"<<std::endl;
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
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
// Function to be called on every change of Pipeline state
// 
void state_change_listener(uint old_state, uint new_state, void* client_data)
{
    std::cout<<"previous state = " << dsl_state_value_to_string(old_state) 
        << ", new state = " << dsl_state_value_to_string(new_state) << std::endl;
}

// 
// Callback function for the ODE Monitor Action - illustrates how to
// dereference the ODE "info_ptr" and access the data fields.
// Note: you would normally use the ODE Print Action to print the info
// to the console window if that is the only purpose of the Action.
// 
void ode_occurrence_monitor(dsl_ode_occurrence_info* pInfo, void* client_data)
{
    std::wcout << "Trigger Name        : " << pInfo->trigger_name << "\n";
    std::cout << "  Unique Id         : " << pInfo->unique_ode_id << "\n";
    std::cout << "  NTP Timestamp     : " << pInfo->ntp_timestamp << "\n";
    std::cout << "  Source Data       : ------------------------" << "\n";
    std::cout << "    Id              : " << pInfo->source_info.source_id << "\n";
    std::cout << "    Batch Id        : " << pInfo->source_info.batch_id << "\n";
    std::cout << "    Pad Index       : " << pInfo->source_info.pad_index << "\n";
    std::cout << "    Frame           : " << pInfo->source_info.frame_num << "\n";
    std::cout << "    Width           : " << pInfo->source_info.frame_width << "\n";
    std::cout << "    Height          : " << pInfo->source_info.frame_height << "\n";
    std::cout << "    Infer Done      : " << pInfo->source_info.inference_done << "\n";

    if (pInfo->is_object_occurrence)
    {
        std::cout << "  Object Data       : ------------------------" << "\n";
        std::cout << "    Class Id        : " << pInfo->object_info.class_id << "\n";
        std::cout << "    Infer Comp Id   : " << pInfo->object_info.inference_component_id << "\n";
        std::cout << "    Tracking Id     : " << pInfo->object_info.tracking_id << "\n";
        std::cout << "    Label           : " << pInfo->object_info.label << "\n";
        std::cout << "    Persistence     : " << pInfo->object_info.persistence << "\n";
        std::cout << "    Direction       : " << pInfo->object_info.direction << "\n";
        std::cout << "    Infer Conf      : " << pInfo->object_info.inference_confidence << "\n";
        std::cout << "    Track Conf      : " << pInfo->object_info.tracker_confidence << "\n";
        std::cout << "    Left            : " << pInfo->object_info.left << "\n";
        std::cout << "    Top             : " << pInfo->object_info.top << "\n";
        std::cout << "    Width           : " << pInfo->object_info.width << "\n";
        std::cout << "    Height          : " << pInfo->object_info.height << "\n";
    }
    else
    {
        std::cout << "  Accumulative Data : ------------------------" << "\n";
        std::cout << "    Occurrences     : " << pInfo->accumulative_info.occurrences_total << "\n";
        std::cout << "    Occurrences In  : " << pInfo->accumulative_info.occurrences_total << "\n";
        std::cout << "    Occurrences Out : " << pInfo->accumulative_info.occurrences_total << "\n";
    }
    std::cout << "  Trigger Criteria  : ------------------------" << "\n";
    std::cout << "    Class Id        : " << pInfo->criteria_info.class_id << "\n";
    std::cout << "    Infer Comp Id   : " << pInfo->criteria_info.inference_component_id << "\n";
    std::cout << "    Min Infer Conf  : " << pInfo->criteria_info.min_inference_confidence << "\n";
    std::cout << "    Min Track Conf  : " << pInfo->criteria_info.min_tracker_confidence << "\n";
    std::cout << "    Infer Done Only : " << pInfo->criteria_info.inference_done_only << "\n";
    std::cout << "    Min Width       : " << pInfo->criteria_info.min_width << "\n";
    std::cout << "    Min Height      : " << pInfo->criteria_info.min_height << "\n";
    std::cout << "    Max Width       : " << pInfo->criteria_info.max_width << "\n";
    std::cout << "    Max Height      : " << pInfo->criteria_info.max_height << "\n";
    std::cout << "    Interval        : " << pInfo->criteria_info.interval << "\n";
}

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    
        //-----------------------------------------------------------------------------
        // First, we want to remove the Object labels and bounding boxes. This is
        // be adding Format-Label and Format-BBox actions to an Occurrence Trigger.

        // Create a Format Label Action to remove the Object Label from view
        // Note: the label can be disabled with the OSD API as well, however
        // that will disable all text/labels, not just object labels. 
        retval = dsl_ode_action_label_format_new(L"remove-label", 
            NULL, false, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Create a Format Bounding Box Action to remove the box border from view
        retval = dsl_ode_action_bbox_format_new(L"remove-bbox", 0,
            NULL, false, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create an Any-Class Occurrence Trigger for our remove Actions
        retval = dsl_ode_trigger_occurrence_new(L"every-occurrence-trigger", 
            DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE);
        if (retval != DSL_RESULT_SUCCESS) break;

        const wchar_t* actions[] = {L"remove-label", L"remove-bbox", NULL};
        retval = dsl_ode_trigger_action_add_many(L"every-occurrence-trigger", 
            actions);
        if (retval != DSL_RESULT_SUCCESS) break;

        //```````````````````````````````````````````````````````````````````````````````````
        // Next, create an Occurrence Trigger to filter on People - defined with
        // a minimuim confidence level to eleminate most false positives

        // New Occurrence Trigger, filtering on PERSON class_id,
        retval = dsl_ode_trigger_occurrence_new(L"person-occurrence-trigger", 
            DSL_ODE_ANY_SOURCE, PGIE_CLASS_ID_PERSON, DSL_ODE_TRIGGER_LIMIT_NONE);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Set a minimum confidence level to avoid false positives.
        retval = dsl_ode_trigger_infer_confidence_min_set(L"person-occurrence-trigger",
            PERSON_MIN_CONFIDENCE);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Set the inference done only filter. 
        retval = dsl_ode_trigger_infer_done_only_set(L"person-occurrence-trigger",
            true);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Set minimum bounding box dimensions to trigger.
        retval = dsl_ode_trigger_dimensions_min_set(L"person-occurrence-trigger",
            PERSON_MIN_WIDTH, PERSON_MIN_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        retval = dsl_ode_action_monitor_new(L"person-occurrence-monitor",
            ode_occurrence_monitor, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the ODE Heat-Mapper to the Person Occurrence Trigger.
        retval = dsl_ode_trigger_action_add(L"person-occurrence-trigger", 
            L"person-occurrence-monitor");
        if (retval != DSL_RESULT_SUCCESS) break;

        // New ODE Handler to handle all ODE Triggers with their Areas and Actions
        retval = dsl_pph_ode_new(L"ode-handler");
        if (retval != DSL_RESULT_SUCCESS) break;
        
        const wchar_t* triggers[] = 
            {L"every-occurrence-trigger", L"person-occurrence-trigger", NULL};
            
        // Add the two Triggers to the ODE PPH to be invoked on every frame. 
        retval = dsl_pph_ode_trigger_add_many(L"ode-handler", 
            triggers);
        if (retval != DSL_RESULT_SUCCESS) break;

        //----------------------------------------------------------------------------
        // Create the remaining Pipeline components
        
        // New File Source
        retval = dsl_source_file_new(L"uri-source-1", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_osd_pph_add(L"on-screen-display", L"ode-handler", DSL_PAD_SINK);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new(L"window-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"uri-source-1",  L"primary-gie", L"iou-tracker", 
            L"on-screen-display", L"window-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Add the EOS listener and XWindow event handler functions defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_pipeline_xwindow_key_event_handler_add(L"pipeline", 
            xwindow_key_event_handler, NULL);
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
    
    // Print out the final result
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
            