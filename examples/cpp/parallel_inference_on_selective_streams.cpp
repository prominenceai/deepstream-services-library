
/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
/* ###############################################################################
#
# This example shows how to use a Remuxer Component to create parallel branches,
# each with their own Inference Components (Preprocessors, Inference Engines, 
# Trackers, for example). 
# IMPORTANT! All branches are (currently) using the same model engine and config.
# files, which is not a valid use case. The actual inference components and 
# models to use for any specific use cases is beyond the scope of this example. 
#
# Each Branch added to the Remuxer can specify which streams to process or
# to process all. Use the Remuxer "branch-add-to" service to add to specific streams.
#
#       std::vector<uint> stream_ids = {0,1};

#       retval = dsl_remuxer_branch_add_to(L"remuxer", L"my-branch-0", 
#            &stream_ids[0], stream_ids.size());
#
# You can use the "branch-add" service if adding to all streams
#
#    dsl_remuxer_branch_add(L"remuxer", 'my-branch-0')
# 
# In this example, 4 RTSP Sources are added to the Pipeline:
#   - branch-1 will process streams [0,1]
#   - branch-2 will process streams [1,2]
#   - branch-3 will process streams [0,2,3]
#
# Three ODE Instance Triggers are created to trigger on new object instances
# events (i.e. new tracker ids). Each is filtering on a unique class-i
# (vehicle, person, and bicycle). 
#
# The ODE Triggers are added to an ODE Handler which is added to the src-pad
# (output) of the Remuxer.
#
# A single ODE Print Action is created and added to each Trigger (shared action).
# Using multiple Print Actions running in parallel -- each writing to the same 
# stdout buffer -- will result in the printed data appearing interlaced. A single 
# Action with an internal mutex will protect from stdout buffer reentrancy. 
# 
################################################################################*/

#include <iostream>
#include <glib.h>
#include <vector>
#include "DslApi.h"


// RTSP Source URI for AMCREST Camera    
std::wstring amcrest_rtsp_uri = 
    L"rtsp://username:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";

// RTSP Source URI for HIKVISION Camera    
std::wstring hikvision_rtsp_uri = 
    L"rtsp://admin:Segvisual44@192.168.1.64:554/Streaming/Channels/101";

// All branches are currently using the same config and model engine files
// which is pointless... The example will be updated to use multiple

// Filespecs (Jetson and dGPU) for the Primary GIEs
std::wstring primary_infer_config_file_1( 
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file_1( 
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

std::wstring primary_infer_config_file_2( 
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file_2( 
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

std::wstring primary_infer_config_file_3( 
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
std::wstring primary_model_engine_file_3( 
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

// Filespecs for the Secondary GIE

std::wstring sgie2_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
std::wstring sgie2_model_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet.etlt_b8_gpu0_int8.engine");
    
uint PGIE_CLASS_ID_VEHICLE = 0;
uint PGIE_CLASS_ID_BICYCLE = 1;
uint PGIE_CLASS_ID_PERSON = 2;
uint PGIE_CLASS_ID_ROADSIGN = 3;

// Source dimensions 720p
uint SOURCE_WIDTH = 1280;
uint SOURCE_HEIGHT = 720;

uint TILER_WIDTH = 1280;
uint TILER_HEIGHT = 720;

uint SINK_WIDTH = TILER_WIDTH;
uint SINK_HEIGHT = TILER_HEIGHT;

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

// Function to be called on End-of-Stream (EOS) event
void eos_event_listener(void* client_data)
{
    std::cout<<"Pipeline EOS event"<<std::endl;
    
    dsl_pipeline_stop(L"pipeline");
    dsl_main_loop_quit();
}

int main(int argc, char** argv)
{  
    DslReturnType retval;

    // # Since we're not using args, we can Let DSL initialize GST on first call
    while(true)
    {
        // 4 new RTSP Sources to produce streams 0 through 3
        retval = dsl_source_rtsp_new(L"rtsp-source-0",     
            hikvision_rtsp_uri.c_str(), DSL_RTP_ALL, 0, 0, 1000, 10); 
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_rtsp_new(L"rtsp-source-1",     
            hikvision_rtsp_uri.c_str(), DSL_RTP_ALL, 0, 0, 1000, 10); 
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_rtsp_new(L"rtsp-source-2",     
            hikvision_rtsp_uri.c_str(), DSL_RTP_ALL, 0, 0, 1000, 10); 
        if (retval != DSL_RESULT_SUCCESS) break;
        retval = dsl_source_rtsp_new(L"rtsp-source-3",     
            hikvision_rtsp_uri.c_str(), DSL_RTP_ALL, 0, 0, 1000, 10); 
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // ----------------------------------------------------------------------------
        // Inference Branch #1 (b1) - single Primary GIE.  Branch component
        // is NOT required if using single component.
        
        // New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new(L"pgie-b1", 
            primary_infer_config_file_1.c_str(), primary_model_engine_file_1.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // ----------------------------------------------------------------------------
        // Inference Branch #2 (b2) - Primary GIE and IOU Tracker.
        
        retval = dsl_infer_gie_primary_new(L"pgie-b2", 
            primary_infer_config_file_2.c_str(), primary_model_engine_file_2.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New IOU Tracker, setting operational width and height
        retval = dsl_tracker_new(L"tracker-b2", iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Branch Component names to add to the new Branch.
        const wchar_t* branch_components_2[] = {L"pgie-b2", 
            L"tracker-b2", NULL};
            
        retval = dsl_branch_new_component_add_many(L"branch-b2", branch_components_2);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        
        // ----------------------------------------------------------------------------
        // Inference Branch #3 (b3) - Primary GIE, Tracker, and Secondary GIE.

        retval = dsl_infer_gie_primary_new(L"pgie-b3", 
            primary_infer_config_file_3.c_str(), primary_model_engine_file_3.c_str(), 4);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_tracker_new(L"tracker-b3", iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_infer_gie_secondary_new(L"vehicletype-sgie-b3", 
            sgie2_config_file.c_str(), sgie2_model_file.c_str(), L"pgie-b3", 0);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a list of Branch Component names to add to the new Branch.
        const wchar_t* branch_components_3[] = {L"pgie-b3", 
            L"tracker-b3", L"vehicletype-sgie-b3", NULL};
            
        retval = dsl_branch_new_component_add_many(L"branch-b3", branch_components_3);
        if (retval != DSL_RESULT_SUCCESS) break;

        //// ----------------------------------------------------------------------------
        //
        // We create a new Remuxer component, that demuxes the batched streams, and
        // then Tee's the unbatched single streams into multiple branches. Each Branch 
        //   - connects to some or all of the single stream Tees as specified. 
        //   - re-muxes the streams back into a single batched stream for processing.
        //   - each branch is then linked to the Remuxer's Metamuxer
        retval = dsl_remuxer_new(L"remuxer");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Define our stream-ids for each branch. Available stream-ids are [0,1,2,3]
        std::vector<uint> stream_ids_b1 = {0,1};
        std::vector<uint> stream_ids_b2 = {1,2};
        std::vector<uint> stream_ids_b3 = {0,2,3};

        // IMPORTANT! Use the "add_to" service to add the Branch to specific streams,
        // or use the "add" service to connect to all streams.
        
        // Add pgie-b1 to the Remuxer to connect to specific streams - stream_ids_b1
        retval = dsl_remuxer_branch_add_to(L"remuxer", L"pgie-b1", 
            &stream_ids_b1[0], stream_ids_b1.size());
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add branch-2 to the Remuxer to connect to specific streams - stream_ids_b2
        retval = dsl_remuxer_branch_add_to(L"remuxer", L"branch-b2", 
            &stream_ids_b2[0], stream_ids_b2.size());
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add branch-3 to the Remuxer to connect to specific streams - stream_ids_b3
        retval = dsl_remuxer_branch_add_to(L"remuxer", L"branch-b3", 
            &stream_ids_b3[0], stream_ids_b3.size());
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------

        // Create the Object Detection Event (ODE) Pad Probe Handler (PPH)
        // and add the handler to the src-pad (output) of the Remuxer
        retval = dsl_pph_ode_new(L"ode-pph");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_remuxer_pph_add(L"remuxer", L"ode-pph", DSL_PAD_SRC);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a Bicycle Instance Trigger to trigger on new
        // instances (new tracker-ids) of the bicycle-class
        retval = dsl_ode_trigger_instance_new(L"bicycle-instance-trigger", 
            DSL_ODE_ANY_SOURCE, 
            PGIE_CLASS_ID_BICYCLE, 
            DSL_ODE_TRIGGER_LIMIT_NONE);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a Bicycle Instance Trigger to trigger on new
        // instances (new tracker-ids) of the vehicle-class
        retval = dsl_ode_trigger_instance_new(L"vehicle-instance-trigger", 
            DSL_ODE_ANY_SOURCE, 
            PGIE_CLASS_ID_VEHICLE, 
            DSL_ODE_TRIGGER_LIMIT_NONE);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a Person Instance Trigger to trigger on new
        // instances (new tracker-ids) of the bicycle-class
        retval = dsl_ode_trigger_instance_new(L"person-instance-trigger", 
            DSL_ODE_ANY_SOURCE, 
            PGIE_CLASS_ID_PERSON, 
            DSL_ODE_TRIGGER_LIMIT_NONE);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Single new ODE Print Action that will be shared by all Triggers.
        retval = dsl_ode_action_print_new(L"print-action", false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the Single Action to each of the Instance Triggers.
        retval = dsl_ode_trigger_action_add(L"bicycle-instance-trigger", 
            L"print-action");
        if (retval != DSL_RESULT_SUCCESS) break;
        
        retval = dsl_ode_trigger_action_add(L"person-instance-trigger", 
            L"print-action");
        if (retval != DSL_RESULT_SUCCESS) break;
        
        retval = dsl_ode_trigger_action_add(L"vehicle-instance-trigger", 
            L"print-action");
        if (retval != DSL_RESULT_SUCCESS) break;

        const wchar_t* triggers[] = {L"bicycle-instance-trigger", L"vehicle-instance-trigger", 
            L"person-instance-trigger", NULL};
            
        // Add all three Triggers to the ODE-PPH; bicycle, vehicle, and person.
        retval = dsl_pph_ode_trigger_add_many(L"ode-pph", triggers);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------
        // New Tiler, setting width and height, use default cols/rows set by 
        // the number of streams
        retval = dsl_tiler_new(L"tiler", TILER_WIDTH, TILER_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------
        // Next, create the On-Screen-Displays (OSD) with text, clock and bboxes.
        // enabled. 
        retval = dsl_osd_new(L"osd", true, true, true, true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------
        // New Window Sink
        retval = dsl_sink_window_egl_new(L"window-sink",
            0, 0, SINK_WIDTH, SINK_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above to both Window Sinks
        retval = dsl_sink_window_key_event_handler_add(L"window-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        retval = dsl_sink_window_delete_event_handler_add(L"window-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ----------------------------------------------------------------------------
        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {
            L"rtsp-source-0", L"rtsp-source-1", L"rtsp-source-2", L"rtsp-source-3", 
            L"remuxer", L"tiler", L"osd", L"window-sink", NULL};            
            
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
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
