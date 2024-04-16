/*
The MIT License

Copyright (c) 2022-2024, Prominence AI, Inc.

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

/*################################################################################
#
# The example demonstrates how to create a custom DSL Pipeline Component using
# the DSL GStreamer (GST) API. NOTE! All DSL Pipeline Components are derived 
# from the GST Bin container class. Bins allow you to combine a group of linked 
# elements into one logical element. 
#
# Elements are constructed from plugins installed with GStreamer or 
# using your own proprietary -- with a call to
#
#     dsl_gst_element_new('my-element', 'my-plugin-factory-name' )
#
# This example creates a simple GST Bin with two elements derived from
#  1. A 'queue' plugin - to create a new thread boundary for our Bin
#  2. An 'identity' plugin - a GST debug plugin to mimic our proprietary element
#
# Elements can be added to a bin on creation be calling
#
#    dsl_gst_bin_new_element_add_many('my-bin',
#        ['my-element-1', 'my-element-2', None])
#
# IMPORTANT! When adding your own Custom Components, it is important to
# set the Pipeline's link methods to DSL_PIPELINE_LINK_METHOD_BY_ORDER
# by calling
#
#   dsl_pipeline_link_method_set('pipeline', DSL_PIPELINE_LINK_METHOD_BY_ORDER)
#
# otherwise, all components will be linked in a fixed position (default).
# See the GST API Reference section at
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
#
##############################################################################*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "DslApi.h"

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

// Config and model-engine files 
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");


uint PGIE_CLASS_ID_VEHICLE = 0;
uint PGIE_CLASS_ID_BICYCLE = 1;
uint PGIE_CLASS_ID_PERSON = 2;
uint PGIE_CLASS_ID_ROADSIGN = 3;

uint WINDOW_WIDTH = DSL_1K_HD_WIDTH;
uint WINDOW_HEIGHT = DSL_1K_HD_HEIGHT;

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

// Custom Pad Probe Handler function called with every buffer
// 
uint custom_pad_probe_handler(void* buffer, void* user_data)
{
    // Retrieve batch metadata from the gst_buffer
    NvDsBatchMeta* pBatchMeta = gst_buffer_get_nvds_batch_meta((GstBuffer*)buffer);
    
    // For each frame in the batched meta data
    for (NvDsMetaList* pFrameMetaList = pBatchMeta->frame_meta_list; 
        pFrameMetaList; pFrameMetaList = pFrameMetaList->next)
    {
        // Check for valid frame data
        NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta*)(pFrameMetaList->data);
        if (pFrameMeta != NULL)
        {
            
 
        }
    }
    return DSL_PAD_PROBE_OK;
}

int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // ---------------------------------------------------------------------------
        // Custom DSL Pipeline Component, using the GStreamer "identify" plugin
        // as an example. Any GStreamer or proprietary plugin (with limitations)
        // can be used to create a custom component. See the GST API reference for 
        // more details.
        // https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md

        // IMPORTANT! We create a queue element to be our first element of our bin.
        // The queue will create a new thread on the source pad (output) to decouple 
        // the processing on sink and source pad, effectively creating a new thread for 
        // our custom component.
        retval = dsl_gst_element_new(L"identity-queue", L"queue");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a new element from the identity plugin
        retval = dsl_gst_element_new(L"identity-element", L"identity");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Create a list of Elements to add to the new Bin.
        const wchar_t* elements[] = {L"identity-queue",  L"identity-element", NULL};
        
        // Create a new bin and add the elements to it. The elements will be linked 
        // in the order they're added.
        retval = dsl_gst_bin_new_element_add_many(L"identity-bin", elements);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Once created, the Element's properties can be queryied or updated.
        // For example, we can read the 'flush-on-eos' from our queue
        boolean flush_on_eos;
        retval = dsl_gst_element_property_boolean_get(L"identity-queue",
            L"flush-on-eos", &flush_on_eos);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        std::cout << "flush-on-eos = " << flush_on_eos;
            
        // IMPORTANT! Pad Probe handlers can be added to any sink or src pad of 
        // any GST Element.
            
        // New Custom Pad Probe Handler to call Nvidia's example callback 
        // for handling the Batched Meta Data
        retval = dsl_pph_custom_new(L"custom-pph", 
            custom_pad_probe_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // Add the custom PPH to the Src pad (output) of the identity-element
        // retval = dsl_gst_element_pph_add(L"identity-element", 
        //     L"custom-pph", DSL_PAD_SRC);
        // if (retval != DSL_RESULT_SUCCESS) break;
            
        // ---------------------------------------------------------------------------
        // Create the remaining pipeline components
        
        // New File Source
        retval = dsl_source_file_new(L"uri-source-1", uri_h265.c_str(), true);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Primary GIE using the filespecs defined above, with interval and Id
        retval = dsl_infer_gie_primary_new(L"primary-gie", 
            primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 0);
        if (retval != DSL_RESULT_SUCCESS) break;
        
        // New IOU Tracker, setting operational width and height of input frame
        retval = dsl_tracker_new(L"iou-tracker", 
            iou_tracker_config_file.c_str(), 480, 272);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(L"on-screen-display", true, true, true, false);
        if (retval != DSL_RESULT_SUCCESS) break;

        // New Window Sink, 0 x/y offsets.
        retval = dsl_sink_window_egl_new(L"egl-sink", 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add(L"egl-sink", 
            xwindow_key_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_sink_window_delete_event_handler_add(L"egl-sink", 
            xwindow_delete_event_handler, NULL);
        if (retval != DSL_RESULT_SUCCESS) break;
    
        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"uri-source-1", L"primary-gie", 
            L"iou-tracker", L"identity-bin", L"on-screen-display", L"egl-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // IMPORTANT! set the link method for the Pipeline to link by 
        // add order (and not by fixed position - default)
        retval = dsl_pipeline_link_method_set(L"pipeline",
            DSL_PIPELINE_LINK_METHOD_BY_ADD_ORDER);
            
        // Add the EOS listener function defined above
        retval = dsl_pipeline_eos_listener_add(L"pipeline", eos_event_listener, NULL);
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
            
