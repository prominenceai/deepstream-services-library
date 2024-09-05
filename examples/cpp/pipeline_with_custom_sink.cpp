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

/*################################################################################
#
# The example demonstrates how to create a custom DSL Sink Component with
# using custom GStreamer (GST) Elements.  
#
# Elements are constructed from plugins installed with GStreamer or 
# using your own proprietary with a call to
#
#     dsl_gst_element_new('my-element', 'my-plugin-factory-name' )
#
# IMPORTANT! All DSL Pipeline Components, intrinsic and custom, include
# a queue element to create a new thread boundary for the component's element(s)
# to process in. 
#
# This example creates a simple Custom Sink with four elements in total
#  1. The built-in 'queue' element - to create a new thread boundary.
#  2. An 'nvvideoconvert' element -  to convert the buffer from 
#     'video/x-raw(memory:NVMM)' to 'video/x-raw'
#  3. A 'capsfilter' plugin - to filter the 'nvvideoconvert' caps to 
#     'video/x-raw'
#  4. A 'glimagesink' plugin - the actual Sink element for this Sink component.
#
# Multiple elements can be added to a Custom Sink on creation be calling
#
#    dsl_sink_custom_new_element_add_many('my-bin',
#        ['my-element-1', 'my-element-2', 'my-element-3', None])
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-sink.md
#
##############################################################################*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "DslApi.h"

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov");

// Config and model-engine files 
std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt");
std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

// Config file used by the IOU Tracker    
std::wstring iou_tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

   
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


int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

         // ---------------------------------------------------------------------------
        // Custom DSL Pipeline Sink compossed of the four elements (including the built-in queue). 
        // aSee the GST API reference for 
        // more details.
        // https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md

        // Create a new element from the nvvideoconvert plugin to to convert the buffer from 
        // 'video/x-raw(memory:NVMM)' to 'video/x-raw'
        retval = dsl_gst_element_new(L"nvvideoconvert-element", L"nvvideoconvert");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        retval = dsl_gst_element_new(L"capsfilter-element", L"capsfilter");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a new caps object to set the caps for the capsfilter
        retval = dsl_gst_caps_new(L"caps-object", L"video/x-raw");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Set the caps property for the capsfilter using the caps object created above 
        retval = dsl_gst_element_property_caps_set(L"capsfilter-element", 
            L"caps", L"caps-object");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Done with the caps object so let's delete it.
        retval = dsl_gst_caps_delete(L"caps-object");
        if (retval != DSL_RESULT_SUCCESS) break;

        retval = dsl_gst_element_new(L"glimagesink-element", L"glimagesink");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        const wchar_t* elements[] {L"nvvideoconvert-element",
            L"capsfilter-element", L"glimagesink-element", NULL};

        // Create a new bin and add the elements to it. The elements will be linked 
        // in the order they're added.
        retval = dsl_sink_custom_new_element_add_many(L"glimagesink-sink", 
            elements);
        if (retval != DSL_RESULT_SUCCESS) break;
                       
        // ---------------------------------------------------------------------------
        // Create the remaining pipeline components
        
        // New File Source
        retval = dsl_source_file_new(L"uri-source-1", uri_h265.c_str(), false);
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

        // Create a list of Pipeline Components to add to the new Pipeline.
        const wchar_t* components[] = {L"uri-source-1", L"primary-gie", 
            L"iou-tracker", L"on-screen-display", L"glimagesink-sink", NULL};
        
        // Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many(L"pipeline", components);
        if (retval != DSL_RESULT_SUCCESS) break;
            
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
    std::wcout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
            
