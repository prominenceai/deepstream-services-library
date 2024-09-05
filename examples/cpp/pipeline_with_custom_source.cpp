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
# This example demonstrates how to create a custom DSL Source Component  
# using two GStreamer (GST) Elements created from two GST Plugins:
#   1. 'videotestsrc' as the source element.
#   2. 'capsfilter' to limit the video from the videotestsrc to  
#      'video/x-raw, framerate=15/1, width=1280, height=720'
#   
# Elements are constructed from plugins installed with GStreamer or 
# using your own proprietary plugin with a call to
#
#     dsl_gst_element_new('my-element', 'my-plugin-factory-name' )
#
# Multiple elements can be added to a Custom Source on creation be calling
#
#    dsl_source_custom_new_element_add_many('my-custom-source',
#        ['my-element-1', 'my-element-2', None])
#
# As with all DSL Video Sources, the Custom Souce will also include the 
# standard buffer-out-elements (queue, nvvideconvert, and capsfilter). 
# The Source in this example will be linked as follows:
#
#   videotestscr->capsfilter->queue->nvvideconvert->capsfilter
#
# See the GST and Source API reference sections for more information
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-source.md
#
##############################################################################*/

#include <iostream>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include "DslApi.h"

std::wstring uri_h265(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");


uint WINDOW_WIDTH = 1280;
uint WINDOW_HEIGHT = 720;

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


int main(int argc, char** argv)
{
    DslReturnType retval = DSL_RESULT_FAILURE;

    // Since we're not using args, we can Let DSL initialize GST on first call    
    while(true) 
    {    

        // ---------------------------------------------------------------------------
        // Custom DSL Source Component, using the GStreamer "videotestsrc" plugin and
        // "capsfilter as a simple example. See the GST and Source API reference for 
        // more details.
        // https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
        // https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-source.md

        // Create a new element from the videotestsrc plugin
        retval = dsl_gst_element_new(L"videotestsrc-element", L"videotestsrc");
        if (retval != DSL_RESULT_SUCCESS) break;
            
        // Set the pattern to 19 – SMPTE 100%% color bars 
        retval = dsl_gst_element_property_uint_set(L"videotestsrc-element",
            L"pattern", 19);
        if (retval != DSL_RESULT_SUCCESS) break;

        // Create a new element using the capsfilter plugin
        retval = dsl_gst_element_new(L"capsfilter-element", L"capsfilter");
        if (retval != DSL_RESULT_SUCCESS) break;


        // Create a new caps object to set the caps for the capsfilter
        retval = dsl_gst_caps_new(L"caps-object", 
            L"video/x-raw, framerate=15/1, width=1280,height=720");
        if (retval != DSL_RESULT_SUCCESS) break;


        // Set the caps property for the capsfilter using the caps object created above 
        retval = dsl_gst_element_property_caps_set(L"capsfilter-element", 
            L"caps", L"caps-object");
        if (retval != DSL_RESULT_SUCCESS) break;

        // Done with the caps object so let's delete it.
        retval = dsl_gst_caps_delete(L"caps-object");
        if (retval != DSL_RESULT_SUCCESS) break;

        const wchar_t* elements[] = {
            L"videotestsrc-element", L"capsfilter-element", NULL};
        // Create a new Custom Source and add the elements to it. The elements will 
        // be linked in the order they're added.
        retval = dsl_source_custom_new_element_add_many(L"custom-source", 
            true, elements);
        if (retval != DSL_RESULT_SUCCESS) break;

        // ---------------------------------------------------------------------------
        // Create the remaining pipeline components
        
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
        const wchar_t* components[] = {L"custom-source", L"egl-sink", NULL};
        
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
    std::cout << dsl_return_value_to_string(retval) << std::endl;

    dsl_delete_all();

    std::cout<<"Goodbye!"<<std::endl;  
    return 0;
}
            
