
################################################################################
# The MIT License
#
# Copyright (c) 2024, Prominence AI, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

################################################################################
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
################################################################################

#!/usr/bin/env python

import sys

from dsl import *


uri_file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()
 
## 
# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_pipeline_stop('pipeline')
    dsl_main_loop_quit()

## 
# Function to be called on End-of-Stream (EOS) event
## 
def eos_event_listener(client_data):
    print('Pipeline EOS event')
    dsl_pipeline_stop('pipeline')
    dsl_main_loop_quit()

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")
        
##
# To be used as client_data with our Source Meter PPH, and passed to our 
# client_calback
##
class ReportData:
  def __init__(self):
    self.m_report_count = 0
    
## 
# Meter Sink client callback funtion
## 
def meter_pph_handler(session_avgs, interval_avgs, source_count, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    report_data = cast(client_data, POINTER(py_object)).contents.value

    # Print header
    header = ""
    for source in range(source_count):
        subheader = f"FPS {source} (AVG)"
        header += "{:<15}".format(subheader)
    print()
    print(header)

    # Print FPS counters
    counters = ""
    for source in range(source_count):
        counter = "{:.2f} ({:.2f})".format(interval_avgs[source], session_avgs[source])
        counters += "{:<15}".format(counter)
    print(counters)
    print()

    # Increment reporting count
    report_data.m_report_count += 1

    # Print out the current Component Queue levels
    dsl_component_queue_current_level_print_many(['custom-source', 'egl-sink', None],
        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
    dsl_component_queue_current_level_print_many(['custom-source', 'egl-sink', None],
        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES)
    
    return True        
def main(args):

    while True:

        # ---------------------------------------------------------------------------
        # Custom DSL Source Component, using the GStreamer "videotestsrc" plugin and
        # "capsfilter as a simple example. See the GST and Source API reference for 
        # more details.
        # https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
        # https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-source.md

        # Create a new element from the videotestsrc plugin
        retval = dsl_gst_element_new('videotestsrc-element', factory_name='videotestsrc')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the pattern to 19 – SMPTE 100%% color bars 
        retval = dsl_gst_element_property_uint_set('videotestsrc-element',
            'pattern', 19)

        # Create a new element using the capsfilter plugin
        retval = dsl_gst_element_new('capsfilter-element', factory_name='capsfilter')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new caps object to set the caps for the capsfilter
        retval = dsl_gst_caps_new('caps-object', 
            'video/x-raw, framerate=15/1, width=1280,height=720')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the caps property for the capsfilter using the caps object created above 
        retval = dsl_gst_element_property_caps_set('capsfilter-element', 
            'caps', 'caps-object')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Done with the caps object so let's delete it.
        retval = dsl_gst_caps_delete('caps-object')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Custom Source and add the elements to it. The elements will 
        # be linked in the order they're added.
        retval = dsl_source_custom_new_element_add_many('custom-source', 
            is_live=False, elements=['videotestsrc-element', 'capsfilter-element', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # ---------------------------------------------------------------------------
        # Create the remaining pipeline components

        # New Window Sink, 0 x/y offsets and dimensions defined above.
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, 
            WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add('egl-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('egl-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['custom-source', 'egl-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        report_data = ReportData()
        
        retval = dsl_pph_meter_new('meter-pph', interval=1, 
            client_handler=meter_pph_handler, client_data=report_data)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Meter to the Source pad of the Pipeline's Streammuxer.
        retval = dsl_pipeline_streammux_pph_add('pipeline', 'meter-pph')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break
        
    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
