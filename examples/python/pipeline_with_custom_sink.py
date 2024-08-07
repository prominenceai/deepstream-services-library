
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
################################################################################

#!/usr/bin/env python

import sys

from dsl import *

# Import NVIDIA's pyds Pad Probe Handler example
from nvidia_pyds_pad_probe_handler import custom_pad_probe_handler

uri_file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov"

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

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
        
def main(args):

    while True:

        # ---------------------------------------------------------------------------
        # Custom DSL Pipeline Sink compossed of the four elements (including the built-in queue). 
        # See the GST API reference for more details.
        # https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md

        # Create a new element from the nvvideoconvert plugin to to convert the buffer from 
        # 'video/x-raw(memory:NVMM)' to 'video/x-raw'
        retval = dsl_gst_element_new('nvvideoconvert-element', 
            factory_name='nvvideoconvert')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new element from the capsfilter plugin to filter the 
        # nvvideoconvert's capabilities to 'video/x-raw'.
        retval = dsl_gst_element_new('capsfilter-element', 
            factory_name='capsfilter')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new caps object to set the caps for the capsfilter
        retval = dsl_gst_caps_new('caps-object', 
            'video/x-raw')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the caps property for the capsfilter using the caps object. 
        retval = dsl_gst_element_property_caps_set('capsfilter-element', 
            'caps', 'caps-object')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Done with the caps object so let's delete it.
        retval = dsl_gst_caps_delete('caps-object')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new element from the glimagesink plugin - actual Sink element
        retval = dsl_gst_element_new('glimagesink-element', 
            factory_name='glimagesink')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new Custom Sink Component and add the elements to it. 
        # IMPORTANT! The elements will be linked in the order they're added.
        retval = dsl_sink_custom_new_element_add_many('glimagesink-sink', 
            ['nvvideoconvert-element', 'capsfilter-element', 'glimagesink-element',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ---------------------------------------------------------------------------
        # Create the remaining pipeline components

        # New URI File Source using the filespec defined above
        retval = dsl_source_file_new('uri-source', uri_file, False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock, and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=False, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker',
            'on-screen-display', 'glimagesink-sink', None])
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
