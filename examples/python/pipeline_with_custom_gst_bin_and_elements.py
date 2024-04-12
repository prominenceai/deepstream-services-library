
################################################################################
# The MIT License
#
# Copyright (c) 2019-2024, Prominence AI, Inc.
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
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
#
################################################################################

#!/usr/bin/env python

import sys

from dsl import *

# Import NVIDIA's pyds Pad Probe Handler example
from nvidia_pyds_pad_probe_handler import custom_pad_probe_handler

# uri_file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
uri_file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"


# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

TILER_WIDTH = 1280
TILER_HEIGHT = 720
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
        
def main(args):

    while True:

        # ---------------------------------------------------------------------------
        # Custom DSL Pipeline Component, using the GStreamer "identify" plugin
        # as an example. Any GStreamer or proprietary plugin (with limitations)
        # can be used to create a custom component. See the GST API reference for 
        # more details.
        # https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md

        # IMPORTANT! We create a queue element to be our first element of our bin.
        # The queue will create a new thread on the source pad (output) to decouple 
        # the processing on sink and source pad, effectively creating a new thread for 
        # our custom component.
        retval = dsl_gst_element_new('identity-queue', factory_name='queue')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new element from the identity plugin
        retval = dsl_gst_element_new('identity-element', factory_name='identity')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new bin and add the elements to it. The elements will be linked 
        # in the order they're added.
        ret_val = dsl_gst_bin_new_element_add_many('identity-bin', 
            elements = ['identity-queue', 'identity-element', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Once created, the Element's properties can be queryied or updated.
        # For example, we can read the 'flush-on-eos' from our queue
        retval, flush_on_eos = dsl_gst_element_property_boolean_get('identity-queue',
            property='flush-on-eos')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        print('flush-on-eos =', flush_on_eos)
            
        # IMPORTANT! Pad Probe handlers can be added to any sink or src pad of 
        # any GST Element.
            
        # New Custom Pad Probe Handler to call Nvidia's example callback 
        # for handling the Batched Meta Data
        retval = dsl_pph_custom_new('custom-pph', 
            client_handler=custom_pad_probe_handler, client_data=None)
        
        # Add the custom PPH to the Src pad (output) of the identity-element
        retval = dsl_gst_element_pph_add('identity-element', 
            handler='custom-pph', pad=DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # ---------------------------------------------------------------------------
        # Create the remaining pipeline components

        # New URI File Source using the filespec defined above
        retval = dsl_source_uri_new('uri-source', uri_file, False, False, 0)
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

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=False, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

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
            ['uri-source', 'primary-gie', 'iou-tracker', 'identity-bin',
            'on-screen-display', 'egl-sink', None])
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

        # IMPORTANT! set the link method for the Pipeline to link by 
        # add order (and not by fixed position - default)
        retval = dsl_pipeline_link_method_set('pipeline',
            DSL_PIPELINE_LINK_METHOD_BY_ORDER)
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
