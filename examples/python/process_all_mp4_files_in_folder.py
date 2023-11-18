################################################################################
# The MIT License
#
# Copyright (c) 2019-2023, Prominence AI, Inc.
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
# This simple example demonstrates how to process (infer-on) all .mp4 files
# in a given folder. 
#
# The inference Pipeline is built with the following components:
#   - URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
# 
# A Custom Pad-Probe-Handler is added to the Sink-Pad of the OSD
# to process the frame and object meta-data for each buffer received
#
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events.
#
# IMPORTANT! it is the end-of-stream (EOS) listener "eos_event_listener"
# that stops the Pipeline on EOS, sets the URI to the next file in 
# the list, and starts the Pipeline again.
#  
################################################################################

#!/usr/bin/env python

import sys
from dsl import *
import os

# Import NVIDIA's OSD Sink Pad Buffer Probe (pyds) example
from nvidia_osd_sink_pad_buffer_probe import osd_sink_pad_buffer_probe

# path to the directory that hold the mp4 files to process.
dir_path = "/opt/nvidia/deepstream/deepstream/samples/streams"

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# List of files to process... populated at the top of main. 
file_list = []

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
    global file_list
    print('Pipeline EOS event')
    dsl_pipeline_stop('pipeline')
    
    # if there are more file to process, update the URI to the next
    # file and Play the Pipeline again.
    if (len(file_list)):
        dsl_source_uri_uri_set('uri-source', file_list.pop(0))
        dsl_pipeline_play('pipeline')

    # else, all files have be processed, time to quit.
    else:
        dsl_main_loop_quit()

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        global file_list
        
        for file in os.listdir(dir_path):
            if file.endswith(".mp4"):
                file_list.append(os.path.join(dir_path, file),)

        # New URI File Source using the filespec defined above
        retval = dsl_source_uri_new('uri-source', file_list.pop(0), False, False, 0)
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
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Custom Pad Probe Handler to call Nvidia's example callback 
        # for handling the Batched Meta Data
        retval = dsl_pph_custom_new('custom-pph', 
            client_handler=osd_sink_pad_buffer_probe, client_data=None)
        
        # Add the custom PPH to the Sink pad of the OSD
        retval = dsl_osd_pph_add('on-screen-display', 
            handler='custom-pph', pad=DSL_PAD_SINK)
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
            ['uri-source', 'primary-gie', 'iou-tracker', 
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

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
