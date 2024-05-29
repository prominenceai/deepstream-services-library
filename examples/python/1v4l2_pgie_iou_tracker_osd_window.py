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
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - V4L2 Source - Web Camera
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#
# The key-release handler function will update the V4L2 device picture settings
# based on the key value as follows during runtime.
#   * brightness - or more correctly the black level. 
#                  enter 'B' to increase, 'b' to decrease
#   * contrast   - color contrast setting or luma gain.
#                  enter 'C' to increase, 'c' to decrease
#   * hue        - color hue or color balence.
#                  enter 'H' to increase, 'h' to decrease
#
# The Picture Settings are all integer values, range 
################################################################################

#!/usr/bin/env python

import sys
import time
from dsl import *

# Picture settings, read after device negotiation.
brightness=0 
contrast=0
hue=0 

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):

    global brightness, contrast, hue
    print('key released = ', key_string)
    update = False
    
    # Upper case 'B' - increase the picture brighness by 10
    if key_string == 'B':
        brightness += 10
        update = True
        
    # Lower case 'b' - decrease the picture brighness by 10
    elif key_string == 'b':
        brightness -= 10
        update = True
        
    # Upper case 'C' - increase the picture contrast (luma gain) by 10
    elif key_string == 'C':
        contrast += 10
        update = True
        
    # Lower case 'c' - decrease the picture contrast (luma gain) by 10
    elif key_string == 'c':
        contrast -= 10
        update = True
        
    # Upper case 'H' - increase the picture hue (color balence) by 10
    elif key_string == 'H':
        hue += 10
        update = True
        
    # Lower case 'h' - decrease the picture hue (color balence) by 10
    elif key_string == 'h':
        hue -= 10
        update = True

    elif key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()

    if update:
        retval = dsl_source_v4l2_picture_settings_set('v4l2-source',
            brightness, contrast, hue)

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

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        # New V4L2 Live Web Camera Source
        retval = dsl_source_v4l2_new('v4l2-source', 
            '/dev/video0')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with inference interval=0
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

        # New 3D Window Sink with 0 x/y offsets, and same dimensions as Camera Source
        # EGL Sink runs on both platforms. 3D Sink is Jetson only.
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED):
            retval = dsl_sink_window_3d_new('window-sink', 0, 0, 
                WINDOW_WIDTH, WINDOW_HEIGHT)
        else:
            retval = dsl_sink_window_egl_new('window-sink', 0, 0, 
                WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add("window-sink", 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add("window-sink", 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['v4l2-source', 'primary-gie', 'iou-tracker', 'on-screen-display', 
            'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_eos_listener_add('pipeline', 
            eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Once playing, we can retrieve the device information: 
        # name, file-descriptor, and device-flags
        retval, device_name = dsl_source_v4l2_device_name_get('v4l2-source')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval, device_fd = dsl_source_v4l2_device_fd_get('v4l2-source')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval, device_flags = dsl_source_v4l2_device_flags_get('v4l2-source')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Global variables for picture settings - updated in key-release handler
        global brightness, contrast, hue

        # Get the initial values updated after the Pipeline is playing.
        retval, brightness, contrast, hue = \
            dsl_source_v4l2_picture_settings_get('v4l2-source')
        if retval != DSL_RETURN_SUCCESS:
            break

        print('V4L2 Device Propertes')
        print('   Name       :', device_name)
        print('   File Desc  :', device_fd)
        print('   Flags      :', device_flags)
        print('   Brightness :', brightness)
        print('   Contrast   :', contrast)
        print('   Hue        :', hue)

        # Start and join the main loop 
        dsl_main_loop_run()
        
        # main loop has exited update the final return value to succes
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))

