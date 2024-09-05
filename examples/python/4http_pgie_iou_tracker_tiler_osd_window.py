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
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - Four HTTP URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - 2D Tiler
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - source-buffering messages
#   - key-release events
#   - delete-window events
#
# When using non-live streaming sources -- like the HTTP URI in this example --
# the application should pause the Pipeline when ever a Source is buffering. The 
# buffering_message_handler() callback funtion is added to the Pipeline to
# be called when a buffering-message is recieved on the Pipeline bus.
# The callback input parameters are 
#    - source - Source of the message == <source-name>-uridecodebin
#    - percent - the current buffer size as a percentage of the high watermark.
#    - client_data - unused in this simple example
# When a buffering message is received (percent < 100) the calback will pause
# the Pipeline. When a buffering message with 100% is received the callback
# resumes the Pipeline playback,
#
################################################################################

#!/usr/bin/env python

import sys
import time
from dsl import *


source_uri = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4'

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

# Tiler Output Dimensions
TILER_WIDTH = 1920
TILER_HEIGHT = 1080

# Window Sink Dimensions
WINDOW_WIDTH = TILER_WIDTH
WINDOW_HEIGHT = TILER_HEIGHT

# Simple flag to track the current buffering state
buffering = False

## 
# Function to be called when a buffering-message is received on the Pipeline bus.
## 
def buffering_message_handler(source, percent, client_data):

    global buffering

    if percent == 100:
        print('playing pipeline - buffering complete at 100 % for Source', source)
        dsl_pipeline_play('pipeline')
        buffering = False

    else:
        if not buffering:
            print('pausing pipeline - buffering starting at ', percent,
                '% for Source', source)
            dsl_pipeline_pause('pipeline')
        buffering = True

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

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        ## Four new URI File Sources using our single HTTP URI.
        retval = dsl_source_uri_new('uri-source-1', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-2', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-3', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-4', source_uri, False, False, 0)
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

        # New Tiled Display, setting width and height, use default cols/rows 
        # set by source count
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
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
            ['uri-source-1', 'uri-source-2', 'uri-source-3', 'uri-source-4', 
             'primary-gie', 'iou-tracker', 'tiler', 'on-screen-display', 
             'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ## Add the Pipeline callback functions defined above
        retval = dsl_pipeline_buffering_message_handler_add('pipeline',
            buffering_message_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
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

        # Start and join the main loop 
        dsl_main_loop_run()

        # done so break out of while
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))

