################################################################################
# The MIT License
#
# Copyright (c) 2021-2023, Prominence AI, Inc.
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
# The example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary Triton Inference Server (PTIS)
#   - NcDCF Low Level Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
################################################################################

#!/usr/bin/env python

import sys
import time

from dsl import *

# File path for the single File Source
file_path = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4'

# Filespec for the Primary Triton Inference Server (PTIS) config file
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt'

# Filespec for the NvDCF Tracker config file
dcf_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml'

# Source file dimensions are 960 × 540 - use this to set the Streammux dimensions.
SOURCE_WIDTH = 960
SOURCE_HEIGHT = 540

# Window Sink dimensions same as Streammux dimensions - no scaling.
WINDOW_WIDTH = SOURCE_WIDTH
WINDOW_HEIGHT = SOURCE_HEIGHT

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

# Function to be called on End-of-Stream (EOS) event
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

        # New File Source using the file path specified above, repeat diabled.
        retval = dsl_source_file_new('file-source', file_path, False)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Primary TIS using the filespec specified above, with interval = 3
        # Note: need to set the interval to greater than 1 as the DCF Tracker
        # add a medimum level GPU computational load vs. the KTL and IOU CPU Trackers.
        retval = dsl_infer_tis_primary_new('primary-tis', primary_infer_config_file, 3)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New NvDCF Tracker, setting operation width and height
        # NOTE: width and height paramaters must be multiples of 32 for dcf
        retval = dsl_tracker_new('dcf-tracker', 
            config_file = dcf_tracker_config_file,
            width = 640, 
            height = 384) 
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink with x/y offsets and dimensions.
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

        # Add all the components to a new pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['file-source', 'primary-tis', 'dcf-tracker', 'on-screen-display', 
            'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Update the Pipeline's Streammux dimensions to match the source dimensions.
        retval = dsl_pipeline_streammux_dimensions_set('pipeline',
            SOURCE_WIDTH, SOURCE_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the listener callback functions defined above
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

        # Join with main loop until released - blocking call
        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
