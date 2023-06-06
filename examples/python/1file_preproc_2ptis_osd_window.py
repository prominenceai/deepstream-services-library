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
#   - Preprocessor
#   - 2 Primary Triton Inference Server (PTIS)
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

# Config file used with the Preprocessor
preproc_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_preprocess.txt'

# Config file for the first Primary Triton Inference Server (PTIS)
primary_infer_config_file_1 = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt'

# Config file for the second Primary Triton Inference Server (PTIS)
primary_infer_config_file_2 = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-triton/config_infer_plan_engine_primary.txt'

# Source file dimensions are 960 × 540 - use this to set the Streammux dimensions.
source_width = 960
source_height = 540

# Window Sink dimensions same as Streammux dimensions - no scaling.
sink_width = source_width
sink_height = source_height

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
            
        # New Preprocessor component using the config filespec defined above.
        retval = dsl_preproc_new('preprocessor', preproc_config_file)
        if retval != DSL_RETURN_SUCCESS:
            break

        # First new Primary TIS using the config file-1 specified above, with interval = 0
        retval = dsl_infer_tis_primary_new('primary-tis-1', 
            primary_infer_config_file_1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Second new Primary TIS using the config file-2 specified above, with interval = 0
        retval = dsl_infer_tis_primary_new('primary-tis-2', 
            primary_infer_config_file_1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new('window-sink', 0, 0, sink_width, sink_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to a new pipeline
#        retval = dsl_pipeline_new_component_add_many('pipeline', 
#            ['file-source', 'preprocessor', 'primary-tis-1', 'primary-tis-2', 
#            'on-screen-display', 'window-sink', None])

        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['file-source', 'primary-tis-1', 'primary-tis-2', 
            'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Update the Pipeline's Streammux dimensions to match the source dimensions.
        retval = dsl_pipeline_streammux_dimensions_set('pipeline',
            source_width, source_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline", 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_delete_event_handler_add("pipeline", 
            xwindow_delete_event_handler, None)
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
