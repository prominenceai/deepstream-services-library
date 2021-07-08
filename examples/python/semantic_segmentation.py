################################################################################
# The MIT License
#
# Copyright (c) 2021-2021, Prominence AI, Inc.
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

#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")
import time
from dsl import *

# NOTE: the below example video runs at 15/1 frame-rate which seems to be beyond 
# the limit of the PGIE to perform segmentation. This becomes apparent half-way
# through the video when the Pipeline starts experiencing QOS issues. A slower
# camera rate may be required.
input_file = "../../test/streams/nevskiy-avenue-night3.mp4"

# Filespecs for the Primary GIE
primary_infer_config_file = '../../test/configs/segvisual_config_semantic.txt'
primary_model_engine_file = '../../test/models/Segmentation/semantic/unetres18_v4_pruned0.65_800_data.uff_b1_gpu0_fp16.engine'

# Segmentation Visualizer output dimensions should (typically) match the
# inference dimensions defined in segvisual_config_semantic.txt (512x512)
width = 512
height = 512

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
    dsl_main_loop_quit()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New File Source
        retval = dsl_source_file_new('file-source', file_path=input_file, repeat_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new('primary-gie', False,
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Segmentation Visualizer with output dimensions
        retval = dsl_segvisual_new('segvisual', width=width, height=height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 100, 1000, width, height)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Example of how to force the aspect ratio during window resize
        dsl_sink_window_force_aspect_ratio_set('window-sink', force=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['file-source', 'primary-gie', 'segvisual', 'window-sink', None])
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

