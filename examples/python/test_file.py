################################################################################
# The MIT License
#
# Copyright (c) 2021, Prominence AI, Inc.
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
import time

from dsl import *

#-------------------------------------------------------------------------------------------
#
# This example demonstrates the use of an Image Source to infer on a single JPEG image
#
# The PTIS is added to a new Pipeline with the single URI JPF Source, KTL Tracker, 
# On-Screen-Display (OSD), and Window Sink with 1280x720 dimensions.

# File path for the single File Source
file_path = '../../test/streams/person-capture-action_00008_20230209-105848.jpeg'
#file_path = '../../test/streams/sample_720p.0.jpg'

# Filespecs for the Primary GIE and IOU Trcaker
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'

# Window Sink Dimensions
sink_width = 1280
sink_height = 720

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_player_stop('player')
        dsl_main_loop_quit()
 
    dsl_main_loop_quit()

# Function to be called on Player Termiantion event
def termination_event_listener(client_data):
    print('Player Termination event')
#    dsl_player_stop('player')
    dsl_main_loop_quit()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New Single Image Source - single frame to End of Stream.
        retval = dsl_source_image_single_new('image-source', 
            file_path = file_path)
#        retval = dsl_source_image_stream_new('image-source', 
#            file_path, False, 1, 20, 10)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval, width, height = dsl_source_video_dimensions_get('image-source')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new('window-sink', 0, 0, width, height)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retval = dsl_sink_image_multi_new('image-sink', './frame_%04d.jpg', width, height)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_player_new('player', 'image-source', 'image-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

# Add all the components to a new pipeline
#        retval = dsl_pipeline_new_component_add_many('pipeline', 
#            ['image-source', 'image-sink', None])
#        if retval != DSL_RETURN_SUCCESS:
#            break

#        retval = dsl_pipeline_streammux_dimensions_set('pipeline', width, height)
#        if retval != DSL_RETURN_SUCCESS:
#            break
        
        retval = dsl_player_xwindow_key_event_handler_add("player", 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
#        retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline", 
#            xwindow_key_event_handler, None)
#        if retval != DSL_RETURN_SUCCESS:
#            break
#        retval = dsl_pipeline_xwindow_delete_event_handler_add("pipeline", 
#            xwindow_delete_event_handler, None)
#        if retval != DSL_RETURN_SUCCESS:
#            break

        # Add the listener callback functions defined above
#        retval = dsl_pipeline_state_change_listener_add('pipeline', state_change_listener, None)
#        if retval != DSL_RETURN_SUCCESS:
#            break
#        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
#        if retval != DSL_RETURN_SUCCESS:
#            break
        retval = dsl_player_termination_event_listener_add('player', 
            termination_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the Player
        retval = dsl_player_play('player')
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
