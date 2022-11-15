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

# ------------------------------------------------------------------------------------
# This example demonstrates interpipe dynamic switching. Four DSL Players
# are created, each with a File Source and Interpipe Sink. A single
# inference Pipeline with an Interpipe Source is created as the single listener
# 
# The Interpipe Source's "listen_to" setting is updated based on keyboard input.
# The xwindow_key_event_handler (see below) is added to the Pipeline's Window Sink.
# The handler, on key release, sets the "listen_to" setting to the Interpipe Sink
# name that corresponds to the key value - 1 through 4.

#!/usr/bin/env python

import sys
import time

from dsl import *

uri1 = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov'
uri2 = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_push.mov'
uri3 = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov'
uri4 = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov'

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
tracker_config_file = '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)

    if key_string >= '1' and key_string <= '4':
        dsl_source_interpipe_listen_to_set('inter-pipe-source',
            listen_to='inter-pipe-sink-'+key_string)
            
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
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        
        # Four new file sources using the filespecs defined above
        retval = dsl_source_file_new('file-source-1', uri1, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('file-source-2', uri2, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('file-source-3', uri3, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('file-source-4', uri4, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Four new inter-pipe sinks
        retval = dsl_sink_interpipe_new('inter-pipe-sink-1', 
            forward_eos=True, forward_events=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_interpipe_new('inter-pipe-sink-2', 
            forward_eos=True, forward_events=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_interpipe_new('inter-pipe-sink-3', 
            forward_eos=True, forward_events=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_interpipe_new('inter-pipe-sink-4', 
            forward_eos=True, forward_events=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Four new Players, each with a file source and inter-pipe sink
        retval = dsl_player_new('player-1', 'file-source-1', 'inter-pipe-sink-1')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_player_new('player-2', 'file-source-2', 'inter-pipe-sink-2')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_player_new('player-3', 'file-source-3', 'inter-pipe-sink-3')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_player_new('player-4', 'file-source-4', 'inter-pipe-sink-4')
        if retval != DSL_RETURN_SUCCESS:
            break

        #-----------------------------------------------------------------------------
        # Create the Inference Pipeline with an inter-pipe source that can
        # dynamically switch between the four players and their inter-pipe sink
        
        # New inter-pipe source - listen to inter-pipe-sink-1 to start with.
        retval = dsl_source_interpipe_new('inter-pipe-source',
            listen_to='inter-pipe-sink-1', is_live=False,
            accept_eos=True, accept_events=True)
        
        # New Primary GIE's using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker setting max width and height of input frame
        retval = dsl_tracker_new('iou-tracker', tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD's with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', components=[
            'inter-pipe-source', 'primary-gie', 'iou-tracker', 'on-screen-display',
            'window-sink', None])
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

        ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Start the four players first (although, it's safe to start the Pipeline first)
        retval = dsl_player_play('player-1')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_player_play('player-2')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_player_play('player-3')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_player_play('player-4')
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

    # Cleanup all DSL/GST resources
    dsl_delete_all()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
