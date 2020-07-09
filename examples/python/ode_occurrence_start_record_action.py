################################################################################
# The MIT License
#
# Copyright (c) 2019-2020, Robert Howell. All rights reserved.
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
from dsl import *

uri_file = "../../test/streams/sample_1080p_h264.mp4"

# Filespecs for the Primary GIE and IOU Trcaker
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b4_gpu0_fp16.engine'
tracker_config_file = '../../test/configs/iou_config.txt'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '':
        dsl_main_loop_quit()
 
## 
# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_main_loop_quit()

## 
# Function to be called on End-of-Stream (EOS) event
## 
def eos_event_listener(client_data):
    print('Pipeline EOS event')
    dsl_main_loop_quit()

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

## 
# Function to be called on every change of Pipeline state
## 
def record_complete_listener(session_info, client_data):
    print(' ***  Recording Complete  *** ')
    
    # reset the Trigger so that a new session can be started.
    print(dsl_return_value_to_string(dsl_ode_trigger_reset('bicycle-occurrence-trigger')))


def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # This example is used to demonstrate the use of a First Occurrence Trigger and a Start Record Action
        # to control a Record Sink.  A callback function, called on completion of the recording session, will
        # reset the Trigger allowing a new session to be started on next occurrence.
        # Addional actions are added to "Capture" the frame to an image-file and "Fill" the frame red as a visual marker.

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New Record-Sink that will buffer encoded video while waiting for the ODE trigger/action, defined below, 
        # to start a new session on first occurrence. The default 'cache-size' and 'duration' are defined in
        # Setting the bit rate to 12 Mbps for 1080p ??? 
        retval = dsl_sink_record_new('record-sink', outdir="./", codec=DSL_CODEC_H265, container=DSL_CONTAINER_MKV, 
            bitrate=12000000, interval=0, client_listener=record_complete_listener)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````

        # Create a Fill-Area Action as a visual indicator to identify the frame that triggered the recording
        retval = dsl_ode_action_fill_frame_new('red-flash-action', red=1.0, blue=0.5, green=0.5, alpha=0.7)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new Capture Action to capture the full-frame to jpeg image, and save to file. 
        # The action will be triggered on firt occurrence of a bicycle and will be saved to the current dir.
        retval = dsl_ode_action_capture_frame_new('bicycle-capture-action', outdir="./")
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create a new Capture Action to start a new record session
        retval = dsl_ode_action_sink_record_start_new('start-record-action', 
            record_sink='record-sink', start=2, duration=10, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create the Bicycle Occurrence Trigger. We will reset the trigger in the recording complete callback
        retval = dsl_ode_trigger_occurrence_new('bicycle-occurrence-trigger', class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Add the actions to our Bicycle Occurence Trigger.
        retval = dsl_ode_trigger_action_add_many('bicycle-occurrence-trigger',
            actions=['red-flash-action', 'bicycle-capture-action', 'start-record-action', None])
        if retval != DSL_RETURN_SUCCESS:
            break
    
        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New ODE Handler for our Trigger
        retval = dsl_ode_handler_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add('ode-handler', 'bicycle-occurrence-trigger')
        if retval != DSL_RETURN_SUCCESS:
            break
    
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        retval = dsl_source_uri_new('uri-source', uri_file, is_live=False, cudadec_mem_type=0, intra_decode=0, drop_frame_interval=0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New KTL Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new('iou-tracker', tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiled Display, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new('tiler', 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break
 
        # New OSD with clock enabled... .
        retval = dsl_osd_new('on-screen-display', True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline - except for our second source and overlay sink 
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker', 'tiler', 'ode-handler', 'on-screen-display', 'window-sink', 'record-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline", xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_delete_event_handler_add("pipeline", xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', state_change_listener, None)
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

    # Cleanup all DSL/GST resources
    dsl_delete_all()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
    