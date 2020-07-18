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
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
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
# Function called on "first occurrence" of the bicycle
## 
def update_tiler(ode_id, trigger, buffer, frame_data, objec_data, client_data):
    
    print('Setting tiller dimensions in prep for new source, retval = ', 
        dsl_return_value_to_string(dsl_tiler_dimensions_set('tiler', width=1280, height=360)))
        
        

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Create a new set of Actions to manipulate our Pipeline on First occurrence of the Bicycle
        
        retval = dsl_ode_action_callback_new('update-tiler-action', client_handler=update_tiler, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_action_source_add_new('source-add-action', pipeline='pipeline', source='new-uri-source')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_action_sink_add_new('sink-add-action', pipeline='pipeline', sink='overlay-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_action_sink_remove_new('sink-remove-action', pipeline='pipeline', sink='new-uri-source')
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, we'll create two Bicycle Occurrence Triggers, one for each of our two sources
        # We need to explicitly set the trigger's "source id-filter" to as the default value = DSL_ODE_ANY_SOURCE
        
        retval = dsl_ode_trigger_occurrence_new('bicycle-occurrence-trigger-s0', class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_source_id_set('bicycle-occurrence-trigger-s0', source_id=0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_trigger_occurrence_new('bicycle-occurrence-trigger-s1', class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_source_id_set('bicycle-occurrence-trigger-s1', source_id=1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, we'll add the actions to our two Bicycle Occurence Trigger.
        
        # The Source-0 Trigger will invoke the Add Source Action, followed by the Update Tiler Action
        retval = dsl_ode_trigger_action_add_many('bicycle-occurrence-trigger-s0',
            actions=['update-tiler-action', 'source-add-action', None])
        if retval != DSL_RETURN_SUCCESS:
            break
    
        # The Source-1 Trigger will invoke the Add and Remove Sink Actions
        retval = dsl_ode_trigger_action_add_many('bicycle-occurrence-trigger-s1',
            actions=['sink-add-action', None])
#            actions=['sink-add-action', 'sink-remove-action', None])
        if retval != DSL_RETURN_SUCCESS:
            break
    
        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New ODE Handler to handle the Triggers and their Actions    
        retval = dsl_ode_handler_new('ode-hanlder')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add_many('ode-hanlder',
            triggers=['bicycle-occurrence-trigger-s0', 'bicycle-occurrence-trigger-s1', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # Two URI File Sources using the filespec defined above, one added prior to Playing the Pipeline
        # The second will be added on first occurrence of a bicycle ### IMPORTANT: See batch properites below
        retval = dsl_source_uri_new('initial-uri-source', uri_file, is_live=False, cudadec_mem_type=0, intra_decode=0, drop_frame_interval=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('new-uri-source', uri_file, is_live=False, cudadec_mem_type=0, intra_decode=0, drop_frame_interval=0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 4)
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

        retval = dsl_sink_overlay_new('overlay-sink', 1, 0, 0, offsetX=100, offsetY=100, width=1280, height=360)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline - except for our second source and overlay sink 
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['initial-uri-source', 'primary-gie', 'iou-tracker', 'tiler', 'ode-hanlder', 'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ### IMPORTANT: ###
        
        # we need to explicitely set the stream-muxer Batch properties, otherwise the Pipeline
        # will use the current number of Sources when set to Playing, which would be 1 and too small
        retval = dsl_pipeline_streammux_batch_properties_set('pipeline', batch_size=2, batch_timeout=4000000)
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
    