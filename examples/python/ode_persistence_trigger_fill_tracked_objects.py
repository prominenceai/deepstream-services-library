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

#!/usr/bin/env python

import sys
from dsl import *

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

WINDOW_WIDTH = DSL_1K_HD_WIDTH
WINDOW_HEIGHT = DSL_1K_HD_HEIGHT

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
    
        # This example demonstrates the use of three ODE Persistence Triggers to trigger on
        # all tracked Objects - as identified by an IOU Tracker - that persist accross consecutive
        # frames for a specifid period of time. Each trigger specifies a range of minimum and
        # maximum times of persistence. 
        #   Trigger 1: 0 - 3 seconds - action = fill object with opaque green color
        #   Trigger 2: 3 - 6 seconds - action = fill object with opaque yellow color
        #   Trigger 3: 6 - 0 seconds - action = fill object with opaque red color
        # This will have the effect of coloring an object by its time in view
        
        #```````````````````````````````````````````````````````````````````````````````````
        # Create a Format Label Action to remove the Object Label from view
        # Note: the label can be disabled with the OSD API as well. 
        retval = dsl_ode_action_label_format_new('remove-label', 
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a Format Bounding Box Action to remove the box border from view
        retval = dsl_ode_action_bbox_format_new('remove-border', border_width=0,
            border_color=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Any-Class Occurrence Trigger for our remove label and bbox actions
        retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger', source='uri-source-1',
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger', 
            actions=['remove-label', 'remove-border', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Create three new RGBA fill colors to fill the bounding boxes of new objects
        retval = dsl_display_type_rgba_color_custom_new('opaque-green', red=0.0, green=1.0, blue=0.0, alpha=0.3)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('opaque-yellow', red=1.0, green=1.0, blue=0.0, alpha=0.3)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_display_type_rgba_color_custom_new('opaque-red', red=1.0, green=0.0, blue=0.0, alpha=0.3)
        if retval != DSL_RETURN_SUCCESS:
            break
            
            
        #```````````````````````````````````````````````````````````````````````````````````
        # Create three new Actions to fill the bounding boxes, one for each Persistence Trigger
        retval = dsl_ode_action_bbox_format_new('fill-opaque-green',
            border_width = 0,
            border_color = None,
            has_bg_color = True,
            bg_color = 'opaque-green')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_action_bbox_format_new('fill-opaque-yellow',
            border_width = 0,
            border_color = None,
            has_bg_color = True,
            bg_color = 'opaque-yellow')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_action_bbox_format_new('fill-opaque-red',
            border_width = 0,
            border_color = None,
            has_bg_color = True,
            bg_color = 'opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Create the three persistence triggers for the PERSON class, each with their unique range
        # Set the minimum hight critera - we only care about people that are near the Camera
        retval = dsl_ode_trigger_persistence_new('minimum-persitence-trigger', source='uri-source-1',
            class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_NONE, minimum=0, maximum=2)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_dimensions_min_set('minimum-persitence-trigger', 
            min_width=0, min_height=100)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_persistence_new('medium-persitence-trigger', source='uri-source-1',
            class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_NONE, minimum=2, maximum=4)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_dimensions_min_set('medium-persitence-trigger', 
            min_width=0, min_height=100)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_persistence_new('maximum-persitence-trigger', source='uri-source-1',
            class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_NONE, minimum=4, maximum=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_dimensions_min_set('maximum-persitence-trigger', 
            min_width=0, min_height=100)
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Next, we add our Actions to our Triggers
        retval = dsl_ode_trigger_action_add('minimum-persitence-trigger', action='fill-opaque-green')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('medium-persitence-trigger', action='fill-opaque-yellow')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('maximum-persitence-trigger', action='fill-opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # New ODE Handler to handle all ODE Triggers    
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers=[
            'every-occurrence-trigger', 
            'minimum-persitence-trigger', 
            'medium-persitence-trigger', 
            'maximum-persitence-trigger',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        ####################################################################################
        #
        # Create the remaining Pipeline components
        
        # New URI File Source using the filespec defined above
        retval = dsl_source_uri_new('uri-source-1', uri_h265, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

         # Add our ODE Pad Probe Handler to the Sink pad of the OSD
        retval = dsl_osd_pph_add('on-screen-display', 
            handler='ode-handler', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and dimensions
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to the Window Sink
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
            ['uri-source-1', 'primary-gie', 'iou-tracker',
            'on-screen-display', 'egl-sink', None])
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
