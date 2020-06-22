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
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_fp16.engine'
tracker_config_file = '../../test/configs/iou_config.txt'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MIN_OBJECTS = 3
MAX_OBJECTS = 8

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

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        # This example is used to demonstrate the Use of Minimum, Maximum, and Range Triggers.
        # The triggers, upon meeting all criteria, will fill a rectangle Area on the Frame 
        # with color indicating: 
        #    Yellow = object count below Minimum
        #    Red = object count above Maximum 
        #    Green = object count in range of Minimim to Maximum.
        
        # A secondary indicatory of filling the full Frame with a shade of red will be used
        # to stress that the object count within the frame has exceeded the Maximum
        
        # An additional Summation Trigger with Display Action will display the total number of objects 
        # next to the colored/filled area-indicator
        
        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        
        # Create a Fill-Area Action to simulate a 'camera-flash' as a visual indicator that an Object Image Capture 
        # has occurred. This Action will be shared between all Triggers created for image capture
        retval = dsl_ode_action_fill_frame_new('camera-flash-action', red=1.0, blue=0.8, green=1.0, alpha=0.7)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Action to Capture a detected object to jpeg file. This Action will be shared between all 
        # Triggers created for image capture, each Trigger with its own criteria.
        retval = dsl_ode_action_capture_object_new('capture-object-action', outdir='./')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Print Action to print out the Capture Object's Attributes and Trigger information
        retval = dsl_ode_action_print_new('print-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Create an Area that defines a vertical rectangle to the right of the sidewalk and left of the street
        # This area's background will be displayed shaded a default color of white. The Area will be added
        # To the 'person-occurrence' Trigger as criteria for ODE occurrence
        retval = dsl_ode_area_new('person-criteria-area', left=520, top=0, width=40, height=1089, display=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create three new Occurrence triggers, each with their own specific criteria

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New Vehicle occurrence Trigger with a limit of one, with Minimum Dimensions set. 
        # The default minimum width and height values are set to 0 = disabled on creation.
        retval = dsl_ode_trigger_occurrence_new('vehicile-occurrence', 
            class_id=PGIE_CLASS_ID_VEHICLE, limit=DSL_ODE_TRIGGER_LIMIT_ONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        # Set the minimum object rectangle dimensions criteria, in pixels, required to Trigger ODE occurrence
        retval = dsl_ode_trigger_dimensions_min_set('vehicile-occurrence', min_width=250, min_height=250)
        if retval != DSL_RETURN_SUCCESS:
            break
        # Set the inferrence-done-only criteria, since we are capturing images using the object's rectangle
        # dimensions, we want to make sure the dimensions are for the current frame. Tracked frames use the
        # rectangle dimensions from the last Inference, i.e. the last frame with the bInferDone meta flag set
        retval = dsl_ode_trigger_infer_done_only_set('vehicile-occurrence', infer_done_only=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New Person occurrence Trigger with a limit of one, with the ODE Area created above as critera. 
        retval = dsl_ode_trigger_occurrence_new('person-occurrence', 
            class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_ONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        # Set the inference-done-only criteria for the person occurrence as well
        retval = dsl_ode_trigger_infer_done_only_set('person-occurrence', infer_done_only=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        # Add the area-of-overlap critera, requiring at least one pixel of overlap between object rectangle and Area
        retval = dsl_ode_trigger_area_add('person-occurrence', 'person-criteria-area')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # *** Important Note ***
        # see https://forums.developer.nvidia.com/t/nvinfer-is-not-populating-confidence-field-in-nvdsobjectmeta-ds-4-0/79319/20
        # for the required DS 4.02 patch instruction to populate the confidence values in object's meta data structure
        
        # New Bicycle occurrence Trigger with a limit of one, with Minimum Confidence set. 
        # The default minimum confidence value is set to 0 = disabled on creation.
        retval = dsl_ode_trigger_occurrence_new('bicycle-occurrence', 
            class_id=PGIE_CLASS_ID_BICYCLE, limit=DSL_ODE_TRIGGER_LIMIT_ONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        # Set the minimum Inference Confidence required to Trigger ODE occurrence.
        retval = dsl_ode_trigger_confidence_min_set('bicycle-occurrence', min_confidence=0.05)
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, we will add all three of our Actions defined above to each of our Trgggers

        retval = dsl_ode_trigger_action_add_many('vehicile-occurrence', actions=[
            'camera-flash-action', 'capture-object-action', 'print-action',
            None] )
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_trigger_action_add_many('person-occurrence', actions=[
            'camera-flash-action', 'capture-object-action', 'print-action', None] )
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_ode_handler_new('ode-hanlder')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add_many('ode-hanlder', triggers=[
            'vehicile-occurrence',
            'person-occurrence',
            'bicycle-occurrence',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # New URI File Source using the filespec defined above
        retval = dsl_source_uri_new('uri-source', uri_file, False, 0, 0, 0)
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

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker', 'tiler', 'ode-hanlder', 'on-screen-display', 'window-sink', None])
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
