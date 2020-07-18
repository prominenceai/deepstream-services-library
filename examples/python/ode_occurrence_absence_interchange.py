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
import time

uri_file = "../../test/streams/sample_1080p_h264.mp4"

# Filespecs for the Primary GIE and IOU Tracker
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
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
    if (new_state == DSL_STATE_PLAYING):
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        # This example is used to demonstrate the Use of Two Triggers, one Occurrence and one Absence, to
        # interchange between detecting the first Person coming into view and the last Person to leave view
        # i.e the new Occurrence trigger interchanging with New Absence trigger. This can be accomplished
        # by setting the Limit of each Trigger to one, and then using a Reset-Trigger action to reset the 
        # the other Trigger on Occurrence. I.e. Occurrence resets Absence and Absence resets Occurrence
        # 
        # Note: this Video is a poor choice for the purpose of this example, and should be updated in the 
        # future. You will need to let the Video run until the people become small enough for the GIE to 
        # start missing on the inference for the Triggers sto start interchanging. 
        
        # Each of the two Triggers will have three actions to invoke on ODE occurrence: 
        #    1. A-visual flash by filling in the full frame with a semi-opaque background. 
        #       Red for new occurrence, White flash for new absence
        #    2. Print the ODE occurrence data to the console
        #    3. Reset the other Trigger
        
        # This example uses other Triggers for the Purpose of displaying the Person Count on the display,
        # as well as hiding Object display-text and borders.
        
        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````

        # Create a new Reset-Trigger Action to reset the Person-Occurrence Trigger that has a limit of one.
        # This Action will be added to / invoked by the Person-Absence Trigger that has a limit of one as well.
        retval = dsl_ode_action_trigger_reset_new('reset-occurrence', 'new-person-occurrence')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Reset-Trigger Action to reset the Person-Absence Trigger that has a limit of one.
        # This Action will be added to / invoked by the Person-Occurrence Trigger that has a limit of one as well.
        retval = dsl_ode_action_trigger_reset_new('reset-absence', 'new-person-absence')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create a Fill-Area Action to simulate a 'red-flash' as a visual indicator that a new Occurrence
        # has occurred. This Action will used by the Occurrence Trigger only
        retval = dsl_ode_action_fill_frame_new('occurrence-flash-action', red=1.0, blue=0.0, green=0.0, alpha=0.6)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Fill-Area Action to simulate a 'white-flash' as a visual indicator that a new Absence
        # has occurred. This Action will used by the Absence Trigger only
        retval = dsl_ode_action_fill_frame_new('absence-flash-action', red=1.0, blue=1.0, green=1.0, alpha=0.3)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Action used to display all Object detection summations for each frame. 
        retval = dsl_ode_action_display_new('display-action', offsetX=48, offsetY=60, offsetY_with_classId=False)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Action to hide the display text for each detected object
        retval = dsl_ode_action_hide_new('hide-text-action', text=True, border=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Action to hide both the display text and border for each detected object
        retval = dsl_ode_action_hide_new('hide-both-action', text=True, border=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Print Action to print out the ODE occurrence information to the console
        retval = dsl_ode_action_print_new('print-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create the New Occurrence and New Absence Triggers, and add the Actions to Flash, Print and Reset the other.
        retval = dsl_ode_trigger_occurrence_new('new-person-occurrence', class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_ONE )
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('new-person-occurrence', actions=
            ['occurrence-flash-action', 'print-action', 'reset-absence', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_trigger_absence_new('new-person-absence', class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_ONE )
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('new-person-absence', actions=
            ['absence-flash-action', 'print-action', 'reset-occurrence', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create the Summation and Occurrence Triggers to display the Object Count and Hide each Object's Display Text
        
        # New ODE Trigger for Person summation - i.e. new ODE occurrence on Person summation for each frame.
        # Note: The Display-Action will use the Trigger's unique name as the label for the summation display
        retval = dsl_ode_trigger_summation_new('Person count:', class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('Person count:', action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE occurrence Trigger to hide the Display Text and Border for all vehicles
        retval = dsl_ode_trigger_occurrence_new('every-vehicle', class_id=PGIE_CLASS_ID_VEHICLE, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('every-vehicle', action='hide-both-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE occurrence Trigger to hide the Display Text and Border for all bicycles
        retval = dsl_ode_trigger_occurrence_new('every-bicycle', class_id=PGIE_CLASS_ID_BICYCLE, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('every-bicycle', action='hide-both-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE occurrence Trigger to hide just the Display Text for every person, we will leave the border visible
        retval = dsl_ode_trigger_occurrence_new('every-person', class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('every-person', action='hide-text-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_ode_handler_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add_many('ode-handler', triggers=[
            'new-person-occurrence',
            'new-person-absence',
            'Person count:',
            'every-vehicle',
            'every-bicycle',
            'every-person',
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
            ['uri-source', 'primary-gie', 'iou-tracker', 'tiler', 'ode-handler', 'on-screen-display', 'window-sink', None])
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
