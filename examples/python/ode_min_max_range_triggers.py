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
        
        # New Rectangle Area to be filled on Minumum/Maximum/Range ODE Trigger occurrences
                retval = dsl_ode_area_new('indicator', left=10, top=60, width=34, height=34, display=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create all Fill-Area, Fill-Frame, Display and Hide Actions
        
        # New Action to fill the entire frame with a light shade of red - indicating object summation above maximum
        retval = dsl_ode_action_fill_area_new('fill-area-red', 'indicator', red=1.0, blue=0.0, green=0.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Action to fill the entire frame with a light shade of yellow - indicating object summation below minimium
        retval = dsl_ode_action_fill_area_new('fill-area-yellow', 'indicator', red=1.0, blue=0.0, green=1.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Action to fill the indicator with a full shade of green - indicating object summation with in range
        retval = dsl_ode_action_fill_area_new('fill-area-geen', 'indicator', red=0.0, blue=0.0, green=1.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Action to fill the entire frame with a light shade of red - a secondary indication of object summation above maximum
        retval = dsl_ode_action_fill_frame_new('shade-frame-red', red=1.0, blue=0.0, green=0.0, alpha=0.2)
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


        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create Maximum, Minimum and Range Triggers, while adding their corresponding Fill colors

        # New Maximum occurrence Trigger, with class id filter disabled, and with no limit on the number of occurrences
        retval = dsl_ode_trigger_maximum_new('maximum-objects', 
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE, maximum=MAX_OBJECTS)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('maximum-objects', actions=
            ['shade-frame-red', 'fill-area-red', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Minimum occurrence Trigger, with class id filter disabled, and with no limit on the number of occurrences
        retval = dsl_ode_trigger_minimum_new('minimum-objects', 
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE, minimum=MIN_OBJECTS)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('minimum-objects', action='fill-area-yellow')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Range of occurrence Trigger, with class id filter disabled, and with no limit on the number of occurrences
        retval = dsl_ode_trigger_range_new('range-of-objects', class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE, 
            lower=MIN_OBJECTS, upper=MAX_OBJECTS)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('range-of-objects', action='fill-area-geen')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create the Summation and Occurrence Triggers to display the Object Count and Hide each Object's Display Text
        
        # New ODE Trigger for Object summation - i.e. new ODE occurrence on detection summation for each frame.
        retval = dsl_ode_trigger_summation_new('Objects', class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('Objects', action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE occurrence Trigger to hide the Display Text for all detected objects
        retval = dsl_ode_trigger_occurrence_new('every-object', class_id=DSL_ODE_ANY_CLASS, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('every-object', action='hide-text-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_ode_handler_new('ode-hanlder')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add_many('ode-hanlder', triggers=[
            'maximum-objects',
            'minimum-objects',
            'range-of-objects',
            'Objects',
            'every-object',
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
