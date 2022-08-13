################################################################################
# The MIT License
#
# Copyright (c) 2019-2021, Prominence AI, Inc.
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

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
tracker_config_file = '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

# Object count tiers 1 and 2
T1_OBJECT_COUNT = 5
T2_OBJECT_COUNT = 8

TILER_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH
TILER_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

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
        
        # New RGBA color types to be used for our object count indicator
        retval = dsl_display_type_rgba_color_custom_new('full-green', red=.0, green=1.0, blue=0.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('full-yellow', red=1.0, green=1.0, blue=0.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('full-red', red=1.0, green=0.0, blue=0.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create all Display Types and Actions for our object count indicators
            

        ind_left=10
        ind_top=60
        ind_width=33
        ind_height=ind_width
        
        # Three new RGBA Rectangles, one for each of our Minumum/Maximum/Range ODE Trigger occurrences
        retval = dsl_display_type_rgba_rectangle_new('green-rectangle', 
            left=ind_left, top=ind_top, width=ind_width, height=ind_height, border_width=0, 
            color='full-green', has_bg_color=True, bg_color='full-green')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_rectangle_new('yellow-rectangle', 
            left=ind_left, top=ind_top, width=ind_width, height=ind_height, border_width=0, 
            color='full-yellow', has_bg_color=True, bg_color='full-yellow')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_rectangle_new('red-rectangle', 
            left=ind_left, top=ind_top, width=ind_width, height=ind_height, border_width=0, 
            color='full-red', has_bg_color=True, bg_color='full-red')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Three new Display Meta Actions, one for each of our Minumum/Maximum/Range ODE Trigger occurrences
        retval = dsl_ode_action_display_meta_add_new('add-green-rectangle', display_type='green-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_action_display_meta_add_new('add-yellow-rectangle', display_type='yellow-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_action_display_meta_add_new('add-red-rectangle', display_type='red-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, new colors, fonts, rectangles and display action for displaying our object counts on the screen 

        retval = dsl_display_type_rgba_color_custom_new('full-white', red=1.0, green=1.0, blue=1.0, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('full-black', red=0.0, green=0.0, blue=0.0, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_font_new('arial-16-white', font='arial', size=16, color='full-white')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new  Action used to display the summation of all Objects.
        # Format the display string using the occurrences token.
        retval = dsl_ode_action_display_new('display-action', 
            format_string = "Object count = %" + str(DSL_METRIC_OBJECT_OCCURRENCES),  
            offset_x = 45,
            offset_y = 60, 
            font = 'arial-16-white', 
            has_bg_color = True, 
            bg_color = 'full-black')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a Format Label Action to remove the Object Label from view
        # Note: the label can be disabled with the OSD API as well. 
        retval = dsl_ode_action_label_format_new('remove-label', 
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Create three new Count triggers while adding their corresponding Fill colors

        # First Count trigger with range of 0 to 5, and with no limit on the number of occurrences
        retval = dsl_ode_trigger_count_new('r1-object-count', source=DSL_ODE_ANY_SOURCE,
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE, 
            minimum=0, maximum=T1_OBJECT_COUNT)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('r1-object-count', action='add-green-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Secound Count trigger with range of 6 to 8, and with no limit on the number of occurrences
        retval = dsl_ode_trigger_count_new('r2-object-count', source=DSL_ODE_ANY_SOURCE,
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE, 
            minimum=T1_OBJECT_COUNT+1, maximum=T2_OBJECT_COUNT)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('r2-object-count', action='add-yellow-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Third Count trigger with range of 9 to 0 (no-max), and with no limit on the number of occurrences
        retval = dsl_ode_trigger_count_new('r3-object-count', source=DSL_ODE_ANY_SOURCE,
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE, 
            minimum=T2_OBJECT_COUNT, maximum=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('r3-object-count', action='add-red-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create the Summation and Occurrence Triggers to display the Object Count and Hide each Object's Display Text
        
        # New ODE Trigger for Object summation - i.e. new ODE occurrence on detection summation for each frame.
        retval = dsl_ode_trigger_summation_new('objects', source=DSL_ODE_ANY_SOURCE, 
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('objects', action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE occurrence Trigger to hide the Display Text for all detected objects
        retval = dsl_ode_trigger_occurrence_new('every-object', source=DSL_ODE_ANY_SOURCE, 
            class_id=DSL_ODE_ANY_CLASS, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('every-object', action='remove-label')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers=[
            'r1-object-count',
            'r2-object-count',
            'r3-object-count',
            'objects',
            'every-object',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # New URI File Source using the filespec defined above
        retval = dsl_source_uri_new('uri-source', uri_h265, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New KTL Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new('iou-tracker', tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiled Display, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
 
         # Add our ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_tiler_pph_add('tiler', handler='ode-handler', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker', 'tiler', 'on-screen-display', 'window-sink', None])
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
