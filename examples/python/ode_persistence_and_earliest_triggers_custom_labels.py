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
sys.path.insert(0, "../../")
import time

from dsl import *

#-------------------------------------------------------------------------------------------
#
# This script demonstrates the use of a Persistence Trigger to trigger on each Vehicle
# that is tracked for more than one frame, to calculate the time of Object persistence
# from the first frame the object was detected. 
#
# The Tracked Object's label is then "customed" to show the tracking Id and time of 
# persistence for each tracked Vehicle.
#
# The script also creats an Earliest Trigger to trigger on the Vehicle that appeared 
# the earliest -- i.e. the object with greatest persistence value -- and displays that
# Object's persistence using a ODE Display Action.
# 

# File path for the single File Source
file_path = '/opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_qHD.mp4'

# Filespecs for the Primary Triton Inference Server (PTIS)
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt'

# Window Sink Dimensions
sink_width = 1280
sink_height = 720

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
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_main_loop_quit()
 
## 
# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_main_loop_quit()

# Function to be called on End-of-Stream (EOS) event
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

        #------------------------------------------------------------------------
        # Step 1:  Create Actions and a Trigger to remove/hide the default
        #          label contents and bounding-box from all Objects
        
        # Create a Format Label Action to remove the label.
        # This action will be added to an Occurrence Trigger to clear
        # remove all labels of all content for every Object. 
        retval = dsl_ode_action_format_label_new('remove-label', 
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Format Bounding Box Action to remove the box border from view
        retval = dsl_ode_action_format_bbox_new('remove-border', border_width=0,
            border_color=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Occurrence Trigger to trigger on every Object
        retval = dsl_ode_trigger_occurrence_new('occurrence-trigger', 
            source = 'uri-source',
            class_id = DSL_ODE_ANY_CLASS, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the above Action to the Occurrence trigger to clear/remove both the 
        # label contents and bounding-box border for all Objects.
        retval = dsl_ode_trigger_action_add_many('occurrence-trigger', 
            actions=['remove-label', 'remove-border', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #------------------------------------------------------------------------
        # Step 2:  Create new custom RGBA Colors and Fonts for our custom
        #          labels and bounding-boxes

        # Create the forground and background colors for our custom, formated Object Label 
        retval = dsl_display_type_rgba_color_new('full-white', 
            red=1.0, blue=1.0, green=1.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_new('opaque-black', 
            red=0.0, blue=0.0, green=0.0, alpha=0.8)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_font_new('verdana-bold-16-white', 
            font='verdana bold', size=16, color='full-white')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_font_new('verdana-bold-20-white', 
            font='verdana bold', size=20, color='full-white')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a border color for the Custom BBox format
        retval = dsl_display_type_rgba_color_new('full-green', 
            red=0.0, blue=0.0, green=1.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break

        #------------------------------------------------------------------------
        # Step 3:  Create Actions and a Trigger to customize the Object 
        #          label contents and bounding-box for all Vehicles tracked.

        # Create a Customize Label Action to add the Tracking Id and Persistence 
        # time value to each object that is tracked across two consecutive frames
        retval = dsl_ode_action_customize_label_new('customize-label', 
            content_types = [DSL_OBJECT_LABEL_TRACKING_ID,
                DSL_OBJECT_LABEL_PERSISTENCE], 
            size = 2,
            mode = DSL_WRITE_MODE_TRUNCATE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Format Label Action to
        retval = dsl_ode_action_format_label_new('format-label', 
            font = 'verdana-bold-16-white', 
            has_bg_color = True, 
            bg_color = 'opaque-black')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create Format Bounding Box Action to custom the box border 
        # for all Vehicles tracked - to be added with the custom label
        retval = dsl_ode_action_format_bbox_new('format-bbox', border_width=3,
            border_color='full-green', has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Persistence Trigger with 0 minimum and maximum criteria
        # which will trigger on every Vehicle that is tracked
        retval = dsl_ode_trigger_persistence_new('persitence-trigger', 
            source = 'uri-source',
            class_id = PGIE_CLASS_ID_VEHICLE, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE, 
            minimum = 0, 
            maximum = 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_trigger_action_add_many('persitence-trigger', 
            actions=['customize-label', 'format-label', 'format-bbox', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #------------------------------------------------------------------------
        # Step 4:  Create a Display Action and an Earliest Trigger to display the time
        #          of persistence of the object that was the earliest to enter view
        retval = dsl_ode_action_display_new('display-action',
            x_offset = 1000,
            y_offset = 100,
            y_offset_with_classId = False,
            font = 'verdana-bold-20-white',
            has_bg_color = True,
            bg_color = 'opaque-black')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retval = dsl_ode_trigger_earliest_new('Following for', 
            source = 'uri-source',
            class_id = PGIE_CLASS_ID_VEHICLE, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_ode_trigger_action_add('Following for',
            action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        #------------------------------------------------------------------------
        # Step 5:  Create an Object Detection Event (ODE) Handler to handle
        #          Occurence, Persistence, and Earliest Triggers defined above.
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # NOTE: Order of addition is important - specifies execution order
        retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers = 
            ['occurrence-trigger', 'persitence-trigger', 'Following for', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        #------------------------------------------------------------------------
        # Step 6:  Create the remaining Pipeline components

        # New File Source using the file path specified above, repeat enabled.
        retval = dsl_source_file_new('uri-source', file_path, True)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Primary TIS using the filespec specified above, with interval = 0
        retval = dsl_infer_tis_primary_new('primary-tis', primary_infer_config_file, 1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New KTL Tracker, setting output width and height of tracked objects
        retval = dsl_tracker_ktl_new('ktl-tracker', 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with clock and text enabled... using default values.
        retval = dsl_osd_new('on-screen-display', True, True)
        if retval != DSL_RETURN_SUCCESS:
            break
            
         # Add our ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_osd_pph_add('on-screen-display', handler='ode-handler', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new('window-sink', 0, 0, sink_width, sink_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to a new pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-tis', 'ktl-tracker', 'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline", xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_delete_event_handler_add("pipeline", xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the listener callback functions defined above
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
