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

TILER_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH
TILER_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


# Min object height in pixels
MINIMUM_OBJ_HEIGHT = 80

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
    
        # This example demonstrates the use of an ODE Distance Trigger to trigger on
        # occurrence of two objects of different class id that are closer that a 
        # minimum distance - specifically testing the distance between People and Vehicles.
        # The bounding boxes for the two objects that are witin the minimim distance will. 
        # be filled with a color for visual indication of the events.
        
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
            
        # Create an Any-Class Occurrence Trigger for Format Actions
        retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger', source='uri-source-1',
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger', 
            actions=['remove-label', 'remove-border', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Create a new RGBA fill color to fill the bounding boxes of objects witin distance
        retval = dsl_display_type_rgba_color_custom_new('opaque-red', red=1.0, green=0.0, blue=0.0, alpha=0.5)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new RGBA fill color to fill the bounding boxes of objects of minimum height
        retval = dsl_display_type_rgba_color_custom_new('opaque-white', red=1.0, green=1.0, blue=1.0, alpha=0.3)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        #```````````````````````````````````````````````````````````````````````````````````
        # Create the Action to fill the bounding boxes of the two objects within minimim distance
        retval = dsl_ode_action_bbox_format_new('fill-red-action',
            border_width = 0,
            border_color = None,
            has_bg_color = True,
            bg_color = 'opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create the Action to fill the bounding boxes of all objects with minimim height
        retval = dsl_ode_action_bbox_format_new('fill-white-action',
            border_width = 0,
            border_color = None,
            has_bg_color = True,
            bg_color = 'opaque-white')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Create the new Distance trigger with minimim distance critera as a percentage
        # of the width of Class A in the A/B distance measurement. ODE Occurrence will be 
        # triggered if the distance between any Person and Vehicle is measured to be less 
        # that the 300% of the width of the Person's BBox. Maximum is set to 0 == no maximum
        # Note: Class A and Class B can be set to the same Class Id or DSL_ODE_ANY_CLASS.
        # test_point is DSL_BBOX_POINT_SOUTH == measuring from center points of bottom edges
        # test_method is DSL_DISTANCE_METHOD_PERCENT_WIDTH_A == % of Person's BBox width
        retval = dsl_ode_trigger_distance_new('distance-trigger', 
            source = 'uri-source-1',
            class_id_a = PGIE_CLASS_ID_PERSON, 
            class_id_b = PGIE_CLASS_ID_VEHICLE, 
            limit=DSL_ODE_TRIGGER_LIMIT_NONE,
            minimum = 300,
            maximum = 0,
            test_point = DSL_BBOX_POINT_SOUTH,
            test_method = DSL_DISTANCE_METHOD_PERCENT_WIDTH_A)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an additional Occurrence Trigger using the same minimum hight critera
        # just to highlight all objects that are being tested for distance. 
        retval = dsl_ode_trigger_occurrence_new('min-height-trigger',
            source = 'uri-source-1',
            class_id = DSL_ODE_ANY_CLASS,
            limit=DSL_ODE_TRIGGER_LIMIT_NONE)
            
        # Set the minimum Object height critera for both triggers        
        retval = dsl_ode_trigger_dimensions_min_set('distance-trigger', 
            min_width = 0, 
            min_height = MINIMUM_OBJ_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_dimensions_min_set('min-height-trigger', 
            min_width = 0, 
            min_height = MINIMUM_OBJ_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        #```````````````````````````````````````````````````````````````````````````````````
        # Next, we add our Actions to our Triggers
        retval = dsl_ode_trigger_action_add('distance-trigger', action='fill-red-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('min-height-trigger', action='fill-white-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # New ODE Handler to handle the ODE Trigger
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_trigger_add_many('ode-handler', 
            triggers=['every-occurrence-trigger', 'distance-trigger', 'min-height-trigger', None])
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
            ['uri-source-1', 'primary-gie', 'iou-tracker', 'tiler', 
            'on-screen-display', 'window-sink', None])
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
