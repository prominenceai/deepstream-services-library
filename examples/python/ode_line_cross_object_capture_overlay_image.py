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
# This example demonstrates the use of an ODE Cross Trigger with an ODE Line Area 
# and ODE Accumulator to accumulate occurrences of an object (person) crossing 
# the line. The Accumulator uses an ODE Display Action to add the current counts 
# of the IN and OUT crossings as display-metadata to each frame.
#
# The bounding box and historical trace of each object - tracked by the Cross Trigger
# - is assing a new random RGBA color and added as display-metadata to each frame.
#
# An ODE Capture Object Action with an Image Render Player is added to the Cross
# Trigger to capture and render an image of each object (person) that crosses the 
# line. Each image is display for 3 seconds. All files are written to the current
# directory (configurable).


#!/usr/bin/env python

import sys
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

WINDOW_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH
WINDOW_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT

# Minimum Inference confidence level to Trigger ODE Occurrence
# Used for all ODE Triggers
PERSON_MIN_CONFIDENCE = 0.4 # 40%

# Minimum and maximum bounding box height Trigger criteria.
# We only care to track objects that are near the line
# Used for all ODE Triggers
PERSON_MAX_BBOX_HEIGHT = 360
PERSON_MIN_BBOX_HEIGHT = 140

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

##
# Function to be called on Player termination event
##
def player_termination_event_listener(client_data):
    print(' ***  Display Image Complete  *** ')

##
# Function to be called on Object Capture (and file-save) complete
##
def capture_complete_listener(capture_info_ptr, client_data):
    print(' ***  Object Capture Complete  *** ')
   
    capture_info = capture_info_ptr.contents
    print('capture_id: ', capture_info.capture_id)
    print('filename:   ', capture_info.filename)
    print('dirpath:    ', capture_info.dirpath)
    print('width:      ', capture_info.width)
    print('height:     ', capture_info.height)
   

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
   
        #-----------------------------------------------------------------------------
        # First, create the RGBA Display Types that will be used to display the 
        # current metrics; currently tracking, total tracked, and IN and OUT crossings.
        
        retval = dsl_display_type_rgba_color_predefined_new('light-yellow',
            color_id = DSL_COLOR_PREDEFINED_LIGHT_YELLOW, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('heavy-opaque-black',
            red=0.0, green=0.0, blue=0.0, alpha = 0.3)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_font_new('arial-16-yellow',
            font='verdana bold', size=16, color='light-yellow')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a rectangle to be used as a black background for the metrics.
        retval = dsl_display_type_rgba_rectangle_new('black-background',
            left = 1190,
            top = 30,
            width = 300,
            height = 120,
            border_width = 1,
            color = 'heavy-opaque-black',
            has_bg_color = True,
            bg_color = 'heavy-opaque-black')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New action to add the black-background as display-meta.
        retval = dsl_ode_action_display_meta_add_new('add-background',
            display_type = 'black-background')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New always trigger to call on the Display Action to
        # add the display-meta to each frame.
        retval = dsl_ode_trigger_always_new('every-frame-trigger',
            DSL_ODE_ANY_SOURCE, DSL_ODE_PRE_OCCURRENCE_CHECK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Display Action to the Always Trigger.
        retval = dsl_ode_trigger_action_add('every-frame-trigger',
            'add-background')
       
        #-----------------------------------------------------------------------------
        # Next, create a new Occurrence Trigger with Actions to remove/hide
        # all object bounding boxes and labels. 

        # Create a new Action to remove the bounding box by default.
        # The bounding box will be reformatted by the ODE Cross Trigger
        retval = dsl_ode_action_bbox_format_new('remove-bbox',
            border_width = 0,
            border_color = None,
            has_bg_color = False,
            bg_color = None)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Create a new Action to remove the Object labels by default.
        # The bounding box will be reformatted by the ODE Cross Trigger
        retval = dsl_ode_action_label_format_new('remove-label',
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Any-Class Occurrence Trigger for our remove label and border actions
        retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = DSL_ODE_ANY_CLASS,
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Add both Format Actions to the every-occurrence Trigger
        retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger',
            actions = ['remove-bbox', 'remove-label', None])
        if retval != DSL_RETURN_SUCCESS:
            break
           
        #-----------------------------------------------------------------------------
        # Next, create new Display Types, ODE Line Area, ODE Print and Capture Object
        # actions, and ODE Cross Trigger.

        # Create a custom RED RGBA color for the RGBA Line and Line Area
        retval = dsl_display_type_rgba_color_custom_new('opaque-red',
            red=1.0, green=0.2, blue=0.2, alpha=0.6)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Create the RGBA Line Display Type with a width of 6 pixels for hysteresis
        retval = dsl_display_type_rgba_line_new('line',
            x1=260, y1=680, x2=600, y2=660, width=6, color='opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Create the ODE line area to use as criteria for ODE occurrence.
        # Use the center point on the bounding box's bottom edge for testing
        retval = dsl_ode_area_line_new('line-area', line='line',
            show=True, bbox_test_point=DSL_BBOX_POINT_SOUTH)    
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Cross Trigger filtering on PERSON class_id to track and trigger on
        # objects that fully cross the line. The person must be tracked for a minimum
        # of 5 frames prior to crossing the line to trigger an ODE occurrence.
        # The trigger can save/use up to a maximum of 200 frames of history to create
        # the object's historical trace to test for line-crossing.
        retval = dsl_ode_trigger_cross_new('person-crossing-line',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON,
            limit = DSL_ODE_TRIGGER_LIMIT_NONE,
            min_frame_count = 5,
            max_trace_points = 200,
            test_method = DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set a minimum confidence level to avoid false negatives.
        retval = dsl_ode_trigger_confidence_min_set('person-crossing-line',
            min_confidence = PERSON_MIN_CONFIDENCE)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Set minimum bounding-box dimensions... we don't care about
        # objects off in the distance, away from the line. 
        retval = dsl_ode_trigger_dimensions_min_set('person-crossing-line',
            min_width = 0, min_height = PERSON_MIN_BBOX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set maximum bounding-box dimensions... we don't care about
        # objects close to the camera, away from the line. 
        retval = dsl_ode_trigger_dimensions_max_set('person-crossing-line',
            max_width = 0, max_height = PERSON_MAX_BBOX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New RGBA Random Color to use for Object Trace and BBox    
        retval = dsl_display_type_rgba_color_random_new('random-color',
            hue = DSL_COLOR_HUE_RANDOM,
            luminosity = DSL_COLOR_LUMINOSITY_RANDOM,
            alpha = 1.0,
            seed = 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the Cross Trigger's view settings to enable display of the Object Trace
        retval = dsl_ode_trigger_cross_view_settings_set('person-crossing-line',
            enabled=True, color='random-color', line_width=4)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Add the line area to the New Cross Trigger
        retval = dsl_ode_trigger_area_add('person-crossing-line', area='line-area')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Scale BBox Action to increase the capture area for the capture
        # action defined below. We scale up by a factor of percentage
        retval = dsl_ode_action_bbox_scale_new('scale-bbox-action', scale=150)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create a new Capture Action to capture the object to jpeg image, and save to file.
        retval = dsl_ode_action_capture_object_new('person-capture-action', outdir="./")
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_action_print_new('print-action', force_flush=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the capture complete listener function to the action
        retval = dsl_ode_action_capture_complete_listener_add('person-capture-action',
            capture_complete_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Make sure to add the scale-bbox action first.
        retval = dsl_ode_trigger_action_add_many('person-crossing-line', actions=[
            'scale-bbox-action', 'person-capture-action', 'print-action', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create the Image Render Player with a NULL file_path to by updated by 
        # the Capture Action
        dsl_player_render_image_new(
            name = 'image-player',
            file_path = None,
            render_type = DSL_RENDER_TYPE_OVERLAY,
            offset_x = 700,
            offset_y = 300,
            zoom = 200,
            timeout = 3) # show indefinetely, until new image is captured

        # Add the Termination listener callback to the Player
        retval = dsl_player_termination_event_listener_add('image-player',
            client_listener=player_termination_event_listener, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            return

        # Add the Player to the Object Capture Action. The Action will add/queue
        # the file_path to each image file created during capture.
        retval = dsl_ode_action_capture_image_player_add('person-capture-action',
            player='image-player')


        #`````````````````````````````````````````````````````````````````````````````
        # Next, create an ODE Accumulator and ODE Display Action for the Line Cross 
        # metrics. Action is added to the Accumulator which is added to the Trigger.
        
        # Create a new Display Action used to display the Accumulated ODE Occurrences.
        # Format the display string using the occurrences in and out tokens.
        retval = dsl_ode_action_display_new('display-cross-metrics-action',
            format_string =
                "In : %" + str(DSL_METRIC_OBJECT_OCCURRENCES_DIRECTION_IN) +
                ", Out : %" + str(DSL_METRIC_OBJECT_OCCURRENCES_DIRECTION_OUT),  
            offset_x = 1200,
            offset_y = 100,
            font = 'arial-16-yellow',
            has_bg_color = False,
            bg_color = None)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Create an ODE Accumulator to add to the Cross Trigger. The Accumulator
        # will work with the Trigger to accumulate the IN and OUT occurrence metrics.
        retval = dsl_ode_accumulator_new('cross-accumulator')
        if retval != DSL_RETURN_SUCCESS:
            break
       
        # Add the Display Action to the Accumulator. The Accumulator will call on
        # the Display Action to display the new accumulative metrics after each frame.
        retval = dsl_ode_accumulator_action_add('cross-accumulator',
            'display-cross-metrics-action')
        if retval != DSL_RETURN_SUCCESS:
            break
       
        # Add the Accumulator to the Line Cross Trigger.
        retval = dsl_ode_trigger_accumulator_add('person-crossing-line',
            'cross-accumulator')

        #-----------------------------------------------------------------------------
        # Next, create an ODE Summation Trigger, using the exact same criteria as
        # the ODE Cross Trigger, to trigger on the summation of objects for each 
        # frame. This will tell us how many objects are currently being tracked.
        retval = dsl_ode_trigger_summation_new('objects-sumation-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON,
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Use same criteria as Cross Trigger
        retval = dsl_ode_trigger_confidence_min_set('objects-sumation-trigger',
            min_confidence = PERSON_MIN_CONFIDENCE)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Use same criteria as Cross Trigger
        retval = dsl_ode_trigger_dimensions_min_set('objects-sumation-trigger',
            min_width = 0, min_height = PERSON_MIN_BBOX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Use same criteria as Cross Trigger
        retval = dsl_ode_trigger_dimensions_max_set('objects-sumation-trigger',
            max_width = 0, max_height = PERSON_MAX_BBOX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Display Action used to display the number of Objects
        # detected/tracked. Format the display string using the occurrences.
        retval = dsl_ode_action_display_new('display-summation-action',
            format_string =
                "Currently Tracking  : %" + str(DSL_METRIC_OBJECT_OCCURRENCES) ,
            offset_x = 1200,
            offset_y = 40,
            font = 'arial-16-yellow',
            has_bg_color = False,
            bg_color = None)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Add the Display Action to the Summation Trigger
        retval = dsl_ode_trigger_action_add('objects-sumation-trigger',
            'display-summation-action')
        if retval != DSL_RETURN_SUCCESS:
            break
           
        #-----------------------------------------------------------------------------

        # New ODE Trigger for New Person Instance - i.e. one new ODE occurrence
        # for every person object with a new Tracking Id.
        retval = dsl_ode_trigger_instance_new('new-instance-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON,
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Use same criteria as Cross and Summation Triggers
        retval = dsl_ode_trigger_confidence_min_set('new-instance-trigger',
            min_confidence = PERSON_MIN_CONFIDENCE)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Use same criteria as Cross and Summation Triggers
        retval = dsl_ode_trigger_dimensions_min_set('new-instance-trigger',
            min_width = 0, min_height = PERSON_MIN_BBOX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Use same criteria as Cross and Summation Triggers
        retval = dsl_ode_trigger_dimensions_max_set('new-instance-trigger',
            max_width = 0, max_height = PERSON_MAX_BBOX_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an ODE Accumulator to add to the Instance Trigger. The Accumulator
        # will work with the Trigger to accumulate the new Instance metric.
        retval = dsl_ode_accumulator_new('instance-accumulator')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Display Action used to display the number of Objects
        # detected/tracked. Format the display string using the occurrences.
        retval = dsl_ode_action_display_new('display-total-action',
            format_string =
                "Total Tracked  : %" + str(DSL_METRIC_OBJECT_OCCURRENCES) ,
            offset_x = 1200,
            offset_y = 70,
            font = 'arial-16-yellow',
            has_bg_color = False,
            bg_color = None)
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Add the Display Action to the Accumulator 
        retval = dsl_ode_accumulator_action_add('instance-accumulator',
            'display-total-action')
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Add the Accumulator to the New Instance Trigger.
        retval = dsl_ode_trigger_accumulator_add('new-instance-trigger',
            'instance-accumulator')
           
        #-----------------------------------------------------------------------------

        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Add the two ODE Trigger to the ODE Handler - order is important    
        retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers=[
            'every-frame-trigger', 'every-occurrence-trigger', 'person-crossing-line',
            'objects-sumation-trigger', 'new-instance-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break
       
       
        ############################################################################################
        #
        # Create the remaining Pipeline components
       
        # New File Source using the file path defined at the top of the file
        retval = dsl_source_file_new('file-source',
            file_path = uri_h265,
            repeat_enabled = True)
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

        # New OSD with text, clock and bbox display all enabled.
        retval = dsl_osd_new('on-screen-display',
            text_enabled=True, clock_enabled=False, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

         # Add our ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_osd_pph_add('on-screen-display', handler='ode-handler', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline',
            ['file-source', 'primary-gie', 'iou-tracker', 'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break
           
        # Set the XWindow into full-screen mode for a kiosk look
        retval = dsl_pipeline_xwindow_fullscreen_enabled_set('pipeline', True)
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
