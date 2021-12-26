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

file_path = "../../test/streams/sample_1080p_h264.mp4"

# Filespecs for the Primary GIE and IOU Trcaker
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
tracker_config_file = '../../test/configs/iou_config.txt'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MIN_OBJECTS = 3
MAX_OBJECTS = 8

TILER_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH
TILER_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT
WINDOW_WIDTH = TILER_WIDTH
WINDOW_HEIGHT = TILER_HEIGHT

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
    
        # This example demonstrates the use of a Polygon Area for Inclusion 
        # or Exlucion critera for ODE occurrence. Change the variable below to try each.
        
        #```````````````````````````````````````````````````````````````````````````````````

        # Create a Format Label Action to remove the Object Label from view
        # Note: the label can be disabled with the OSD API as well. 
        retval = dsl_ode_action_format_label_new('remove-label', 
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a Format Bounding Box Action to remove the box border from view
        retval = dsl_ode_action_format_bbox_new('remove-border', border_width=0,
            border_color=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Any-Class Occurrence Trigger for our Hide Action
        retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger', source=DSL_ODE_ANY_SOURCE,
            class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger', 
            actions=['remove-label', 'remove-border', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_display_type_rgba_color_new('opaque-red', red=1.0, green=0.0, blue=0.0, alpha=0.3)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new  Action used to fill a bounding box with the opaque red color
        retval = dsl_ode_action_format_bbox_new('fill-action',
            border_width = 0,
            border_color = None,
            has_bg_color = True,
            bg_color = 'opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break

        # create a list of X,Y coordinates defining the points of the Polygon.
        # Polygon can have a minimum of 3, maximum of 8 points (sides)
        coordinates = [dsl_coordinate(365,600), dsl_coordinate(580,620), 
            dsl_coordinate(600, 770), dsl_coordinate(180,750)]
            
        # Create the Polygon display type 
        retval = dsl_display_type_rgba_polygon_new('polygon1', 
            coordinates=coordinates, num_coordinates=len(coordinates), border_width=4, color='opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # create the ODE inclusion area to use as criteria for ODE occurrence
        retval = dsl_ode_area_inclusion_new('polygon-area', polygon='polygon1', 
            show=True, bbox_test_point=DSL_BBOX_POINT_SOUTH)    
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Occurrence Trigger, filtering on PERSON class_id, 
        # and with no limit on the number of occurrences
        retval = dsl_ode_trigger_occurrence_new('person-in-area-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_ode_trigger_area_add('person-in-area-trigger', area='polygon-area')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retval = dsl_ode_trigger_action_add('person-in-area-trigger', action='fill-action')
        if retval != DSL_RETURN_SUCCESS:
            break


        # New Occurrence Trigger, filtering on PERSON class_id, for our capture object action
        # with a limit of one which will be reset in the capture-complete callback
        retval = dsl_ode_trigger_instance_new('person-enter-area-trigger', 
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Processing at 30 frames per second can create a lot of images quickly. This is
        # true even when triggering on new Instances, as the object Tracker has LIMITED
        # accuracy with such a low camera angle, i.e. it has trouble when objects cross
        # in front of one another. We need to throttle the processing interval to 
        # allow each captured image time to load and play in the Image Player
        # Change from the default of 0 (every frame) to every 15 frames (2 Hz).
        retval = dsl_ode_trigger_interval_set('person-enter-area-trigger', 15)

        # Using the same Inclusion area as the New Occurrence Trigger
        retval = dsl_ode_trigger_area_add('person-enter-area-trigger', area='polygon-area')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Capture Action to capture the object to jpeg image, and save to file. 
        retval = dsl_ode_action_capture_object_new('person-capture-action', 
            DSL_NVBUF_MEM_TYPE_DEFAULT, outdir="./")
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Add the capture complete listener function to the action
        retval = dsl_ode_action_capture_complete_listener_add('person-capture-action', 
            capture_complete_listener, None)

        retval = dsl_ode_trigger_action_add('person-enter-area-trigger', 
            action='person-capture-action')
        if retval != DSL_RETURN_SUCCESS:
            break


        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````
        
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers=[
            'every-occurrence-trigger', 
            'person-in-area-trigger', 
            'person-enter-area-trigger',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        #```````````````````````````````````````````````````````````````````````````````````````````````````````````````

        # Create the Image Render Player with a NULL file_path to by updated by the Capture Action
        dsl_player_render_image_new(
            name = 'image-player',
            file_path = None,
            render_type = DSL_RENDER_TYPE_OVERLAY,
            offset_x = 400, 
            offset_y = 100, 
            zoom = 150,
            timeout = 0) # show indefinetely, until new image is captured

        # Add the Termination listener callback to the Player 
        retval = dsl_player_termination_event_listener_add('image-player',
            client_listener=player_termination_event_listener, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            return

        # Add the Player to the Object Capture Action. The Action will add/queue
        # the file_path to each image file created during capture. 
        retval = dsl_ode_action_capture_image_player_add('person-capture-action', 
            player='image-player')

        
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # New File Source using the file path defined at the top of the file
        retval = dsl_source_file_new('file-source', 
            file_path = file_path, 
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
            ['file-source', 'primary-gie', 'iou-tracker', 'tiler', 'on-screen-display', 'window-sink', None])
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
