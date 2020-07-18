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

        # Create new RGBA color types
        retval = dsl_display_type_rgba_color_new('opaque-red', red=1.0, green=0.0, blue=0.0, alpha = 0.2)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_new('full-black', red=0.0, green=0.0, blue=0.0, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_new('opaque-black', red=0.0, green=0.0, blue=0.0, alpha = 0.2)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_new('full-white', red=1.0, green=1.0, blue=1.0, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_new('opaque-white', red=1.0, green=1.0, blue=1.0, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_new('full-blue', red=0.0, green=0.0, blue=1.0, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create new RGBA font type2
        retval = dsl_display_type_rgba_font_new('arial-14-white', font='arial', size=14, color='full-white')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_font_new('arial-20-blue', font='arial', size=20, color='full-blue')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create two areas to be used as criteria for ODE Occurrence. The first area
        # will be for the Person class alone...  and defines a vertical rectangle to 
        # the left of the pedestrian sidewalk. The pixel values are relative to the
        # Stream-Muxer output dimensions (default 1920 x 1080), vs. the Tiler/Sink
        retval = dsl_ode_area_new('person-area', left=200, top=0, width=10, height=1089, display=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # The second area will be shared by both Person and Vehicle classes... and defines
        # a vertical rectangle to the right of the sidewalk and left of the street
        # This area's background will be shaded yellow for caution
        retval = dsl_ode_area_new('shared-area', left=500, top=0, width=60, height=1089, display=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_area_color_set('shared-area', red=1.0, green=1.0, blue=0.0, alpha = 0.1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Fill Action that will fill the Object's rectangle with a shade of red to indicate that
        # overlap with one or more of the defined Area's has occurred, i.e. ODE occurrence. The action will be
        # used with both the Person and Car class Ids to indicate thay have entered the area of caution
        retval = dsl_ode_action_fill_object_new('red-fill-action', 'opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Capture Action to capture the full-frame to jpeg image, and save to file. 
        # The action will be triggered on firt occurrence of a bicycle and will be save to the current dir.
        retval = dsl_ode_action_capture_frame_new('bicycle-capture', outdir="./")
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new  Action used to display all Object detection summations for each frame. Use the classId
        # to add an additional vertical offset so the one action can be shared accross classId's
        retval = dsl_ode_action_display_new('display-action', offsetX=15, offsetY=50, offsetY_with_classId=True,
            font='arial-14-white', has_bg_color=True, bg_color='full-black')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New RGBA Rectangle to use as a background for summation display. Using the full black with alpha=1.0
        retval = dsl_display_type_rgba_rectangle_new('black-rectangle', left=10, top=45, width=190, height=110, 
            border_width=0, color='full-black', has_bg_color=True, bg_color='full-black')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # A second RGBA Rectangle to use as a dropped shadow for summation display. Using the opaque black
        retval = dsl_display_type_rgba_rectangle_new('black-shadow', left=16, top=51, width=190, height=110, 
            border_width=0, color='opaque-black', has_bg_color=True, bg_color='opaque-black')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # A new RGBA Circle to be used as a custom arrow head.
        retval = dsl_display_type_rgba_circle_new('blue-circle', x_center=530, y_center=50, radius=5, 
            color='full-blue', has_bg_color=True, bg_color='full-blue')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_line_new('blue-line', x1=530, y1=50, x2=730, y2=50, width=2, color='full-blue')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_display_type_rgba_text_new('blue-text', 'Shared Trigger Area', x_offset=733, y_offset=30, 
            font='arial-20-blue', has_bg_color=False, bg_color='full-blue')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Action to display the black rectangle as a background for the summantion display
        retval = dsl_ode_action_overlay_frame_new('overlay-black-rectangle', 'black-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break
        # Create a new Action to display the opaque black rectangle as a dropped shadow for the summantion display
        retval = dsl_ode_action_overlay_frame_new('overlay-black-shadow', 'black-shadow')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Action to display the blue circle arrow head
        retval = dsl_ode_action_overlay_frame_new('overlay-blue-circle', 'blue-circle')
        if retval != DSL_RETURN_SUCCESS:
            break
        # Create a new Action to display the blue annotation line
        retval = dsl_ode_action_overlay_frame_new('overlay-blue-line', 'blue-line')
        if retval != DSL_RETURN_SUCCESS:
            break
        # Create a new Action to display the blue text to annotate the shared area
        retval = dsl_ode_action_overlay_frame_new('overlay-blue-text', 'blue-text')
        if retval != DSL_RETURN_SUCCESS:
            break


        retval = dsl_ode_trigger_always_new('always-trigger', when=DSL_ODE_PRE_OCCURRENCE_CHECK)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('always-trigger', actions=[
            'overlay-black-shadow', 
            'overlay-black-rectangle', 
            'overlay-blue-circle',
            'overlay-blue-line',
            'overlay-blue-text',
            None])
        retval = dsl_ode_trigger_always_new('always-trigger', when=DSL_ODE_PRE_OCCURRENCE_CHECK)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('always-trigger', action='overlay-black-rectangle')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Occurrence Trigger, filtering on the Person Class Id, with no limit on the number of occurrences
        # Add the two Areas as Occurrence (overlap) criteria and the action to Fill the background red on occurrence
        retval = dsl_ode_trigger_occurrence_new('person-area-overlap', class_id=PGIE_CLASS_ID_PERSON, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_area_add_many('person-area-overlap', areas=['person-area', 'shared-area', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('person-area-overlap', action='red-fill-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Occurrence Trigger, filtering on the Vehicle ClassId, with no limit on the number of occurrences
        # Add the single Shared Area and the action to Fill the background red on occurrence 
        retval = dsl_ode_trigger_occurrence_new('vehicle-area-overlap', class_id=PGIE_CLASS_ID_VEHICLE, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_area_add('vehicle-area-overlap', area='shared-area')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('vehicle-area-overlap', action='red-fill-action')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Occurrence Trigger, filtering on the Bicycle ClassId, with a limit of one occurrence
        # Add the capture-frame action to the first occurrence event
        retval = dsl_ode_trigger_occurrence_new('bicycle-first-occurrence', class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('bicycle-first-occurrence', action='bicycle-capture')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New ODE Triggers for Object summation - i.e. new ODE occurrence on detection summation.
        # Each Trigger will share the same ODE Display Action
        retval = dsl_ode_trigger_summation_new('Vehicles:', class_id=PGIE_CLASS_ID_VEHICLE, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('Vehicles:', action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_summation_new('Bicycles:', class_id=PGIE_CLASS_ID_BICYCLE, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('Bicycles:', action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_summation_new('Pedestrians:', class_id=PGIE_CLASS_ID_PERSON, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('Pedestrians:', action='display-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # A hide action to use with two occurrence Triggers, filtering on the Person Class Id and Vehicle Class Id
        # We will use an every occurrece Trigger to hide the Display Text and Rectangle Border for each object detected
        # We will leave the Bicycle Display Text and Border untouched
        retval = dsl_ode_action_hide_new('hide-action', text=True, border=True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_occurrence_new('person-every-occurrence', class_id=PGIE_CLASS_ID_PERSON, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('person-every-occurrence', action='hide-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_occurrence_new('vehicle-every-occurrence', class_id=PGIE_CLASS_ID_VEHICLE, limit=0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('vehicle-every-occurrence', action='hide-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_ode_handler_new('ode-hanlder')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add_many('ode-hanlder', triggers=[
            'vehicle-area-overlap',
            'person-area-overlap', 
            'bicycle-first-occurrence',
            'always-trigger',
            'Vehicles:',
            'Bicycles:',
            'Pedestrians:',
            'person-every-occurrence',
            'vehicle-every-occurrence',
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
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
 
        # New OSD with clock enabled... .
        retval = dsl_osd_new('on-screen-display', True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
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
