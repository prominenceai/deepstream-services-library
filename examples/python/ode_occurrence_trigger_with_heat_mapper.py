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

TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
TILER_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT
WINDOW_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
WINDOW_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT

# Minimum Inference confidence level to Trigger ODE Occurrence
PERSON_MIN_CONFIDENCE = 0.4 # 40%

SPECTRAL_PALETTE = 'spectral-palette'
RED_PALETTE = 'red-palette'
GREEN_PALETTE = 'green-palette'
BLUE_PALETTE = 'blue-palette'
GREY_PALETTE = 'grey-palette'

COLOR_PALETTES = [SPECTRAL_PALETTE, RED_PALETTE, GREEN_PALETTE, BLUE_PALETTE, GREY_PALETTE]
color_palette_index  = DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL

# ------------------------------------------------------------------------------------
# This example demonstrates the use of an ODE Heat-Mapper added to an 
# ODE Occurrence trigger that triggers on every Person occurrence.
# The occurrence data is mapped/ovelaid on everyframe. The example creates 
# all 5 predefined RGBA Color Palettes - Spectral, Red, Green, Blue, and Grey.
# The ODE Heat-Mapper is created with the Spectral palette, but can be updated
# at runtime by pressing the 'N' key.
#
# Several keys, bound to the Window Sink, are mapped to the ODE Heat Mapper services  
#    - 'N' key maps to 'next' color palette with - dsl_ode_heat_mapper_color_palette_set
#    - 'C' key maps to 'clear' heat-map metrics  - dsl_ode_heat_mapper_metrics_clear
#    - 'P' key maps to 'print' heat-map metrics  - dsl_ode_heat_mapper_metrics_print
#    - 'L' key maps to 'log' heat-map metrics    - dsl_ode_heat_mapper_metrics_log
#    - 'G' key maps to 'get' heat-map metrics    - dsl_ode_heat_mapper_metrics_get

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    global color_palette_index
    
    print('key released = ', key_string)
    
    # N key maps to next color palette
    if key_string.upper() == 'N':
        color_palette_index = \
            (color_palette_index + 1) % (DSL_COLOR_PREDEFINED_PALETTE_GREY +1)
        dsl_ode_heat_mapper_color_palette_set('person-heat-mapper',
            COLOR_PALETTES[color_palette_index])    

    # C key maps to clear heat-map metrics
    if key_string.upper() == 'C':
        dsl_ode_heat_mapper_metrics_clear('person-heat-mapper')
        
    # P key maps to print heat-map metrics to the console
    if key_string.upper() == 'P':
        dsl_ode_heat_mapper_metrics_print('person-heat-mapper')

    # D key maps to print (dump) heat-map metrics to the console
    if key_string.upper() == 'L':
        dsl_ode_heat_mapper_metrics_log('person-heat-mapper')

    # G key maps to get heat-map metrics
    if key_string.upper() == 'G':
        retval, buffer, size = dsl_ode_heat_mapper_metrics_get('person-heat-mapper')
        
        # access buffer as an array of size i.e. buffer[i] 

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
        
        #-----------------------------------------------------------------------------
        # First, we want to remove the Object labels and bounding boxes. This is
        # be adding Format-Label and Format-BBox actions to an Occurrence Trigger.

        # Create a Format Label Action to remove the Object Label from view
        # Note: the label can be disabled with the OSD API as well, however
        # that will disable all text/labels, not just object labels. 
        retval = dsl_ode_action_label_format_new('remove-label', 
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a Format Bounding Box Action to remove the box border from view
        retval = dsl_ode_action_bbox_format_new('remove-bbox', border_width=0,
            border_color=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Any-Class Occurrence Trigger for our remove Actions
        retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            class_id = DSL_ODE_ANY_CLASS, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger', 
            actions=['remove-label', 'remove-bbox', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Next, create an Occurrence Trigger to filter on People - defined with
        # a minimuim confidence level to eleminate most false positives

        # New Occurrence Trigger, filtering on PERSON class_id,
        retval = dsl_ode_trigger_occurrence_new('person-occurrence-trigger', 
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON,
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set a minimum confidence level to avoid false positives.
        retval = dsl_ode_trigger_infer_confidence_min_set('person-occurrence-trigger',
            min_confidence = PERSON_MIN_CONFIDENCE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Next, we create color palettes for each of the Predefined Palettes. Each
        # palette can be tested with the ODE Heat-Mapper to see what looks best.
        # The 'N' key will be mapped to select the 'Next' color palette - simpling 
        # cycling through each of the indecies in the COLOR_PALETTES array.
        
        retval = dsl_display_type_rgba_color_palette_predefined_new(SPECTRAL_PALETTE,
            palette_id = DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL,
            alpha = 0.6)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retval = dsl_display_type_rgba_color_palette_predefined_new(RED_PALETTE,
            palette_id = DSL_COLOR_PREDEFINED_PALETTE_RED,
            alpha = 0.6)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retval = dsl_display_type_rgba_color_palette_predefined_new(GREEN_PALETTE,
            palette_id = DSL_COLOR_PREDEFINED_PALETTE_GREEN,
            alpha = 0.6)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_palette_predefined_new(BLUE_PALETTE,
            palette_id = DSL_COLOR_PREDEFINED_PALETTE_BLUE,
            alpha = 0.6)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_palette_predefined_new(GREY_PALETTE,
            palette_id = DSL_COLOR_PREDEFINED_PALETTE_GREY,
            alpha = 0.6)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        #-----------------------------------------------------------------------------
        # Next, we create the ODE Heat-Mapper with a set number of rows and columns
        # using a 9:6 ratio so that the map entries will be square. In this example,
        # we use the South (center bottom) point on the Bounding Box to obtain samples.
        
        # New ODE Heat-Mapper to map the Person occurrences overtime
        retval = dsl_ode_heat_mapper_new('person-heat-mapper',
            cols = 64, 
            rows = 36, 
            bbox_test_point = DSL_BBOX_POINT_SOUTH,
            color_palette = COLOR_PALETTES[color_palette_index])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Enable and setup display of the heat-map legend.
        retval = dsl_ode_heat_mapper_legend_settings_set('person-heat-mapper',
            enabled = True,
            location = DSL_HEAT_MAP_LEGEND_LOCATION_TOP, 
            width = 2, 
            height = 2)
            
        # Add the ODE Heat-Mapper to the Person Occurrence Trigger.
        retval = dsl_ode_trigger_heat_mapper_add('person-occurrence-trigger', 
            heat_mapper='person-heat-mapper')
        if retval != DSL_RETURN_SUCCESS:
            break

        #`````````````````````````````````````````````````````````````````````````````
        # Next, create the ODE Pad Probe Handler (PPH) and increase the number of
        # Display metadata structures so that the cols * rows sqaures can be displayed.
        #    64 * 36 / 16 elements (rectangles) per structure = 144. Since we only
        # car about mapping people walking on the sidewalk (left half of view) we can 
        # reduce the number by at least 50%. The Display metadata is allocated/freed 
        # on each frame and therefore will add overhead to the Pipeline's execution.
        
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions    
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Increase the number of display-meta structures that are allocated 
        # (default=1) per frame to handle all of the rectangles in the heat map.
        retval = dsl_pph_ode_display_meta_alloc_size_set('ode-handler', 70)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Add the two Triggers to the ODE PPH to be invoked on every frame. 
        retval = dsl_pph_ode_trigger_add_many('ode-handler', 
            triggers=['every-occurrence-trigger', 'person-occurrence-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # New URI File Source using the filespec defined above
        retval = dsl_source_file_new('uri-source', uri_h265, True)
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
