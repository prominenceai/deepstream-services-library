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

################################################################################
# This example demonstrates how to manually control -- using key release and 
# button press events -- the 2D Tiler's output stream to: 
#   - show a specific source on key input (source No.) or mouse click on tile.
#   - to return to showing all sources on 'A' key input, mouse click, or timeout.
#   - to cycle through all sources on 'C' input showing each for timeout.
# 
# Note: timeout is controled with the global variable SHOW_SOURCE_TIMEOUT 
################################################################################

import sys
import time

from dsl import *

file_path1 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"
file_path2 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4"
file_path3 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov"
file_path4 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov"

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'

tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

# Window Sink Dimensions - used to create the sink, however, in this
# example the Pipeline XWindow service is called to enabled full-sreen
TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
TILER_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT

#WINDOW_WIDTH = TILER_WIDTH
#WINDOW_HEIGHT = TILER_HEIGHT
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

SHOW_SOURCE_TIMEOUT = 2

# Function to be called on End-of-Stream (EOS) event
def eos_event_listener(client_data):
    print('Pipeline EOS event')
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
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    
    global SHOW_SOURCE_TIMEOUT

    # P = pause pipline
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
        
    # R = resume pipeline, if paused
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
        
    # Q or Esc = quit application
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()
        
    # if one of the unique soure Ids, show source
    elif key_string >= '0' and key_string <= '3':
        retval, source = dsl_source_name_get(int(key_string))
        if retval == DSL_RETURN_SUCCESS:
            dsl_tiler_source_show_set('tiler', source=source, timeout=SHOW_SOURCE_TIMEOUT, has_precedence=True)
            
    # C = cycle All sources
    elif key_string.upper() == 'C':
        dsl_tiler_source_show_cycle('tiler', timeout=SHOW_SOURCE_TIMEOUT)

    # A = show All sources
    elif key_string.upper() == 'A':
        dsl_tiler_source_show_all('tiler')

## 
# Function to be called on XWindow Button Press event
## 
def xwindow_button_event_handler(button, x_pos, y_pos, client_data):
    print('button = ', button, ' pressed at x = ', x_pos, ' y = ', y_pos)
    
    global SHOW_SOURCE_TIMEOUT

    if (button == Button1):
        # get the current XWindow dimensions - the XWindow was overlayed with our Window Sink
        retval, width, height = dsl_pipeline_xwindow_dimensions_get('pipeline')
        
        # call the Tiler to show the source based on the x and y button cooridantes
        # and the current window dimensions obtained from the XWindow
        dsl_tiler_source_show_select('tiler', x_pos, y_pos, width, height, timeout=SHOW_SOURCE_TIMEOUT)

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # Create two predefined RGBA colors, white and black, that will be
        # used to create text to display the source number on each stream. 
        retval = dsl_display_type_rgba_color_predefined_new('full-white', 
            color_id = DSL_COLOR_PREDEFINED_WHITE, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        retval = dsl_display_type_rgba_color_predefined_new('full-black', 
            color_id = DSL_COLOR_PREDEFINED_BLACK, alpha = 1.0)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        retval = dsl_display_type_rgba_font_new('arial-18-white', 
            font='arial', size=18, color='full-white')
        if retval != DSL_RETURN_SUCCESS:
            return retval
            
        # Create a new "source-number" display-type using the new RGBA
        # colors and font created above.
        retval = dsl_display_type_source_number_new('source-number', 
            x_offset=15, y_offset=20, font='arial-18-white', 
            has_bg_color=True, bg_color='full-black')
        if retval != DSL_RETURN_SUCCESS:
            return retval
            
        # Create a new Action to add the display-type's metadata
        # to a frame's meta on invocation.
        retval = dsl_ode_action_display_meta_add_new('add-souce-number', 
            display_type='source-number')
        if retval != DSL_RETURN_SUCCESS:
            return retval

        # Create an ODE Always triger to call the "add-meta" Action to display
        # the source number on every frame for each source. 
        retval = dsl_ode_trigger_always_new('always-trigger', 
            source=DSL_ODE_ANY_SOURCE, when=DSL_ODE_PRE_OCCURRENCE_CHECK)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        retval = dsl_ode_trigger_action_add('always-trigger', 
            action='add-souce-number')
        if retval != DSL_RETURN_SUCCESS:
            return retval
            
        # Create a new ODE Pad Probe Handler (PPH) to add to the Tiler's Src Pad
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the Trigger to the ODE PPH which will be added to the Tiler below.
        retval = dsl_pph_ode_trigger_add('ode-handler', trigger='always-trigger')
        if retval != DSL_RETURN_SUCCESS:
            break

        # 4 new File Sources
        retval = dsl_source_file_new('file-source-1', file_path1, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        dsl_source_file_new('file-source-2', file_path2, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        dsl_source_file_new('file-source-3', file_path3, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        dsl_source_file_new('file-source-4', file_path4, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new('iou-tracker', 
            tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiler, setting width and height, use default cols/rows set by 
        # the number of sources
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_tiler_pph_add('tiler', 'ode-handler', DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', text_enabled=True, 
            clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', ['file-source-1', 
            'file-source-2', 'file-source-3', 'file-source-4', 'primary-gie', 
            'iou-tracker', 'tiler', 'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Enabled the XWindow for full-screen-mode
        retval = dsl_pipeline_xwindow_fullscreen_enabled_set('pipeline', enabled=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the EOS listener and XWindow event handler functions defined above
        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_key_event_handler_add('pipeline', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_button_event_handler_add('pipeline', 
            xwindow_button_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_delete_event_handler_add('pipeline', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Start and join the main-loop
        dsl_main_loop_run()
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
