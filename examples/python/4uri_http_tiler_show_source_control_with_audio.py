################################################################################
# The MIT License
#
# Copyright (c) 2025, Prominence AI, Inc.
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
#
# This example demonstrates how to manually control -- using key release and 
# button press events -- the 2D Tiler's output stream to: 
#   - show a specific source on key input (source No.) or mouse click on tile.
#   - to return to showing all sources on 'A' key input, mouse click, or timeout.
#   - to cycle through all sources on 'C' input showing each for timeout.
# 
# Note: timeout is controled with the global variable SHOW_SOURCE_TIMEOUT
# 
# The example uses 4 HTTP URI Source with their media-type set to
# DSL_MEDIA_TYPE_AUDIO_VIDEO.
#
# The Pipeline's built-in Audiomixer is enabled to mix all streams to a
# single combined stream.  The Audiomixer is setup to mute all sources when
# the Pipeline is first played. When the Tiler is called on to show a single 
# source, or cycle to a new source, the single Source's Audio is enabled. An 
# ALSA Sink is used to play the single stream.
# 
# The example uses a basic inference Pipeline consisting of:
#   - 4 HTTP URI Sources with both Audio and Video enabled
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - 2D Tiler
#   - On-Screen Display
#   - Window Sink
#   - ALSA Audio Sink
#  
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - show-source events 
#   - Pipeline change-of-state events
#   - Buffering-message events
################################################################################

import sys
import time

from dsl import *

uri_path1 = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
uri_path2 = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4"
uri_path3 = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
uri_path4 = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

# Window Sink Dimensions - used to create the sink, however, in this
# example the Sink's XWindow service is called to enabled full-sreen
TILER_WIDTH = DSL_1K_HD_WIDTH
TILER_HEIGHT = DSL_1K_HD_HEIGHT

#WINDOW_WIDTH = TILER_WIDTH
#WINDOW_HEIGHT = TILER_HEIGHT
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

SHOW_SOURCE_TIMEOUT = 10 # in units of seconds

# List of all source names, used when calling 
#   * dsl_component_media_type_set_many and
#   * dsl_pipeline_audiomix_mute_enabled_set_many
SOURCES = ['uri-source-1', 'uri-source-2', 'uri-source-3', 'uri-source-4', None]

buffering = False

## 
# Function to be called when a buffering-message is recieved on the Pipeline bus.
## 
def buffering_message_handler(source, percent, client_data):

    global buffering

    if percent == 100:
        print('playing pipeline - buffering complete at 100 %')
        dsl_pipeline_play('pipeline')
        buffering = False

    else:
        if not buffering:
            print('pausing pipeline - buffering starting at ', percent, '%')
            dsl_pipeline_pause('pipeline')
        buffering = True


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
            dsl_tiler_source_show_set('tiler', source=source, 
                timeout=SHOW_SOURCE_TIMEOUT, has_precedence=True)
            
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
        # Get the current XWindow dimensions from our Window Sink
        # The Window Sink is derived from Render Sink parent class so  
        # use the Render Sink API to get the dimensions.
        retval, width, height = dsl_sink_window_dimensions_get('window-sink')
        
        # call the Tiler to show the source based on the x and y button cooridantes
        # and the current window dimensions obtained from the XWindow
        dsl_tiler_source_show_select('tiler', 
            x_pos, y_pos, width, height, timeout=SHOW_SOURCE_TIMEOUT)

##
# Function to be called when the Tiler switches to a new 
# single Source stream or back to all Sources
##
def source_show_listener(name, source, stream_id, client_data):

    # start by muting all    
    dsl_pipeline_audiomix_mute_enabled_set_many('pipeline',
        SOURCES, True)
    
    # if now showing a single source
    if stream_id != -1:
        print(name, " is now showing Source =", source)

        # unmute the single source
        dsl_pipeline_audiomix_mute_enabled_set('pipeline',
            source, False)
    else:
        print(name, " is now showing all sources")    



def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # This example uses ODE services to overlay the stream-id (0 through 3) 
        # on to each stream for visual verification of which stream the
        #  
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
            
        # Create a new "source-stream-id" display-type using the new RGBA
        # colors and font created above.
        retval = dsl_display_type_source_stream_id_new('source-stream-id', 
            x_offset=15, y_offset=20, font='arial-18-white', 
            has_bg_color=True, bg_color='full-black')
        if retval != DSL_RETURN_SUCCESS:
            return retval
            
        # Create a new Action to add the display-type's metadata
        # to a frame's meta on invocation.
        retval = dsl_ode_action_display_meta_add_new('add-souce-stream-id', 
            display_type='source-stream-id')
        if retval != DSL_RETURN_SUCCESS:
            return retval

        # Create an ODE Always triger to call the "add-meta" Action to display
        # the source stream-id on every frame for each source. 
        retval = dsl_ode_trigger_always_new('always-trigger', 
            source=DSL_ODE_ANY_SOURCE, when=DSL_ODE_PRE_OCCURRENCE_CHECK)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        retval = dsl_ode_trigger_action_add('always-trigger', 
            action='add-souce-stream-id')
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

        if retval != DSL_RETURN_SUCCESS:
            break

        # ---------------------------------------------------------------------------
        # Four HTTP URI Sources
        retval = dsl_source_uri_new('uri-source-1', uri_path1, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-2', uri_path2, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-3', uri_path3, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-4', uri_path4, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # IMPORTANT! the Sources media-type must be updated to enable Audio
        retval = dsl_component_media_type_set_many(SOURCES, 
            DSL_MEDIA_TYPE_AUDIO_VIDEO)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ---------------------------------------------------------------------------
        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ---------------------------------------------------------------------------
        # New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_new('iou-tracker', 
            tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        #-----------------------------------------------------------
        # New Tiler, setting width and height, use default cols/rows set by 
        # the number of sources.
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # IMPORTANT!
        # We must explicity set the columns and rows in order to use
        # the dsl_tiler_source_show_select service to select a tile.
        retval = dsl_tiler_tiles_set('tiler', columns=2, rows=2)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the ODE Pad Probe Handler to the Sink pad of the Tiler
        retval = dsl_tiler_pph_add('tiler', 'ode-handler', DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the show-source listener function defined above
        retval = dsl_tiler_source_show_listener_add('tiler', 
            source_show_listener, None)
        
        #-----------------------------------------------------------
        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', text_enabled=True, 
            clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        #-----------------------------------------------------------
        # New 3D Window Sink with 0 x/y offsets, and same dimensions as Tiler output
        # EGL Sink runs on both platforms. 3D Sink is Jetson only.
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED):
            retval = dsl_sink_window_3d_new('window-sink', 0, 0, 
                WINDOW_WIDTH, WINDOW_HEIGHT)
        else:
            retval = dsl_sink_window_egl_new('window-sink', 0, 0, 
                WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add('window-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('window-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_button_event_handler_add('window-sink', 
            xwindow_button_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        #-----------------------------------------------------------
        # New alsa sink using default sound card and device
        retval = dsl_sink_alsa_new('alsa-sink', 'default')
        if retval != DSL_RETURN_SUCCESS:
            break

        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------------------
        retval = dsl_pipeline_new('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! the Pipeline's Audiomixer must be enabled before the Sources,
        # with media-type = Audio-Video can be added.
        retval = dsl_pipeline_audiomix_enabled_set('pipeline', True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_component_add_many('pipeline', ['uri-source-1', 
            'uri-source-2', 'uri-source-3', 'uri-source-4', 'primary-gie', 
            'iou-tracker', 'tiler', 'on-screen-display', 'window-sink', 'alsa-sink', 
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the EOS listener and XWindow event handler functions defined above
        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        ## Add the buffering-handler defined above to the pipeline
        retval = dsl_pipeline_buffering_message_handler_add('pipeline',
            buffering_message_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pipeline_audiomix_mute_enabled_set_many('pipeline', 
            sources=SOURCES, enabled=True)
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
