
################################################################################
# The MIT License
#
# Copyright (c) 2019-2023, Prominence AI, Inc.
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

################################################################################
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - RTSP Source
#   - Primary GST Inference Engine (PGIE)
#   - DCF Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - error-message events
#   - Pipeline change-of-state events
#   - RTSP Source change-of-state events.
#  
# IMPORTANT! The error-message-handler callback fucntion will stop the Pipeline 
# and main-loop, and then exit. If the error condition is due to a camera
# connection failure, the application could choose to let the RTSP Source's
# connection manager periodically reattempt connection for some length of time.
#
################################################################################

#!/usr/bin/env python

import sys
import time

from dsl import *

# RTSP Source URI for AMCREST Camera    
amcrest_rtsp_uri = 'rtsp://username:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'    

# RTSP Source URI for HIKVISION Camera    
hikvision_rtsp_uri = 'rtsp://username:password@192.168.1.64:554/Streaming/Channels/101'    

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the NvDCF Tracker config file
dcf_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml'

# IMPORTANT! "DCF Tracker width and height paramaters must be multiples of 32
tracker_width = 640 
tracker_height = 384

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
# Function to be called with every error message received
# by the Pipeline bus manager
##
def error_message_handler(source, message, client_data):
    print('Error: source = ', source, ' message = ', message)
    dsl_pipeline_stop('pipeline')
    dsl_main_loop_quit()
    
## 
# Function to be called on every change of Pipeline state
## 
def pipeline_state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

## 
# Function to be called on every change of RTSP Source state
## 
def rtsp_state_change_listener(old_state, new_state, client_data):
    print('RTSP Source previous state = ', 
        old_state, ', new state = ', new_state)

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
            
        # New RTSP Source for the specific RTSP URI with a timeout of 10s
        # IMPORTANT! a timeout > 0 enables the source's connection management.   
        retval = dsl_source_rtsp_new('rtsp-source',     
            uri = hikvision_rtsp_uri,  # using hikvision URI defined above   
            protocol = DSL_RTP_ALL,    # use RTP ALL protocol
            skip_frames = 0,           # decode every frame
            drop_frame_interval = 0,   # decode every frame  
            latency=1000,              # 1000 ms of jitter buffer
            timeout=10)                # 10 second new buffer timeout   
        if (retval != DSL_RETURN_SUCCESS):    
            return retval    

        # Add the RTSP state-change listener calback to our RTSP Source
        retval = dsl_source_rtsp_state_change_listener_add('rtsp-source',
            rtsp_state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:    
            break
            
        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New NvDCF Tracker, setting operation width and height
        retval = dsl_tracker_new('dcf-tracker', 
            config_file = dcf_tracker_config_file,
            width = tracker_width, 
            height = tracker_height) 
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', text_enabled=True, 
            clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets with reduced dimensions
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above
        retval = dsl_sink_window_key_event_handler_add('egl-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('egl-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to a new pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['rtsp-source', 'primary-gie', 'dcf-tracker', 'on-screen-display', 
            'egl-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the error-message handler defined above
        retval = dsl_pipeline_error_message_handler_add('pipeline', 
            error_message_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            pipeline_state_change_listener, None)
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

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))