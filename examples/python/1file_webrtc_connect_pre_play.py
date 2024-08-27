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
import time

from dsl import *
from os import wait

#-------------------------------------------------------------------------------------------
#
# This script demonstrates the use of a WebRtc Sink

# File path for the single File Source
file_path = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4'

stun_server = "stun://stun.l.google.com:19302"

# Window Sink Dimensions
sink_width = 640
sink_height = 360


def is_socket_open(socket_open):
    if socket_open:
        print('Pipeline is playing')
        return True
    return False

socket_open = False

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    global socket_open
    print('key released = ', key_string)
    if key_string.upper() == 'C':
        print('closing connection')
    elif key_string.upper() == 'O':
        socket_open = True
    elif key_string.upper() == 'P':
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

## 
# Function to be called on every change of Websocket connection state
## 
def webrtc_sink_client_listener(connection_data_ptr, client_data):
    
    global socket_open

    connection_data = connection_data_ptr.contents

    print('Current connection state for the WebRTC Sink is =',
        connection_data.current_state)

    if connection_data.current_state == DSL_SOCKET_CONNECTION_STATE_INITIATED:

        # Remote client has initiated a Websocket connection
        # time to play the pipeline
        print( 'Play pipeline returned', dsl_pipeline_play('pipeline'))

        socket_open = True

    elif connection_data.current_state == DSL_SOCKET_CONNECTION_STATE_CLOSED:
        # Remote client has closed the Websocket connection
        print( 'Stop pipeline returned', dsl_pipeline_stop('pipeline'))
        dsl_main_loop_quit()


def main(args):

    global socket_open

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New File Source using the file path specified above, repeat enabled.
        retval = dsl_source_file_new('source', file_path, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New WebRTC Sink with .
        retval = dsl_sink_webrtc_new('webrtc-sink',
            stun_server = stun_server, 
            turn_server = None,
            codec = DSL_CODEC_H264,
            bitrate = 4000000,
            interval = 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the client listener callback function to the WebRTC Sink
        retval = dsl_sink_webrtc_client_listener_add('webrtc-sink',
            webrtc_sink_client_listener, None)

        # New Window Sink, 0 x/y offsets and dimensions
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, sink_width, sink_height)
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
            ['source', 'egl-sink', 'webrtc-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_eos_listener_add('pipeline', 
            eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Start the Websocket Server listening on the default port number
        retval = dsl_websocket_server_listening_start(DSL_WEBSOCKET_SERVER_DEFAULT_HTTP_PORT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # NOTE: The pipeline is started in the WebRTC Client listener callback
        # when the remote client initiates a connection.
        print('Waiting for remote client to initiate a connection')

        dsl_main_loop_run()
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
