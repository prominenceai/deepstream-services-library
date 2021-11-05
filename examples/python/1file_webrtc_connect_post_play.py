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

#-------------------------------------------------------------------------------------------
#
# This script demonstrates the use of a WebRtc Sink

# File path for the single File Source
file_path = '/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_qHD.mp4'

stun_server = "stun://stun.l.google.com:19302"

# Window Sink Dimensions
sink_width = 640
sink_height = 360

CUR_SINK_NUMBER = 0
MAX_SINK_NUMBER = 4

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'C':
        print('closing connection')
        dsl_sink_webrtc_connection_close('webrtc-sink')
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
    # dsl_main_loop_quit()

##
# Function to be called on every change of Pipeline state
##
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")


def webrtc_sink_client_listener(connection_data_ptr, client_data):

    connection_data = connection_data_ptr.contents

    print('Current connection state for the WebRTC Sink is =', 
        connection_data.current_state)

##
# Function to be called on every socket connection event
##
def websocket_server_client_listener_cb(path, client_data):

    global CUR_SINK_NUMBER, MAX_SINK_NUMBER
    print("Incoming Websocket connection event with path =", path)

    if CUR_SINK_NUMBER == MAX_SINK_NUMBER:
        print('maximum number of WebRTC Sinks have been created')
        return
    CUR_SINK_NUMBER += 1

    sink_name = 'webrtc-sink-{}'.format(CUR_SINK_NUMBER)
    print(sink_name)
    # New WebRTC Sink with .
    retval = dsl_sink_webrtc_new(sink_name,
        stun_server = stun_server,
        turn_server = None,
        codec = DSL_CODEC_H264,
        bitrate = 4000000,
        interval = 0)
    if retval != DSL_RETURN_SUCCESS:
        print('failed to create new WebRTC Sink')
        return

    # Add the client listener callback function to the WebRTC Sink
    retval = dsl_sink_webrtc_client_listener_add(sink_name,
        webrtc_sink_client_listener, None)
    if retval != DSL_RETURN_SUCCESS:
        print('failed to add client listner to sink')
        return

    retval = dsl_pipeline_component_add('pipeline', sink_name)
    if retval != DSL_RETURN_SUCCESS:
        print('failed to add new WebRTC Sink to Pipeline')
        return

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # First new URI File Source
        # retval = dsl_source_rtsp_new('source', 	
        #     uri = rtsp_uri, 	
        #     protocol = DSL_RTP_ALL, 	
        #     cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 	
        #     intra_decode = False, 	
        #     drop_frame_interval = 0, 	
        #     latency = 100,
        #     timeout = 5)	
        # if (retval != DSL_RETURN_SUCCESS):
        #     break

        # New File Source using the file path specified above, repeat enabled.
        retval = dsl_source_file_new('source', file_path, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_new('window-sink', 0, 0, sink_width, sink_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the window sink to a new pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline',
            ['source', 'window-sink', None])
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

        # Add the client listener to the Websocket Server to listen for socket connections
        retval = dsl_websocket_server_client_listener_add(websocket_server_client_listener_cb, None)

        # Start the Websocket Server listening on the default port number
        retval = dsl_websocket_server_listening_start(DSL_WEBSOCKET_SERVER_DEFAULT_HTTP_PORT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Join with main loop until released - blocking call
        dsl_main_loop_run()
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
