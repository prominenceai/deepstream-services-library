"""
The MIT License

Copyright (c) 2019-Present, Michael Patrick

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

#######################################################################################
#
#   Name:            ds_rtsp_svr.py
#
#   Description:    This program takes as input a video and streams it as an rtsp server 
#
#                   Usage: python3 ds_rtsp_svr.py <video file>
#
#######################################################################################
"""
import sys
sys.path.insert(0,"../../")
import os
from dsl import *


####################################################
#   This callback handles EOS messages
####################################################

def eos_listener(client_data):
    global eos_count
    if eos_count == 0:
        print("EOS Callback primed")
        eos_count = 1
    else :
        print("EOS shutdown received")
        eos_count = 0
        dsl_main_loop_quit()

####################################################
# Function to be called on XWindow ButtonPress event
####################################################
def xwindow_button_event_handler(xpos, ypos, client_data):
    print('button pressed: xpos = ', xpos, ', ypos = ', ypos)

# Function to be called on XWindow KeyRelease event
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline1')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline1')
    elif key_string.upper() == 'Q':
        print('Kill main loop')
        dsl_main_loop_quit()


#####################################################
#
#   Function: test1()
#
#   Description:  This test creates a pipeline, adds
#   a video file as a URI and a sink.  It then plays
#   the video and when it ends, it tears down the
#   pipeline
#
#####################################################

def test1():
    """Execute test """
    uri = os.path.abspath(sys.argv[1])
    rtsp_source_uri = 'rtsp://raspberrypi.local:8554/'
    
    #########################################
    #   Create source
    #########################################
    retVal = dsl_source_uri_new("video1", uri, False, 0, 0, 0)
    
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    #   Create CSI source
    #########################################
    retVal = dsl_source_csi_new('my-csi-source', 1280, 720, 30, 1)

    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    #   Create rtsp source
    #########################################
    retVal = dsl_source_rtsp_new('rtsp1', rtsp_source_uri, DSL_RTP_ALL, 0, 0, 0)

    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    #   Create Display 
    #########################################

    retVal = dsl_tiler_new("display1", 1280, 720)
    
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    # New OSD with clock enabled... using default values.
    retval = dsl_osd_new('on-screen-display', False)
    if retval != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))
    
    #########################################
    #   Create Display Overlay 
    #########################################

    retVal = dsl_sink_overlay_new("sink1", 0,1,1,100, 10, 1200, 1200)

    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    #   Create sink window
    #########################################

    retVal = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal)) 

    #########################################
    #   Create rtsp sink 
    #########################################

    retVal = dsl_sink_rtsp_new("rtsp-sink1", "sscsmatrix-desktop.local", 5400, 8554, DSL_CODEC_H264, 4000000,0)
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal)) 

    #########################################
    #   Create pipeline1
    #########################################

    retVal = dsl_pipeline_new("pipeline1")
    
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    #  Add components to pipeline 
    #########################################
    
#    retVal = dsl_pipeline_component_add_many('pipeline1', ['rtsp1', 'display1', 'on-screen-display', 'window-sink', 'rtsp-sink1', None])
    retVal = dsl_pipeline_component_add_many('pipeline1', ['my-csi-source', 'display1', 'on-screen-display', 'window-sink', 'rtsp-sink1', None])
    
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))


    #########################################
    #   Add an EOS listener
    #########################################

    retVal = dsl_pipeline_eos_listener_add('pipeline1', eos_listener, None)

    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    # Add the XWindow event handler functions defined above
    #########################################
    
    retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline1", xwindow_key_event_handler, None)
    if retval != DSL_RETURN_SUCCESS:
       print(dsl_return_value_to_string(retVal))

    retval = dsl_pipeline_xwindow_button_event_handler_add("pipeline1", xwindow_button_event_handler, None)
    if retval != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    # Play video
    #########################################

    result = dsl_pipeline_play("pipeline1")
    if retVal != DSL_RETURN_SUCCESS:
        print(dsl_return_value_to_string(retVal))

    #########################################
    # Run main loop
    #########################################

    dsl_main_loop_run()

    #########################################
    # Stop  pipeline
    #########################################

    dsl_pipeline_stop("pipeline1")

    #########################################
    # Remove pipeline
    #########################################

    dsl_pipeline_delete_all()

    #########################################
    # Remove component 
    #########################################

    dsl_component_delete_all()

def main():
    """Parse the command line parameters and run test """
    if len(sys.argv) < 2:
        print("")
        print("#################################################################")
        print("#")
        print("#    Error: Missing source file name.")
        print("#    Calling sequence: python3 ds_rtsp_svr.py <Video source file>")
        print("#")
        print("##################################################################")
    else:
        for x in range(1):
            print("Starting run number: ",x)
            test1()

if __name__ == '__main__':
    eos_count = 0
    main()
