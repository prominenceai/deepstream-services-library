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

import sys    
import time    
from dsl import *    

# RTSP Source URI for AMCREST Camera    
amcrest_rtsp_uri = 'rtsp://username:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'    

# RTSP Source URI for HIKVISION Camera    
hikvision_rtsp_uri = 'rtsp://username:password@192.168.0.14:554/Streaming/Channels/101'    

WINDOW_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH    
WINDOW_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT    

##     
# Function to be called on XWindow KeyRelease event    
##     
def xwindow_key_event_handler(key_string, client_data):    
    print('key released = ', key_string)    
    if key_string.upper() == 'P':    
        dsl_player_pause('player')    
    elif key_string.upper() == 'R':    
        dsl_player_play('player')    
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':    
        dsl_player_stop('player')
        dsl_main_loop_quit()
       
def main(args):    

    # Since we're not using args, we can Let DSL initialize GST on first call    
    while True:    

        # For each camera, create a new RTSP Source for the specific RTSP URI    
        retval = dsl_source_rtsp_new('rtsp-source',     
            uri = amcrest_rtsp_uri,     
            protocol = DSL_RTP_ALL,     
            intra_decode = False,     
            drop_frame_interval = 0,     
            latency=100,
            timeout=2)    
        if (retval != DSL_RETURN_SUCCESS):    
            return retval    

        # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display    
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)    
        if retval != DSL_RETURN_SUCCESS:    
            break    

        retval = dsl_player_new('rtsp-player', 'rtsp-source', 'window-sink')
        if retval != DSL_RETURN_SUCCESS:    
            break
            
        # Add the XWindow event handler functions defined above    
        retval = dsl_player_xwindow_key_event_handler_add("rtsp-player", xwindow_key_event_handler, None)    
        if retval != DSL_RETURN_SUCCESS:    
            break    
        
        # Play the player    
        retval = dsl_player_play('rtsp-player')    
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
