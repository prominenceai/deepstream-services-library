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
sys.path.insert(0, "../../")
from dsl import *
import time
import os

dir_path = "/opt/nvidia/deepstream/deepstream/samples/streams"

## 
# Function to be called on Player termination event
## 
def player_termination_event_listener(client_data):
    print('player termination event')
    dsl_main_loop_quit()
    
## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    
    # P = pause player
    if key_string.upper() == 'P':
        dsl_player_pause('player')
        
    # R = resume player, if paused
    elif key_string.upper() == 'R':
        dsl_player_play('player')
        
    # N = advance and play next
    elif key_string.upper() == 'N':
        dsl_player_render_next('player')
        
    # Q or Esc = quit application
    elif key_string.upper() == 'Q' or key_string == '':
        dsl_main_loop_quit()
    

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        for file in os.listdir(dir_path):
            if file.endswith(".mp4"):
            
                # create the Player on first file found
                if not dsl_player_exists('player'):
                
                    # New Video Render Player to play all the MP4 files found
                    retval = dsl_player_render_video_new('player', 
                        file_path = os.path.join(dir_path, file),
                        render_type = DSL_RENDER_TYPE_WINDOW,
                        offset_x = 0, 
                        offset_y = 0, 
                        zoom = 50, 
                        repeat_enabled = False)
                    if retval != DSL_RETURN_SUCCESS:
                        break
                else:
                    retval = dsl_player_render_file_path_queue('player', 
                        file_path = os.path.join(dir_path, file))

        # Add the Termination listener callback to the Player
        retval = dsl_player_termination_event_listener_add('player',
            client_listener=player_termination_event_listener, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_player_xwindow_key_event_handler_add('player', xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the Player until end-of-stream (EOS)
        retval = dsl_player_play('player')
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
