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

file_path = "../../test/streams/sample_1080p_h264.mp4"
image_path = "../../test/streams/first-person-occurrence-438.jpeg"


## 
# Function to be called on Player termination event
## 
def player_termination_event_listener(client_data):
    print('player termination event')
    dsl_main_loop_quit()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New File Source using the filespec defined above
        retval = dsl_source_file_new('file-source', file_path=file_path, repeat_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Image Source using the filespec defined above, with a display timeout of 5 seconds
        retval = dsl_source_image_new('image-source', 
            file_path = image_path, 
            is_live = False,
            fps_n = 1,
            fps_d = 1,
            timeout = 5)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Query the source for the image dimensions to use for our sink dimensions
        retval, image_width, image_height = dsl_source_dimensions_get('image-source')

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, width=image_width, height=image_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Overlay Sink,  x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_overlay_new('overlay-sink', 
             display_id = 0, 
            depth = 0,
            offset_x = 50, 
            offset_y = 50, 
            width = image_width, 
            height = image_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Media Player using the File Source and Window Sink
        retval = dsl_player_new('player1',
            source='image-source', sink='overlay-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Termination listener callback to the Player
        retval = dsl_player_termination_event_listener_add('player1',
            client_listener=player_termination_event_listener, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the Player until end-of-stream (EOS)
        retval = dsl_player_play('player1')
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
