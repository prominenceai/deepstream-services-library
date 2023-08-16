################################################################################
# The MIT License
#
# Copyright (c) 2023, Prominence AI, Inc.
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
# This example shows how to use a single dynamic demuxer branch with a 
# multi-source Pipeline. The Pipeline trunk consists of:
#   - 5 Streaming Images Sources - each streams a single image at a given 
#       frame-rate with a number overlayed representing the stream-id.
#   - The Pipeline's built-in streammuxer muxes the streams into a
#       batched stream as input to the Inference Engine.
#   - Primary GST Inference Engine (PGIE).
#   - IOU Tracker.
#
# The dynamic branch will consist of:
#   - On-Screen Display (OSD)
#   - Window Sink - with window-delete and key-release event handlers.
# 
# The branch is added to one of the Streams when the Pipeline is constructed
# by calling:
#
#    dsl_tee_demuxer_branch_add_to('demuxer', 'branch-0', stream_id)
#
# Once the Pipeline is playing, the example uses a simple periodic timer to 
# call a callback function which advances/cycles the current stream_id 
# variable and moves the branch by calling.
#
#    dsl_tee_demuxer_branch_move_to('demuxer', 'branch-0', stream_id)
#  
################################################################################
#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")
from time import sleep
from threading import Timer, Thread

from dsl import *


# Each png file has a unique id [1..4] overlayed on the picture
# This makes it easy to see which stream the branch is connected to.
image_0 = "../../test/streams/sample_720p.0.png"
image_1 = "../../test/streams/sample_720p.1.png"
image_2 = "../../test/streams/sample_720p.2.png"
image_3 = "../../test/streams/sample_720p.3.png"
image_4 = "../../test/streams/sample_720p.4.png"

# Filespecs for the Primary GIE
inferConfigFile = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
modelEngineFile = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

##
# Global variable - indicates the current stream_id for the 
# single branch (branch-0). The variable will be updated as
#     stream_id = (stream_id+1)%num_streams
# to cycle through each of the source 4 streams. The other branch,
# which is a sole window sink, is connected to the 5th stream
# (stream_id=4) at all times.
stream_id = 0

##
# Number of Source streams for the Pipeline
num_streams = 5

# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_pipeline_stop('pipeline')
    dsl_main_loop_quit()

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    global MAX_SOURCE_COUNT, cur_source_count
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()

##
# Supper class derived from Timer to implement a simple
# timer service to call a callback function periodically
# on Timer timeout.
##
class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

##
# Callback function to move the dynamic branch (branch-0) from
# its current stream to next stream-id in the cycle.
def move_branch():
    global stream_id, num_streams
    
    # set the stream-id to the next stream in the cycle of 5.
    stream_id = (stream_id+1)%num_streams
    
    # we then call the Demuxer service to add it back at the specified stream-id
    print("dsl_tee_demuxer_branch_move_to() returned", 
        dsl_tee_demuxer_branch_move_to('demuxer', 'branch-0', stream_id))

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        global stream_id
        
        # ----------------------------------------------------------------------------
        # Create the five streaming-image sources that will provide the streams for 
        # the single dynamic branch.
        
        retval = dsl_source_image_stream_new('source-0', image_0, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_image_stream_new('source-1', image_1, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_image_stream_new('source-2', image_2, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_image_stream_new('source-3', image_3, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break       
        retval = dsl_source_image_stream_new('source-4', image_4, True, 15, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Create the PGIE and Tracker components that will become the
        # fixed pipeline-trunk to process all batched sources. The Demuxer will
        # demux/split the batched streams back to individual source streams.
        
        # New Primary GIE using the filespecs above, with infer interval
        retval = dsl_infer_gie_primary_new('primary-gie', 
            inferConfigFile, modelEngineFile, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and height
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Next, create the OSD and Overlay-Sink. These two components will make up
        # the dynamic branch. The dynamic branch will be moved from stream to stream
        # i.e. from demuxer pad to pad.

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! the default Window-Sink (and Overlay-Sink) settings must by
        # updated to support dynamic Pipeline updates... specifically, we need to 
        # disable the "sync", "max-lateness", and "qos" properties.
        retval = dsl_sink_window_new('dynamic-sink',
            300, 300, 1280, 720)

        # Disable the "sync" setting    
        retval = dsl_sink_sync_enabled_set('dynamic-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Disable the "max-lateness" setting
        retval = dsl_sink_max_lateness_set('dynamic-sink', -1)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Disable the "qos" setting
        retval = dsl_sink_qos_enabled_set('dynamic-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to the Window Sink
        retval = dsl_sink_window_key_event_handler_add('dynamic-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('dynamic-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create the dynamic branch with the OSD and Window Sink.
        retval = dsl_branch_new_component_add_many('branch-0',
            ['on-screen-display', 'dynamic-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! when creating the Demuxer, we need to set the maximum number
        # of branches equal to the number of Source Streams, even though we will 
        # only be using one branch. The Demuxer needs to allocate a source-pad
        # for each stream prior to playing so that the dynamic Branch can be 
        # moved from stream to stream while the Pipeline is in a state of PLAYING.
        retval = dsl_tee_demuxer_new('demuxer', max_branches=5)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the branch to the Demuxer at stream_id=0
        retval = dsl_tee_demuxer_branch_add_to('demuxer', 'branch-0', stream_id)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['source-0', 'source-1', 'source-2', 'source-3', 'source-4',
            'primary-gie', 'iou-tracker', 'demuxer', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        timer = RepeatTimer(5, move_branch)
        timer.start()
        
        # blocking call
        dsl_main_loop_run()
        
        timer.cancel()

        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
