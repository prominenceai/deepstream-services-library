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

#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")
from time import sleep
from threading import Timer, Thread

from dsl import *

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

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
# single branch (branch-0). The varaible will be updated as
#     stream_id = (stream_id+1)%4
# to cycle through the first 4 streams. The other branch,
# which is a sole window sink, is connected to the 5th stream
# (stream_id=4) at all times.
stream_id = 0

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
    
    if key_string.upper() == 'C':  # only if using frame-capture-sink
        print ('Initiate capture returned', dsl_return_value_to_string(
            dsl_sink_frame_capture_initiate('dynamic-sink')))
            
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
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
    global stream_id
    
    # set the stream-id to the next stream in the cycle of 4.
    stream_id = (stream_id+1)%4
    
    # we then call the Demuxer service to add it back at the specified stream-id
    print("dsl_tee_demuxer_branch_move_to() returned", 
        dsl_tee_demuxer_branch_move_to('demuxer', 'branch-0', stream_id))

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        global stream_id
        
        # -------------------------------------------------------------
        # The first four sources will provide streams for the dynamic branch 
        
#        retval = dsl_source_file_new('source-0', uri_h265, True)
        retval = dsl_source_image_stream_new('source-0', image_0, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
#        retval = dsl_source_file_new('source-1', uri_h265, True)
        retval = dsl_source_image_stream_new('source-1', image_1, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
#        retval = dsl_source_file_new('source-2', uri_h265, True)
        retval = dsl_source_image_stream_new('source-2', image_2, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
#        retval = dsl_source_file_new('source-3', uri_h265, True)
        retval = dsl_source_image_stream_new('source-3', image_3, True, 10, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # The fith sources will provide the steady stream for the second
        # static branch that will consist of a single Window Sink
        
#        retval = dsl_source_file_new('source-4', uri_h265, True)
        retval = dsl_source_image_stream_new('source-4', image_4, True, 15, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Create the PGIE and Tracker components that will become the
        # fixed truck to process all batched sources. The Demuxer will
        
        # New Primary GIE using the filespecs above, with infer interval
        retval = dsl_infer_gie_primary_new('primary-gie', 
            inferConfigFile, modelEngineFile, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Next, create the OSD and Overlay-Sink. These two components will make up
        # the dynamic branch. The dynamic branch will be moved from
        # Source-0 t

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! the default Window-Sink and Overlay-Sink settings must by
        # udated to support dynamic Pipeline updates... specifically, we need to 
        # disable the "sync", "max-lateness", and "qos" properties.
        
        # New Overlay Sink, 0 x/y offsets
#        retval = dsl_sink_rtsp_new('dynamic-sink', 
#            '0.0.0.0', 5400, 8554, DSL_CODEC_H264, 4000000,0)
#        retval = dsl_sink_overlay_new('dynamic-sink', 0, 0, 
#            300, 300, 1280, 720)
        retval = dsl_sink_window_new('dynamic-sink',
            300, 300, 1280, 720)
#        retval = dsl_sink_fake_new('dynamic-sink')
#        retval = dsl_sink_image_multi_new('dynamic-sink', 
#            './frame_%04d.jpg', 640, 360, 1, 10)

#        retval = dsl_ode_action_capture_frame_new('frame-capture-action',
#            outdir = "./")
#        if retval != DSL_RETURN_SUCCESS:
#            break

        ## New Frame-Capture Sink created with the new Capture Action.
#        retval = dsl_sink_frame_capture_new('dynamic-sink', 
#            'frame-capture-action')

        if retval != DSL_RETURN_SUCCESS:
            break

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

        # Create the dynamic branch with the OSD and Window Sink.
        retval = dsl_branch_new_component_add_many('branch-0',
            ['on-screen-display', 'dynamic-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Create a Window Sink that will be added to the demuxer as a second branch
        # which will remain linked to stream-id=4 for the life of the Pipeline
        
        # New Window Sink, 0 x/y offsets
        retval = dsl_sink_window_new('static-sink', 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Disable the "sync" setting    
        retval = dsl_sink_sync_enabled_set('static-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Disable the "max-lateness" setting    
        retval = dsl_sink_max_lateness_set('static-sink', -1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Disable the "qos" processing
        retval = dsl_sink_qos_enabled_set('static-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to the Window Sink
        retval = dsl_sink_window_key_event_handler_add('static-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('static-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! when creating the Demuxer, we need to set the maximum number
        # of branches equal to the number of Source Streams, even though we will 
        # only be using two branches. The Demuxer needs to allocate a source-pad
        # for each stream prior to playing so that the one dynamic Branch can be 
        # moved from stream to stream while the Pipeline is in a state of PLAYING.
        retval = dsl_tee_demuxer_new('demuxer', max_branches=5)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the branchs to the Demuxer at stream_id=0
        retval = dsl_tee_demuxer_branch_add_to('demuxer', 'branch-0', stream_id)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_tee_demuxer_branch_add_to('demuxer', 'static-sink', 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['source-0', 'source-1', 'source-2', 'source-3', 'source-4',
            'primary-gie', 'iou-tracker', 'demuxer', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
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
