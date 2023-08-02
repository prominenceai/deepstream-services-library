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

current_stream_id=0

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
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
            
def change_stream_id():
    global current_stream_id
    
    current_stream_id = (current_stream_id+1)%4

    # we first call the TEE base class service to remove the branch.
    if dsl_tee_branch_remove('demuxer', 'branch-0') == DSL_RETURN_SUCCESS:
        
        time.sleep(2)
        
        # we then call the Demuxer service to add it back at the next stream-id
        retval = dsl_tee_demuxer_branch_add_at('demuxer', 'branch-0', current_stream_id)
        
#
# Thread function to start and wait on the main-loop
#
def main_loop_thread_func():

    # blocking call
    dsl_main_loop_run()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # First new Streaming Image Source
        retval = dsl_source_file_new('source-0', uri_h265, True)
#        retval = dsl_source_image_stream_new('source-0', image_0, False, 5, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-1', uri_h265, True)
#        retval = dsl_source_image_stream_new('source-1', image_1, False, 5, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-2', uri_h265, True)
#        retval = dsl_source_image_stream_new('source-2', image_2, False, 5, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-3', uri_h265, True)
#        retval = dsl_source_image_stream_new('source-3', image_3, False, 5, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_source_file_new('source-4', uri_h265, True)
#        retval = dsl_source_image_stream_new('source-4', image_4, False, 5, 1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with infer interval
        retval = dsl_infer_gie_primary_new('primary-gie', inferConfigFile, modelEngineFile, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! when creating the Demuxer, we need to set the maximum number
        # of branches equal to the number of Source Streams, even though we will 
        # only be using a single branch. The Demuxer needs to allocate a source-
        # pad for each stream prior to playing so that the Branch can be moved from 
        # stream to stream while the Pipeline is in a state of PLAYING.
        retval = dsl_tee_demuxer_new('demuxer', max_branches=5)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create the remaining components for the single demuxed branch
        
        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_sync_enabled_set('window-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break
        # New Fake Sink
        retval = dsl_sink_fake_new('fake-sink')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_overlay_new('overlay-sink', 0, 0, 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_sync_enabled_set('overlay-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_branch_new_component_add_many('branch-0',
            ['on-screen-display', 'overlay-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_tee_demuxer_branch_add_at('demuxer', 'branch-0', stream_id=3)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_tee_demuxer_branch_add_at('demuxer', 'window-sink', stream_id=4)
        if retval != DSL_RETURN_SUCCESS:
            break
       

        # Add the XWindow event handler functions defined above to the Window Sink
        retval = dsl_sink_window_key_event_handler_add('window-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('window-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break


        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['source-0', 'source-1', 'source-2', 'source-3', 'source-4', 
            'primary-gie', 'iou-tracker', 'demuxer', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        cur_source_count = 1

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        main_loop_thread = Thread(target=main_loop_thread_func)
        main_loop_thread.start()

        for i in range(200):
        
            sleep(4)

            # we first call the TEE base class service to remove the branch.
            retval = dsl_tee_branch_remove('demuxer', 'branch-0')
            if retval != DSL_RETURN_SUCCESS:
                break

            # we then call the Demuxer service to add it back at the next stream-id
            retval = dsl_tee_demuxer_branch_add_at('demuxer', 'branch-0', i%4)
            if retval != DSL_RETURN_SUCCESS:
                break
        
#        timer.cancel()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
