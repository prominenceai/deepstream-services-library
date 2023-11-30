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

#!/usr/bin/env python

import sys
import time

from dsl import *

#-------------------------------------------------------------------------------------
# This example shows the use of a Video Dewarper to dewarp a 360d camera stream 
#   - recorded from a 360d camera and provided by NVIDIA as a sample stream.
#
# The Dewarper component is created with the following parameters
#   - a config "file config_dwarper_txt" which tailors this 360d camera 
#     multi-surface use-case.
#   - and a camera-id which refers to the first column of the CSV files 
#     (i.e. csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv).
#     The dewarping parameters for the given camera are read from CSV 
#     files and used to generate dewarp surfaces (i.e. multiple aisle 
#     and spot surface) from 360d input video stream.
# All files are located under:
#   /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/

# Sample 360 degree camera stream from NVIDIA's smart parking example.
input_stream = \
    '/opt/nvidia/deepstream/deepstream/samples/streams/sample_cam6.mp4'

# IMPORTANT! --------------------------
# Config file specific to 360d dewarping - uses csv_files/nvaisle_2M.csv & 
# csv_files/nvspot_2M.csv
dwarper_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper.txt'

# Filespecs for the Primary GIE and IOU Trcaker
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

# Using the same values for streammux dimensions as found in NVIDIAs dewarper example
# /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/deepstream_dewarper_test.c.
streammux_width = 960
streammux_height = 752

# Need to scale the tiler and sink so that all 4 dewarped surfaces -- output from 
# the dewarper -- can be viewed.
tiler_width = streammux_width//2
tiler_height = streammux_height*2
sink_width = tiler_width
sink_height = tiler_height

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

# Function to be called on End-of-Stream (EOS) event
def eos_event_listener(client_data):
    print('Pipeline EOS event')
    dsl_pipeline_stop('pipeline')
    dsl_main_loop_quit()

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New File Source using the file path specified above, repeat diabled.
        retval = dsl_source_file_new('file-source', input_stream, False)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Dewarper - config file is perspective - camera_id is NOT used as
        retval = dsl_dewarper_new('360-dewarper', 
            config_file = dwarper_config_file,
            camera_id = 6)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # We add the Dewarper directly to the source... not the Pipeline.
        retval = dsl_source_video_dewarper_add('file-source', '360-dewarper')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and height.
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # A tiler is required to combine the 4 dewarped output surface into
        # a single stream for the OSD and Window Sink,
        retval = dsl_tiler_new('tiler', tiler_width, tiler_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and dimensions 
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, sink_width, sink_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to the Window Sink
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
            ['file-source', 'primary-gie', 'iou-tracker', 'tiler', 'on-screen-display', 
            'egl-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # -----------------------------------------------------------------------------
        # IMPORTANT! We need to set the Stream-muxer's number of surfaces per frame to 
        # 4 in order to handle the 4 decoded output surfaces produced by the Dewarper.
        retval = dsl_pipeline_streammux_num_surfaces_per_frame_set('pipeline', 4)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # -----------------------------------------------------------------------------
        # IMPORTANT! We need to set the Stream-muxer's batch-size equal to the
        # number of sources (1) times the number of surfaces per frame (4)
        retval = dsl_pipeline_streammux_batch_size_set('pipeline', 
            batch_size=4)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Update the Pipeline's Streammux dimensions to match the source dimensions.
        retval = dsl_pipeline_streammux_dimensions_set('pipeline',
            streammux_width, streammux_height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
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

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
