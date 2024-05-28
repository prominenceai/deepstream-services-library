################################################################################
# The MIT License
#
# Copyright (c) 2019-2024, Prominence AI, Inc.
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
# This simple example demonstrates how to use OpenCV with NVIDIA's pyds.
# The Pipeline used in this example is built with :
#   - 4 URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - Tiler
#   - On-Screen Display (OSD)
#   - Window Sink
# 
# A Custom Pad-Probe-Handler is added to the Sink-Pad of the Tiler
# to process the frame meta-data for each buffer received. The handler
# demonstrates how to 
#   - use pyds.get_nvds_buf_surface() to get a buffer surface.
#   - convert a frame to numpy array format with np.array().
#   - convert the array into cv2 default BGRA format using cv2.cvtColor().
#   - save the array as an image using opencv cv2.imwrite().
#
# IMPORTANT! pyds.get_nvds_buf_surface() requires 
#   1. The color format of the buffer must be set to RGBA by calling
#      dsl_source_video_buffer_out_format_set()
#   2. The memory type must be set to DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED
#      if running on dGPU. This is done by calling 
#        * dsl_pipeline_streammux_nvbuf_mem_type_set() - if using old streammux
#        * dsl_component_nvbuf_mem_type_set_many() - with all sources if using 
#          new streammux. 
#     
# IMPORTANT! The output folders (1 per source) must be created first
#   ./stream_0, ./stream_1, ./stream_2, ./stream_3,
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
################################################################################

#!/usr/bin/env python

import sys
from dsl import *
import pyds
import numpy as np
import cv2

source_uri = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"


# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

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

## 
# Function to be called on End-of-Stream (EOS) event
## 
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

## 
# Custom PPH added to the sink-pad (input) of the Tiler
## 
def custom_pad_probe_handler(buffer, user_data):

    # Retrieve batch metadata from the gst_buffer
    # IMPORTANT! do not use the hash function to cast the buffer.
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break

        # Saving every 100th frame for each source.  
        if frame_meta.frame_num%100 == 0:

            n_frame = pyds.get_nvds_buf_surface(buffer, frame_meta.batch_id)

            # convert the python array into numpy array format.
            frame_image = np.array(n_frame,copy=True,order='C')

            # covert the array into cv2 default BGRA format
            frame_image = cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)

            filename = "./stream_"+str(frame_meta.pad_index) + \
                "/frame_"+str(frame_meta.frame_num)+".jpg"
            print(filename)

            # write out the image
            cv2.imwrite(filename,frame_image)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return DSL_PAD_PROBE_OK

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # 4 New URI File Sources using the filespec defined above
        retval = dsl_source_uri_new('uri-source-0', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_source_uri_new('uri-source-1', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_source_uri_new('uri-source-2', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_source_uri_new('uri-source-3', source_uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! We must set the buffer format to RGBA for each source.        
        retval = dsl_source_video_buffer_out_format_set('uri-source-0', 
            DSL_VIDEO_FORMAT_RGBA)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_source_video_buffer_out_format_set('uri-source-1', 
            DSL_VIDEO_FORMAT_RGBA)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_source_video_buffer_out_format_set('uri-source-2', 
            DSL_VIDEO_FORMAT_RGBA)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_source_video_buffer_out_format_set('uri-source-3', 
            DSL_VIDEO_FORMAT_RGBA)
        if retval != DSL_RETURN_SUCCESS:
            break
            
           
        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiler with dimensions for two tiles - for the two sources
        retval = dsl_tiler_new('tiler', WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Custom Pad Probe Handler to call Nvidia's example callback 
        # for handling the Batched Meta Data
        retval = dsl_pph_custom_new('custom-pph', 
            client_handler=custom_pad_probe_handler, client_data=None)

        # Add the custom PPH to the Sink pad (input) of the Tiler.
        retval = dsl_tiler_pph_add('tiler', 
            handler='custom-pph', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break
                # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and dimensions defined above.
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, 
            WINDOW_WIDTH, WINDOW_HEIGHT)
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

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source-0', 'uri-source-1', 'uri-source-2', 'uri-source-3', 
             'primary-gie', 'iou-tracker',  'tiler', 'on-screen-display', 
             'egl-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # IMPORTANT! pyds.get_nvds_buf_surface requires the buffer memory to use
        # DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED if running on dGPU
        if dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_DISCRETE:

            # If using the new Streammux, then change the memory type of each source
            if dsl_info_use_new_nvstreammux_get():
                retval = dsl_component_nvbuf_mem_type_set_many(
                    ['uri-source-0', 'uri-source-1', 'uri-source-2', 'uri-source-3'],
                    DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED)
                
            # if using the old Streammux we set the memtype of the Streammux itself.    
            else:
                retval = dsl_pipeline_streammux_nvbuf_mem_type_set('pipeline',
                    DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED)
        
        if retval != DSL_RETURN_SUCCESS:
            break
        
        ## Add the listener callback functions defined above
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

        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
