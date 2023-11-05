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
# This example demonstrates the use of a Frame-Capture Sink to encode and
# save video frames to JPEG files on client/viewer demand.
#
# An ODE Frame-Capture Action is provided to The Frame-Capture Sink on creation.
# A client "capture_complete_listener" is added to the the Action to be notified
# when each new file is saved (the ODE Action performs the actual frame-capture).
#
# Child Players (to play the captured image) and Mailers (to mail the image) can
# be added to the ODE Frame-Capture action as well (not shown).
#
# A Custom Pad Probe Handler (PPH) is added to the sink-pad of the OSD component
# to process every buffer flowing over the pad by:
#    - Retrieving the batch-metadata and its list of frame metadata structures
#      (only frame per batched-buffer with 1 Source)
#    - Retrieving the list of object metadata structures from the frame metadata.
#    - Iterate through the list of objects looking for the first occurrence of
#      a bicycle. 
#    - If detected, the current frame-number is schedule to be captured by the
#      Frame-Capture Sink using its Frame-Capture Action.
#
#          dsl_sink_frame_capture_schedule('frame-capture-sink', 
#                   frame_meta.frame_num)
#
# IMPORT All captured frames are copied and buffered in the Sink's processing
# thread. The encoding and saving of each buffered frame is done in the 
# g-idle-thread, therefore, the capture-complete notification is asynchronous.
#

#!/usr/bin/env python

import sys
import time
import pyds

from dsl import *

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file_jetson = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file_jetson = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
primary_infer_config_file_dgpu = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file_dgpu = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

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

## 
# Function to be called on Object Capture (and file-save) complete
## 
def capture_complete_listener(capture_info_ptr, client_data):
    print(' ***  Object Capture Complete  *** ')
    
    capture_info = capture_info_ptr.contents
    print('capture_id: ', capture_info.capture_id)
    print('filename:   ', capture_info.filename)
    print('dirpath:    ', capture_info.dirpath)
    print('width:      ', capture_info.width)
    print('height:     ', capture_info.height)

## 
# Custom Pad Probe Handler function called with every buffer
## 
def custom_pad_probe_handler(buffer, user_data):

    # Retrieve batch metadata from the gst_buffer
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

        # Iterate through the list of object metadata for this frame looking
        # for the first occurrence of a bicycle - our trigger to schedule
        # frame-capture with our . 
        
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
            except StopIteration:
                break
            
            # Check the class-id to see if it's the object type we're looking for
            if obj_meta.class_id == PGIE_CLASS_ID_BICYCLE:

                # IMPORTANT! This example is using a single Source and single
                # Frame-Capture Sink.  If using multiple sources with a Demuxer 
                # and multiple Frame-Capture Sinks you will need to check the 
                # 'frame_meta.source_id' to call on the correct Sink. 
                
                # Schedule the current frame to be captured by the Sink 
                if dsl_sink_frame_capture_schedule('frame-capture-sink', 
                   frame_meta.frame_num) != DSL_RETURN_SUCCESS:
                   print("Custom PPH failed to schedule frame-capture!")
                   
                # Once the frame has been scheduled for capture, there is no
                # need to keep processing so return now.
                return DSL_PAD_PROBE_OK
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return DSL_PAD_PROBE_OK
    
def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        ## First new URI File Source
        retval = dsl_source_file_new('file-source', uri_h265, True)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        ## New Primary GIE using the filespecs above with interval = 0
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED):
            retval = dsl_infer_gie_primary_new('primary-gie', 
                primary_infer_config_file_jetson, primary_model_engine_file_jetson, 0)
        else:
            retval = dsl_infer_gie_primary_new('primary-gie', 
                primary_infer_config_file_dgpu, primary_model_engine_file_dgpu, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', text_enabled=True, 
            clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Custom Pad Probe Handler to call the handler function above 
        # for handling the Batched Meta Data
        retval = dsl_pph_custom_new('custom-pph', 
            client_handler=custom_pad_probe_handler, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_osd_pph_add('on-screen-display', 'custom-pph', DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        ## New Window Sink, 0 x/y offsets and reduced dimensions
        retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_sink_sync_enabled_set('window-sink', False)
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

        # Create a new Capture Action to capture and encode the frame to jpeg image,
        # and save to file. Encoding and saving is done in the g-idle-thread.
        # Saving to the current directory. File names will be generated as
        #    <action-name>_<unique_capture_id>_<%Y%m%d-%H%M%S>.jpeg
        retval = dsl_ode_action_capture_frame_new('frame-capture-action',
            outdir = "./")
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the capture complete listener function to the action
        retval = dsl_ode_action_capture_complete_listener_add('frame-capture-action',
            capture_complete_listener, None)

        ## New Frame-Capture Sink created with the new Capture Action.
        retval = dsl_sink_frame_capture_new('frame-capture-sink', 
            'frame-capture-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to a new pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['file-source', 'primary-gie', 'iou-tracker', 'on-screen-display', 
            'window-sink', 'frame-capture-sink', None])
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

        # Join with main loop until released - blocking call
        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
