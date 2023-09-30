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
# This example shows how to use a Remuxer Component to create parallel branches,
# each with their own Primary Inference Engine (PGIE) and Multi-Object Tracker. 
# Both branches are (currently) using the same model engine and config files.
# A real-world example would use different models.
#
# Each Branch added to the Remuxer can specify which streams to process or
# to process all. Use the Remuxer "branch-add-to" service to add to specific streams.
#
#    stream_ids = [0,1]
#    dsl_tee_remuxer_branch_add_to('my-remuxer', 'my-branch-0', 
#        stream_ids, len[stream_ids])
#
# You can use the base Tee "branch-add" service if adding to all streams
#
#    dsl_tee_branch_add('my-remuxer', 'my-branch-0')
# 
# In this example, 3 sources are added to the Pipeline:
#   - branch-1 will process streams [0,1]
#   - branch-2 will process stream [2]
#
# Note: typically, multiple branches would process at least some of the same
# stream-ids, otherwise, using multiple Pipelines would make more sense.
#
# Each Branch has a unique Object Detection Event (ODE) Pad Probe Handler (PPH)
# added to the source pad (output) of their respective Trackers.
#
# Three ODE Instance Triggers are created to trigger on new object instances
# events (i.e. new tracker ids). Each is filtering on a unique class-i
# (vehicle, person, and bicycle). 
#
# The first two ODE Triggers are added to 'ode-pph-1', the third to 'ode-pph-2'.
#
# A single ODE Print Action is created and added to each Trigger (shared action).
# Using multiple Print Actions running in parallel -- each writing to the same 
# stdout buffer -- will result in the printed data appearing interlaced. A single 
# Action with an internal mutex will protect from stdout buffer reentrancy. 
# 
################################################################################
#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")
from time import sleep
from threading import Timer, Thread

from dsl import *

file_path1 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4"
file_path2 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov"
file_path3 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov"

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

# Scale all streammuxer output buffers to 720p
MUXER_WIDTH = 1280
MUXER_HEIGHT = 720

# Branch 1 will use a Tiler to tile two streams. Branch 2 will process single stream
TILER_1_WIDTH = 1920
TILER_1_HEIGHT = 540

# Sink-1 (for Branch-1) to use same dimensions as Tiler
SINK_1_WIDTH = TILER_1_WIDTH
SINK_1_HEIGHT = TILER_1_HEIGHT

# Sink-2 (for Branch-2) to use same dimensions as the streammuxer
SINK_2_WIDTH = MUXER_WIDTH
SINK_2_HEIGHT = MUXER_HEIGHT


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
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        # 3 new File Sources to produce streams 0 through 3
        retval = dsl_source_file_new('source-1', file_path1, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-2', file_path2, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-3', file_path3, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Create two PGIE's and two Trackers, one for each parallel branch.
        # Note: this example will be updated in the future to use different models
        
        ## New Primary GIE using the filespecs above with interval = 0
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED):
            retval = dsl_infer_gie_primary_new('pgie-1', 
                primary_infer_config_file_jetson, primary_model_engine_file_jetson, 4)
            if retval != DSL_RETURN_SUCCESS:
                break
            retval = dsl_infer_gie_primary_new('pgie-2', 
                primary_infer_config_file_jetson, primary_model_engine_file_jetson, 4)
            if retval != DSL_RETURN_SUCCESS:
                break
        else:
            retval = dsl_infer_gie_primary_new('pgie-1', 
                primary_infer_config_file_dgpu, primary_model_engine_file_dgpu, 4)
            if retval != DSL_RETURN_SUCCESS:
                break
            retval = dsl_infer_gie_primary_new('pgie-2', 
                primary_infer_config_file_dgpu, primary_model_engine_file_dgpu, 4)
            if retval != DSL_RETURN_SUCCESS:
                break

        # New IOU Tracker, setting operational width and height
        retval = dsl_tracker_new('tracker-1', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_tracker_new('tracker-2', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # We create the two Object Detection Event (ODE) Pad Probe Handlers (PPH)
        retval = dsl_pph_ode_new('ode-pph-1')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_new('ode-pph-2')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add one to each of the Tracker output source pads.
        retval = dsl_tracker_pph_add('tracker-1', 'ode-pph-1', DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_tracker_pph_add('tracker-2', 'ode-pph-2', DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Bicycle Instance Trigger to trigger on new
        # instances (new tracker-ids) of the bicycle-class
        retval = dsl_ode_trigger_instance_new('bicycle-instance-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            class_id = PGIE_CLASS_ID_BICYCLE, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Bicycle Instance Trigger to trigger on new
        # instances (new tracker-ids) of the vehicle-class
        retval = dsl_ode_trigger_instance_new('vehicle-instance-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            class_id = PGIE_CLASS_ID_VEHICLE, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Bicycle Instance Trigger to trigger on new
        # instances (new tracker-ids) of the bicycle-class
        retval = dsl_ode_trigger_instance_new('person-instance-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add both the Bicycle and Vehicle Instance Triggers to the ODE-PPH
        # that will be added to the first branch.
        retval = dsl_pph_ode_trigger_add_many('ode-pph-1', 
            ['bicycle-instance-trigger', 'vehicle-instance-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Person Instance Trigger to the ODE-PPH that will be added
        # to the second branch
        retval = dsl_pph_ode_trigger_add('ode-pph-2',  'person-instance-trigger')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Single new ODE Print Action that will be shared by all Triggers.
        retval = dsl_ode_action_print_new('print-action', force_flush=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Single Action to each of the Instance Triggers.
        retval = dsl_ode_trigger_action_add('bicycle-instance-trigger', 
            'print-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('person-instance-trigger', 
            'print-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add('vehicle-instance-trigger', 
            'print-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiler, setting width and height, use default cols/rows set by 
        # the number of streams
        retval = dsl_tiler_new('tiler-1', TILER_1_WIDTH, TILER_1_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
        #retval = dsl_tiler_tiles_set('tiler-1', columns=2, rows=1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Next, create the On-Screen-Displays (OSD) with text, clock and bboxes.
        # enabled. 

        # New OSD 
        retval = dsl_osd_new('osd-1', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_osd_new('osd-2', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Two new Window Sinks, one for each Branch
        retval = dsl_sink_window_new('window-sink-1',
            0, 0, SINK_1_WIDTH, SINK_1_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_new('window-sink-2',
            300, 300, SINK_2_WIDTH, SINK_2_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to both Window Sinks
        retval = dsl_sink_window_key_event_handler_add('window-sink-1', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_key_event_handler_add('window-sink-2', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('window-sink-1', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('window-sink-2', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create the two branches and all of their components
        retval = dsl_branch_new_component_add_many('branch-1',
            ['pgie-1', 'tracker-1', 'tiler-1', 'osd-1', 'window-sink-1', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_branch_new_component_add_many('branch-2',
            ['pgie-2', 'tracker-2', 'osd-2', 'window-sink-2', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ## ----------------------------------------------------------------------------
        #
        # We create a new Remuxer component, that demuxes the batched streams, and
        # then Tee's the unbatched single streams into multiple branches. Each Branch 
        #   - connects to some or all of the single stream Tees as specified. 
        #   - re-muxes the streams back into a single batched stream for processing.
        retval = dsl_tee_remuxer_new('remuxer')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Define our stream-ids for each branch. Available stream-ids are [0,1,2]
        stream_ids_1 = [1,2]
        stream_ids_2 = [2]
        
        # IMPORTANT! Use the "add_to" service to add the Branch to specific stream,
        # or use the common base Tee "add" service to connect to all streams.
        
        # Add branch-1 to the Remuxer - select the appropriate service below
        retval = dsl_tee_remuxer_branch_add_to('remuxer', 'branch-1', 
            stream_ids = stream_ids_1, 
            num_stream_ids = len(stream_ids_1))
#        retval = dsl_tee_branch_add('remuxer', 'branch-1')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add branch-2 to the Remuxer to connect to stream 2.
        retval = dsl_tee_remuxer_branch_add_to('remuxer', 'branch-2', 
            stream_ids = stream_ids_2, 
            num_stream_ids = len(stream_ids_2))
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['source-1', 'source-2', 'source-3', 'remuxer', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        
        #----------------------------------------------------------------------------
        # Scale both muxers to the width and heigth specified at the top of the file.
        retval = dsl_pipeline_streammux_dimensions_set('pipeline',
            MUXER_WIDTH, MUXER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_tee_remuxer_dimensions_set('remuxer',
            MUXER_WIDTH, MUXER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # blocking call
        dsl_main_loop_run()
        
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
