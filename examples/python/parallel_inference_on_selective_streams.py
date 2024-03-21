################################################################################
# The MIT License
#
# Copyright (c) 2023-2024, Prominence AI, Inc.
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
# each with their own Inference Components (Preprocessors, Inference Engines, 
# Trackers, for example). 
# IMPORTANT! All branches are (currently) using the same model engine and config.
# files, which is not a valid use case. The actual inference components and 
# models to use for any specific use cases is beyond the scope of this example. 
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
# In this example, 4 sources are added to the Pipeline:
#   - branch-1 will process streams [0,1]
#   - branch-2 will process streams [1,2]
#   - branch-3 will process streams [0,2,3]
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

from dsl import *

file_path1 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4"
file_path2 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov"
file_path3 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_walk.mov"
file_path4 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"

# All branches are currently using the same config and model engine files
# which is pointless... The example will be updated to use multiple

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file_1 = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file_1 = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

primary_infer_config_file_2 = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file_2 = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

primary_infer_config_file_3 = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file_3 = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

# Filespecs for the Secondary GIE

sgie2_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt'
sgie2_model_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet.etlt_b8_gpu0_int8.engine'


PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

# Source dimensions 720p
SOURCE_WIDTH = 1280
SOURCE_HEIGHT = 720

TILER_WIDTH = 1280
TILER_HEIGHT = 720

SINK_WIDTH = TILER_WIDTH
SINK_HEIGHT = TILER_HEIGHT

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
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        # 34new File Sources to produce streams 0 through 3
        retval = dsl_source_file_new('source-1', file_path1, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-2', file_path2, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-3', file_path3, True)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_file_new('source-4', file_path4, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Inference Branch #1 (b1) - single Primary GIE.  Branch component
        # is NOT required if using single component.
        
        ## New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('pgie-b1', 
            primary_infer_config_file_1, primary_model_engine_file_1, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Inference Branch #2 (b2) - Primary GIE and IOU Tracker.
        
        retval = dsl_infer_gie_primary_new('pgie-b2', 
            primary_infer_config_file_2, primary_model_engine_file_2, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and height
        retval = dsl_tracker_new('tracker-b2', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_branch_new_component_add_many('branch-b2', 
            ['pgie-b2', 'tracker-b2', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # ----------------------------------------------------------------------------
        # Inference Branch #3 (b3) - Primary GIE, Tracker, and Secondary GIE.

        retval = dsl_infer_gie_primary_new('pgie-b3', 
            primary_infer_config_file_3, primary_model_engine_file_3, 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_tracker_new('tracker-b3', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_infer_gie_secondary_new('vehicletype-sgie-b3', 
            sgie2_config_file, sgie2_model_file, 'pgie-b3', 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_branch_new_component_add_many('branch-b3', 
            ['pgie-b3', 'tracker-b3', 'vehicletype-sgie-b3', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        ## ----------------------------------------------------------------------------
        #
        # We create a new Remuxer component, that demuxes the batched streams, and
        # then Tee's the unbatched single streams into multiple branches. Each Branch 
        #   - connects to some or all of the single stream Tees as specified. 
        #   - re-muxes the streams back into a single batched stream for processing.
        #   - each branch is then linked to the Remuxer's Metamuxer
        retval = dsl_remuxer_new('remuxer')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Define our stream-ids for each branch. Available stream-ids are [0,1,2,3]
        stream_ids_b1 = [0,1]
        stream_ids_b2 = [1,2]
        stream_ids_b3 = [0,2,3]

        # IMPORTANT! Use the "add_to" service to add the Branch to specific streams,
        # or use the "add" service to connect to all streams.
        
        # Add pgie-b1 to the Remuxer to connect to specific streams - stream_ids_b1
        retval = dsl_remuxer_branch_add_to('remuxer', 'pgie-b1', 
            stream_ids = stream_ids_b1, 
            num_stream_ids = len(stream_ids_b1))
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add branch-1 to the Remuxer to connect to specific streams - stream_ids_b1
        retval = dsl_remuxer_branch_add_to('remuxer', 'branch-b2', 
            stream_ids = stream_ids_b1, 
            num_stream_ids = len(stream_ids_b1))
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add branch-2 to the Remuxer to connect to specific streams - stream_ids_b2
        retval = dsl_remuxer_branch_add_to('remuxer', 'branch-b3', 
            stream_ids = stream_ids_b3, 
            num_stream_ids = len(stream_ids_b3))
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------

        # Create the Object Detection Event (ODE) Pad Probe Handler (PPH)
        # and add the handler to the src-pad (output) of the Remuxer
        retval = dsl_pph_ode_new('ode-pph')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_remuxer_pph_add('remuxer', 'ode-pph', DSL_PAD_SRC)
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

        # Create a Person Instance Trigger to trigger on new
        # instances (new tracker-ids) of the bicycle-class
        retval = dsl_ode_trigger_instance_new('person-instance-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
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

        # Add all three Triggers to the ODE-PPH; bicycle, vehicle, and person.
        retval = dsl_pph_ode_trigger_add_many('ode-pph', 
            ['bicycle-instance-trigger', 'vehicle-instance-trigger', 
            'person-instance-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # New Tiler, setting width and height, use default cols/rows set by 
        # the number of streams
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Next, create the On-Screen-Displays (OSD) with text, clock and bboxes.
        # enabled. 

        # New OSD 
        retval = dsl_osd_new('osd', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # New Window Sink
        retval = dsl_sink_window_egl_new('window-sink',
            0, 0, SINK_WIDTH, SINK_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to both Window Sinks
        retval = dsl_sink_window_key_event_handler_add('window-sink', 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add('window-sink', 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ----------------------------------------------------------------------------
        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['source-1', 'source-2', 'source-3', 'source-4',
            'remuxer', 'tiler', 'osd', 'window-sink', None])
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
