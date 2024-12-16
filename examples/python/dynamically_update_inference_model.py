################################################################################
# The MIT License
#
# Copyright (c) 2024, Prominence AI, Inc.
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
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary GST Inference Engine (PGIE)
#   - DCF Tracker
#   - Secondary GST Inference Engines (SGIEs)
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to dynamically update an Inference Engine's config and model files.
#  
# The key-release handler function will dynamically update the Secondary
# Inference Engine's config-file on the key value as follows.
#
#    "1" = '../../test/config/config_infer_secondary_vehicletypes.yml'
#    "2" = '../../test/config/config_infer_secondary_vehiclemake.yml'
#
# The new model engine is loaded by the SGIE asynchronously. a client listener 
# (callback) function is added to the SGIE to be notified when the loading is
# complete. See the "model_update_listener" function defined below.
#   
# IMPORTANT! it is best to allow the config file to specify the model engine
# file when updating both the config and model. Set the model_engine_file 
# parameter to None when creating the Inference component.
#  
#        retval = dsl_infer_gie_secondary_new('secondary-gie', 
#            secondary_infer_config_file_1, None, 'primary-gie', 0)
#
# The Config files used are located under /deepstream-services-library/test/config
# The files reference models created with the file 
#     /deepstream-services-library/make_trafficcamnet_engine_files.py
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
import time

from dsl import *

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Secondary Inference Engine config files.
secondary_infer_config_file_1 = \
    '../../test/config/config_infer_secondary_vehicletypes.yml'
secondary_infer_config_file_2 = \
    '../../test/config/config_infer_secondary_vehiclemake.yml'

# flag to indicate if a model engine file update is in progress.
model_updating = False

# Filespec for the NvDCF Tracker config file
dcf_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml'

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    
    global model_updating

    print('key released = ', key_string)

    if key_string.upper() == '1' and model_updating is False:
        model_updating = True
        print('Result of start engine update =', dsl_return_value_to_string(
            dsl_infer_config_file_set('secondary-gie', 
                secondary_infer_config_file_1)))
        
    elif key_string.upper() == '2' and model_updating is False:
        model_updating = True
        print('Result of start engine update =', dsl_return_value_to_string(
            dsl_infer_config_file_set('secondary-gie', 
                secondary_infer_config_file_2)))
        
    elif key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
        dsl_pipeline_stop('pipeline')
        dsl_main_loop_quit()
 
##
# Function to be called when a model update has been completed
#  
def model_update_listener(name, model_engine_file, client_data):

    global model_updating
    print(name, "completed loading model", model_engine_file)

    model_updating = False

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

        # New File Source using the file path specified above, repeat disabled.
        retval = dsl_source_file_new('file-source', uri_h265, False)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # First new Secondary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_secondary_new('secondary-gie', 
            secondary_infer_config_file_1, None, 'primary-gie', 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the model-update-listener callback to the Secodary GIE
        retval = dsl_infer_engine_model_update_listener_add('secondary-gie',
            model_update_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New NvDCF Tracker, setting operation width and height
        # NOTE: width and height paramaters must be multiples of 32 for dcf
        retval = dsl_tracker_new('dcf-tracker', 
            config_file = dcf_tracker_config_file,
            width = 640, 
            height = 384) 
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', text_enabled=True, 
            clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets with reduced dimensions
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, 1280, 720)
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

        # Add all the components to a new pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['file-source', 'primary-gie', 'dcf-tracker', 
             'secondary-gie', 'on-screen-display', 
            'egl-sink', None])
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

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
