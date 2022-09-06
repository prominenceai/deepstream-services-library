################################################################################
# The MIT License
#
# Copyright (c) 2022, Prominence AI, Inc.
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

################################################################################
# This script can be used to generate the tensorflow Resnet caffemodel engine
# files using the config files under the installed NVIDIA Samples folder.
#
# Default is set to nano - Swap/update the primary config pathspec for other platforms.

# Test URI used for all sources
uri = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4'


# Config file for the Primary GIE

# inferConfigFile = \
#     '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
inferConfigFile = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'

# Config files for the Secondary GIEs
sgie1_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carcolor.txt'
sgie2_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carmake.txt'
sgie3_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt'

TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
TILER_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT

## 
# Function to be called on End-of-Stream (EOS) event
## 
def eos_event_listener(client_data):
    print('Pipeline EOS event')
    dsl_main_loop_quit()
        

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        #
        # Create the Pipeline components
        # ... starting with eight URI File Sources
        
        retval = dsl_source_uri_new('Camera 1', uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        dsl_source_uri_new('Camera 2', uri, False, False, 0)
        dsl_source_uri_new('Camera 3', uri, False, False, 0)
        dsl_source_uri_new('Camera 4', uri, False, False, 0)
        dsl_source_uri_new('Camera 5', uri, False, False, 0)
        dsl_source_uri_new('Camera 6', uri, False, False, 0)
        dsl_source_uri_new('Camera 7', uri, False, False, 0)
        dsl_source_uri_new('Camera 8', uri, False, False, 0)

        # New Primary GIE using the filespecs above, with interval and Id. Setting the
        # model_engine_files parameter to None allows for model generation if not found.
        retval = dsl_infer_gie_primary_new('primary-gie', inferConfigFile, None, interval=10)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Secondary GIEs using the filespecs above with interval = 0
        retval = dsl_infer_gie_secondary_new('carcolor-sgie', sgie1_config_file, None, 'primary-gie', 10)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_infer_gie_secondary_new('carmake-sgie', sgie2_config_file, None, 'primary-gie', 10)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_infer_gie_secondary_new('vehicletype-sgie', sgie3_config_file, None, 'primary-gie', 10)
        if retval != DSL_RETURN_SUCCESS:
            break

        #----------------------------------------------------------------------------------------------------
        # Create one of each Tracker Types, KTL and IOU, to test with each. We will only add one
        # at a time, but it's easier to create both and just update the Pipeline assembly below as needed.

        # New KTL Tracker, setting max width and height of input frame
        retval = dsl_tracker_ktl_new('ktl-tracker', 480, 288)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiler, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Fake Sink to terminate the stream.
        retval = dsl_sink_fake_new('fake-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        #----------------------------------------------------------------------------------------------------
        # Pipeline assembly
        #
        # New Pipeline (trunk) with our Sources, Tracker, and Pre-Tiler  as last component
        # Note: *** change 'iou-tracker' to 'ktl-tracker' to try both. KTL => higher CPU load 
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4', 'Camera 5', 'Camera 6',  
            'Camera 7', 'Camera 8', 'primary-gie', 'ktl-tracker', 'carcolor-sgie', 
            'carmake-sgie', 'vehicletype-sgie', 'tiler', 'fake-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the pipeline
        retval = dsl_pipeline_play('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Once the pipeline is playing, the model engine files will have been saved. 
        # Safe to stop the pipline and quit now.
        retval = dsl_pipeline_stop('pipeline')
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
