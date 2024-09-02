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
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - CSI Source
#   - Primary GST Inference Engine (PGIE)
#   - On-Screen Display
#   - 3D Sink
#   - RTSP Sink
# ...and how to add them to a new Pipeline and play.
#
# IMPORTANT! this examples uses a CSI Camera Source and 3D Sink - Jetson only!
#
################################################################################

#!/usr/bin/env python

import sys
import time
from dsl import *

#  RTSP Server Sink: host uri of 0.0.0.0 means "use any available network interface"
host_uri = '0.0.0.0'

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New CSI Live Camera Source
        retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New 3D Window Sink with 0 x/y offsets, and same dimensions as Camera Source
        retval = dsl_sink_window_3d_new('window-sink', 0, 0, 
            SOURCE_WIDTH, SOURCE_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New RTSP Server Sink 
        retVal = dsl_sink_rtsp_server_new('rtsp-sink', 
            host = "0.0.0.0",              # 0.0.0.0 = "this host, this network."
            udp_port = 5400,               # UDP port 5400 uses the Datagram Protocol.             
            rtsp_port = 8554,              # 
            encoder = DSL_ENCODER_HW_H265, # High Efficiency Video Coding (HEVC)
            bitrate = 0,                   # Set to 0 to use plugin default (4000000)
            iframe_interval = 0)           # 0 = encode everyframe           
        if retVal != DSL_RETURN_SUCCESS:
            print(dsl_return_value_to_string(retVal)) 

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['csi-source', 'primary-gie', 'on-screen-display', 
            'window-sink', 'rtsp-sink', None])
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

    dsl_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))

