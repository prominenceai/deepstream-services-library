################################################################################
# The MIT License
#
# Copyright (c) 2019-2020, Robert Howell. All rights reserved.
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
sys.path.insert(0, '../../')
import time
from dsl import *

host_uri = 'define-host-uri-here'

# Filespecs for the Primary GIE
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_fp16.engine'

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New CSI Live Camera Source
        retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)
        if retval != DSL_RETURN_SUCCESS:
            print(retval)
            break

        # New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            print(retval)
            break

        # New Tiled Display, setting width and height, use default cols/rows set by source count
        retval = dsl_demuxer_new('demuxer')
        if retval != DSL_RETURN_SUCCESS:
            print(retval)
            break

        # New OSD with clock enabled... using default values.
        retval = dsl_osd_new('on-screen-display', True)
        if retval != DSL_RETURN_SUCCESS:
            print(retval)
            break

        retval = dsl_sink_overlay_new('overlay-sink', 1, 0, 0, 100, 100, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break

        retVal = dsl_sink_rtsp_new('rtsp-sink', host_uri, 5400, 8554, DSL_CODEC_H265, 4000000,0)
        if retVal != DSL_RETURN_SUCCESS:
            print(dsl_return_value_to_string(retVal)) 
            
        ### Important
        ### using a demuxer for this example, so add the OSD and sinks directly to the Source
        retval = dsl_source_osd_add('csi-source', 'on-screen-display')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_sink_add('csi-source', 'overlay-sink')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_sink_add('csi-source', 'rtsp-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Pipeline to use with the above components
        retval = dsl_pipeline_new('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_component_add_many('pipeline', ['csi-source', 'primary-gie', 'demuxer', None])

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
    dsl_source_sink_remove('csi-source', 'rtsp-sink')
    dsl_source_sink_remove('csi-source', 'overlay-sink')
    dsl_source_osd_remove('csi-source')
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))

