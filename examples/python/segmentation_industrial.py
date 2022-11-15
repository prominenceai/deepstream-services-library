################################################################################
# The MIT License
#
# Copyright (c) 2021-2021, Prominence AI, Inc.
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
import time
from dsl import *
import pyds

file_path = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_industrial.jpg'

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-segmentation-test/dstest_segmentation_config_industrial.txt'

# Segmentation Visualizer output dimensions should (typically) match the
# inference dimensions defined in segvisual_config_industrial.txt (512x512)
width = 512
height = 512

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

def segvisual_src_pad_buffer_probe(buffer, user_data):

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

        # TODO Handle segmentation meta data
        
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return True
    
    
def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # New URI Image Source using the files path defined above, not simulating
        # a live source, stream at 15 hz, and generate EOS after 10 seconds.
        retval = dsl_source_uri_new('image-source', 
            uri = file_path, 
            is_live = False,
            intra_decode = False,
            drop_frame_interval = False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the Config filespec above, with interval and Id
        # Setting the model_engine_file parameter to None == attemp to create model 
        retval = dsl_infer_gie_primary_new('primary-gie',
            infer_config_file = primary_infer_config_file, 
            model_engine_file = None, 
            interval = 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Segmentation Visualizer with output dimensions
        retval = dsl_segvisual_new('segvisual', width=width, height=height)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 100, 1000, width, height)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Example of how to force the aspect ratio during window resize
        dsl_sink_window_force_aspect_ratio_set('window-sink', force=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Custom Pad Probe Handler to handle the Segmentatin Meta Data
        retval = dsl_pph_custom_new('custom-pph', 
            client_handler = segvisual_src_pad_buffer_probe, 
            client_data = None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Custom Pad Probe Handler to the src pad of the Segmentation Visualizer
        retval = dsl_segvisual_pph_add('segvisual', 'custom-pph')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['image-source', 'primary-gie', 'segvisual', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Set the Streammuxer dimensions to the same as GIE Config and Sink dimensions
        retval = dsl_pipeline_streammux_dimensions_set("pipeline", 
            width=width, height=height)

        # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline", 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_delete_event_handler_add("pipeline", 
            xwindow_delete_event_handler, None)
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

