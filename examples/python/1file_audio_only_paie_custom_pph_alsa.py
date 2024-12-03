################################################################################
# The MIT License
#
# Copyright (c) 2019-2023, Prominence AI, Inc.
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
#   - URI Source - with media-type set to DSL_MEDIA_TYPE_AUDIO_ONLY
#   - Primary Audio Inference Engine (PAIE)
#   - ALSA Audio Sink to stream the audio to the default sound card.
# ...and how to add them to a new Pipeline and play
# 
#  
################################################################################

#!/usr/bin/env python

import sys
import time

from dsl import *

http_uri = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4'
uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    ''
primary_model_engine_file = \
    ''


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

        if not dsl_info_use_new_nvstreammux_get():
            print("New nvstreammux must be enabled for audio pipelines. set environment variable with:")
            print("export USE_NEW_NVSTREAMMUX=yes")
            retval = DSL_RESULT_FAILURE
            break

        ## New URI Source with HTTP URI
        retval = dsl_source_uri_new('uri-source', uri_h265, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_component_media_type_set('uri-source', 
            DSL_MEDIA_TYPE_AUDIO_ONLY)
        if retval != DSL_RETURN_SUCCESS:
            break
            
         # New alsa sink using default sound card and device
        retval = dsl_sink_alsa_new('alsa-sink', 'default')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New pipeline
        retval = dsl_pipeline_new('pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pipeline_audiomux_enabled_set('pipeline', True)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pipeline_videomux_enabled_set('pipeline', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to a new pipeline
        retval = dsl_pipeline_component_add_many('pipeline', 
            ['uri-source', 'alsa-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
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
