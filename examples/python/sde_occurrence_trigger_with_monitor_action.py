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
import pyds

from dsl import *

##
## CAUTION: THIS SCRIPT IS A WORK IN PROGRESS!

uri_http = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4'
uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-audio/configs/config_infer_audio_sonyc.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/SONYC_Audio_Classifier/sonyc_audio_classify.onnx_b2_gpu0_fp32.engine'

frame_size = 441000
hop_size = 110250
transform = \
    'melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,sample_rate=44100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,p2db_top_db=(float)80.0'

ENGINE                              = 0
MACHINERY_IMPACT                    = 1
NON_MACHINERY_IMPACT                = 2
POWERED_SAW                         = 3
ALERT_SIGNAL                        = 4
MUSIC                               = 5
HUMAN_VOICE                         = 6
DOG                                 = 7
SMALL_SOUNDING_ENGINE               = 8
MEDIUM_SOUNDING_ENGINE              = 9
LARGE_SOUNDING_ENGINE               = 10
ROCK_DRILL                          = 11
JACKHAMMER                          = 12
HOE_RAM                             = 13
PILE_DRIVER                         = 14
NON_MACHINERY_IMPACT                = 15
CHAINSAW                            = 16
SMALL_MEDIUM_ROTATING_SAW           = 17
LARGE_ROTATING_SAW                  = 18
CAR_HORN                            = 19
CAR_ALARM                           = 20
SIREN                               = 21
REVERSE_BEEPER                      = 22
STATIONARY_MUSIC                    = 23
MOBILE_MUSIC                        = 24
ICE_CREAM_TRUCK                     = 25
PERSON_OR_SMALL_GROUP_TALKING       = 26
PERSON_OR_SMALL_GROUP_SHOUTING      = 27
LARGE_CROUD                         = 28
AMPLIFIED_SPEECH                    = 29
DOG_BARKING_WHINING                 = 30

## 
# Callback function for the SDE Monitor Action - illustrates how to
# dereference the SDE 'info_ptr' and access the data fields.
# Note: you would normally use the SDE Print Action to print the info
# to the console window if that is the only purpose of the Action.
## 
def sde_occurrence_monitor(info_ptr, client_data):
    info = info_ptr.contents
    print('Trigger Name        :', info.trigger_name)
    print('  Unique Id         :', info.unique_sde_id)
    print('  NTP Timestamp     :', info.ntp_timestamp)
    print('  Source Data       : ------------------------')
    print('    Source Id       :', hex(info.source_info.source_id))
    print('    Batch Id        :', info.source_info.batch_id)
    print('    Pad Index       :', info.source_info.pad_index)
    print('    Frame Num       :', info.source_info.frame_num)
    print('    Samples/Frame   :', info.source_info.num_samples_per_frame)
    print('    Sample Rate     :', info.source_info.sample_rate)
    print('    Channels        :', info.source_info.num_channels)
    print('    Infer Done      :', info.source_info.inference_done)

    print('  Sound Data       : ------------------------')
    print('    Class Id        :', info.sound_info.class_id)
    print('    Label           :', info.sound_info.label)
    print('    Classifiers     :', info.sound_info.classifierLabels)
    print('    Infer Conf      :', info.sound_info.inference_confidence)
        
    print('  Trigger Criteria  : ------------------------')
    print('    Source Id       :', hex(info.criteria_info.source_id))
    print('    Class Id        :', info.criteria_info.class_id)
    print('    Min Infer Conf  :', info.criteria_info.min_inference_confidence)
    print('    Max Infer Conf  :', info.criteria_info.max_inference_confidence)
    print('    Interval        :', info.criteria_info.interval)
    print('')
    

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_READY:
        dsl_pipeline_dump_to_dot('pipeline', "state-ready")
    elif new_state == DSL_STATE_PLAYING:
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
        # retval = dsl_source_uri_new('uri-source', uri_http, False, False, 0)
        # if retval != DSL_RETURN_SUCCESS:
        #     break
            
        # retval = dsl_component_media_type_set('uri-source', 
        #     DSL_MEDIA_TYPE_AUDIO_VIDEO)
        # if retval != DSL_RETURN_SUCCESS:
        #     break

        retval = dsl_source_alsa_new('alsa-source', 'default')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_infer_aie_primary_new('paie', 
            primary_infer_config_file, primary_model_engine_file,
            frame_size, hop_size, transform)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_sde_action_monitor_new('occurrence-monitor',
            client_monitor = sde_occurrence_monitor,
            client_data = None)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_sde_trigger_occurrence_new('sde-occurrence-trigger', 
            DSL_SDE_ANY_SOURCE, DSL_SDE_ANY_CLASS, DSL_SDE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_sde_trigger_action_add('sde-occurrence-trigger',
            'occurrence-monitor')
        if retval != DSL_RETURN_SUCCESS:
            break
                   
        retval = dsl_pph_sde_new('sde-pph')
        if retval != DSL_RETURN_SUCCESS:
            break
                   
        retval = dsl_pph_sde_trigger_add('sde-pph', 'sde-occurrence-trigger')
        if retval != DSL_RETURN_SUCCESS:
            break
                   
        retval = dsl_infer_pph_add('paie', 'sde-pph', DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break
                   
        retval = dsl_sink_fake_new('fake-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_component_media_type_set('fake-sink', 
            DSL_MEDIA_TYPE_AUDIO_ONLY)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_sink_sync_enabled_set('fake-sink', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets with reduced dimensions
        retval = dsl_sink_window_egl_new('egl-sink', 0, 0, 1280, 720)
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
            ['alsa-source', 'paie', 'fake-sink', None])
#            ['uri-source', 'paie', 'fake-sink', 'egl-sink', None])
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
