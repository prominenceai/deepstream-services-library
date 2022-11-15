################################################################################
# The MIT License
#
# Copyright (c) 2021, Prominence AI, Inc.
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
from dsl import *

##########################################################################33####
# IMPORTANT! it is STRONGLY advised that you create a new, free Gmail account -- 
# that is seperate/unlinked from all your other email accounts -- strictly for 
# the purpose of sending ODE Event data uploaded from DSL.  Then, add your 
# Personal email address as a "To" address to receive the emails.
#
# Gmail considers regular email programs (i.e Outlook, etc.) and non-registered 
# third-party apps to be "less secure". The email account used for sending email 
# must have the "Allow less secure apps" option turned on. Once you've created 
# this new account, you can go to the account settings and enable Less secure 
# app access. see https://myaccount.google.com/lesssecureapps
#
# CAUTION - Do not check sripts into your repo with valid credentials
#
#######################################################################
user_name = 'my.smtps.server'
password = 'my-server-pw'
server_url = 'smtps://smtp.gmail.com:465'

from_name = ''
from_address = 'my.smtps.server'
to_name = 'Joe Bloe'
to_address = 'joe.blow@gmail.com'

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
TILER_HEIGHT = DSL_STREAMMUX_DEFAULT_HEIGHT
WINDOW_WIDTH = TILER_WIDTH
WINDOW_HEIGHT = TILER_HEIGHT

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

## 
# Function to be called on End-of-Stream (EOS) event
## 
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
# Function to be called on recording start and complete
## 
def recording_event_listener(session_info_ptr, client_data):
    print(' ***  Recording Event  *** ')
    
    session_info = session_info_ptr.contents

    print('session_id: ', session_info.session_id)
    
    # If we're starting a new recording for this source
    if session_info.recording_event == DSL_RECORDING_EVENT_START:
        print('event:      ', 'DSL_RECORDING_EVENT_START')

        # enable the always trigger showing the metadata for "recording in session" 
        retval = dsl_ode_trigger_enabled_set('rec-on-trigger', enabled=True)
        if (retval != DSL_RETURN_SUCCESS):
            print('Enable trigger failed with error: ', dsl_return_value_to_string(retval))

    # Else, the recording session has ended for this source
    else:
        print('event:      ', 'DSL_RECORDING_EVENT_END')
        print('filename:   ', session_info.filename)
        print('dirpath:    ', session_info.dirpath)
        print('duration:   ', session_info.duration)
        print('container:  ', session_info.container_type)
        print('width:      ', session_info.width)
        print('height:     ', session_info.height)

        # disable the always trigger showing the metadata for "recording in session" 
        retval = dsl_ode_trigger_enabled_set('rec-on-trigger', enabled=False)
        if (retval != DSL_RETURN_SUCCESS):
            print('Disable always trigger failed with error: ', dsl_return_value_to_string(retval))
    
        # re-enable the one-shot trigger for the next "New Instance" of a person
        retval = dsl_ode_trigger_reset('bicycle-instance-trigger')    
        if (retval != DSL_RETURN_SUCCESS):
            print('Failed to reset instance trigger with error:', dsl_return_value_to_string(retval))
    
    return None
    
## 
# Function to be called on Player termination event
## 
def player_termination_event_listener(client_data):
    print(' ***  Video Playback Complete  *** ')

    # reset the Player to close its rendering surface
    dsl_player_render_reset('video-player')
    
def setup_smpt_mail():

    global server_url, user_name, password, from_name, from_address, \
        to_name, to_address
        
    retval = dsl_mailer_new('mailer')
    if retval != DSL_RETURN_SUCCESS:
        return retval
    
    retval = dsl_mailer_server_url_set('mailer', server_url)
    if retval != DSL_RETURN_SUCCESS:
        return retval
    retval = dsl_mailer_credentials_set('mailer' , user_name, password)
    if retval != DSL_RETURN_SUCCESS:
        return retval
    retval = dsl_mailer_address_from_set('mailer', from_name, from_address)
    if retval != DSL_RETURN_SUCCESS:
        return retval
    retval = dsl_mailer_address_to_add('mailer', to_name, to_address)
    if retval != DSL_RETURN_SUCCESS:
        return retval
        
    # (optional) queue a test message to be sent out when the main_loop starts
    return dsl_mailer_test_message_send('mailer')    

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # This example is used to demonstrate the use of a First Occurrence Trigger and a Start Record Action
        # to control a Record Sink.  A callback function, called on completion of the recording session, will
        # reset the Trigger allowing a new session to be started on next occurrence.
        # Addional actions are added to "Capture" the frame to an image-file and "Fill" the frame red as a visual marker.
        # A Video Render Player is added to the Capture Action to playback the video on record complete.
        # A Mailer is added to Capture Action to email information about the saved recording; name, location, etc.
        

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Setup the SMTP Server URL, Credentials, and From/To addresss
        retval = setup_smpt_mail()
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New Record-Sink that will buffer encoded video while waiting for the ODE trigger/action, defined below, 
        # to start a new session on first occurrence. The default 'cache-size' and 'duration' are defined in
        # Setting the bit rate to 12 Mbps for 1080p
        retval = dsl_sink_record_new('record-sink', outdir="./", codec=DSL_CODEC_H265, container=DSL_CONTAINER_MKV, 
            bitrate=2000000, interval=0, client_listener=recording_event_listener)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Update the cache size to 5 seconds .
        retval = dsl_sink_record_cache_size_set('record-sink', 5)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create the Video Render Player with a NULL file_path to be updated by the Record Sink
        dsl_player_render_video_new(
            name = 'video-player',
            file_path = None,
            render_type = DSL_RENDER_TYPE_OVERLAY,
            offset_x = 500, 
            offset_y = 20, 
            zoom = 50,
            repeat_enabled = False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Termination listener callback to the Player 
        retval = dsl_player_termination_event_listener_add('video-player',
            client_listener=player_termination_event_listener, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            return

        # Add the Player to the Recorder Sink. The Sink will add/queue
        # the file_path to each video recording created. 
        retval = dsl_sink_record_video_player_add('record-sink', 
            player='video-player')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the Mailer to Recorder Sink. The Sink will use the Mailer to email information
        # -- file location, size, etc. -- on the completion each recorded video.
        retval = dsl_sink_record_mailer_add('record-sink', 
            mailer = 'mailer',
            subject = 'ATTENTION: Bycicle Occurence!')

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Create new RGBA color types
        retval = dsl_display_type_rgba_color_custom_new('opaque-red', red=1.0, blue=0.5, green=0.5, alpha=0.7)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('full-red', red=1.0, blue=0.0, green=0.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('full-white', red=1.0, blue=1.0, green=1.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_color_custom_new('opaque-black', red=0.0, blue=0.0, green=0.0, alpha=0.8)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_display_type_rgba_font_new('impact-20-white', font='impact', size=20, color='full-white')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new Text type object that will be used to show the recording in progress
        retval = dsl_display_type_rgba_text_new('rec-text', 'REC    ', x_offset=10, y_offset=30, 
            font='impact-20-white', has_bg_color=True, bg_color='opaque-black')
        if retval != DSL_RETURN_SUCCESS:
            break
        # A new RGBA Circle to be used to simulate a red LED light for the recording in progress.
        retval = dsl_display_type_rgba_circle_new('red-led', x_center=94, y_center=52, radius=8, 
            color='full-red', has_bg_color=True, bg_color='full-red')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new Action to display the "recording in-progress" text
        retval = dsl_ode_action_display_meta_add_many_new('add-rec-on', display_types=
            ['rec-text', 'red-led', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create an Always trigger to add the "REC" text meta-data to every frame when enabled.
        retval = dsl_ode_trigger_always_new('rec-on-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            when = DSL_ODE_PRE_OCCURRENCE_CHECK)
        if retval != DSL_RETURN_SUCCESS:
            return retval

        # Disable the trigger. Will be re-enabled on DSL_RECORDING_EVENT_START
        # and then disabled again on DSL_RECORDING_EVENT_END
        retval = dsl_ode_trigger_enabled_set('rec-on-trigger', enabled=False) 
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the display-meta add action to the Always trigger
        retval = dsl_ode_trigger_action_add('rec-on-trigger', action='add-rec-on')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Fill-Area Action as a visual indicator to identify 
        #the frame that triggered the recording
        retval = dsl_ode_action_fill_frame_new('red-flash-action', 'opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new Capture Action to capture the full-frame to jpeg image, 
        # and save to file. The action will be triggered on firt instance of 
        # a bicycle and will be saved to the current dir.
        retval = dsl_ode_action_capture_object_new('bicycle-capture-action', outdir="./images")
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create a new Capture Action to start a new record session
        retval = dsl_ode_action_sink_record_start_new('start-record-action', 
            record_sink='record-sink', start=5, duration=10, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````
        # Next, create the New Instance Trigger for the bicycle class. 
        # We will reset the trigger in the recording complete callback
        retval = dsl_ode_trigger_instance_new('bicycle-instance-trigger', 
            source=DSL_ODE_ANY_SOURCE, class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # set the "infer-done-only" criteria so we can capture the confidence level
        retval = dsl_ode_trigger_infer_done_only_set('bicycle-instance-trigger', True)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_ode_action_print_new('print', force_flush=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Add the actions to our Bicycle Occurence Trigger.
        retval = dsl_ode_trigger_action_add_many('bicycle-instance-trigger', actions=[
            'red-flash-action', 
            'bicycle-capture-action', 
            'start-record-action', 
            'print',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New ODE Handler for our Trigger
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers=[
            'bicycle-instance-trigger',
            'rec-on-trigger', 
            None])
        if retval != DSL_RETURN_SUCCESS:
            break
    
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        retval = dsl_source_uri_new('uri-source', uri_h265, is_live=False, 
            intra_decode=False, drop_frame_interval=0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 1
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiled Display, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
 
        # add our ODE Pad Probe Handle to the Sink Pad of the Tiler
        retval = dsl_tiler_pph_add('tiler', 'ode-handler', DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break
 
        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline - except for our second source and overlay sink 
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker', 'tiler', 
            'on-screen-display', 'window-sink', 'record-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the XWindow event handler functions defined above
        retval = dsl_pipeline_xwindow_key_event_handler_add("pipeline", xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_xwindow_delete_event_handler_add("pipeline", xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_eos_listener_add('pipeline', eos_event_listener, None)
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

    # Cleanup all DSL/GST resources
    dsl_delete_all()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
    