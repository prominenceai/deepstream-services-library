################################################################################
# The MIT License
#
# Copyright (c) 2019-2021, Prominence AI, Inc.
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
        
    # (optional) queue a test message to be sent out when main_loop starts
    return dsl_mailer_test_message_send('mailer')

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # This example is used to demonstrate the use of a First Occurrence Trigger and an Email Action
        # to send an emal using SMTP service. Addional actions are added to "Capture" the frame to an 
        # image-file and "Fill" (flash) the frame red as a visual marker.

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Setup the SMTP Server URL, Credentials, and From/To addresss
        retval = setup_smpt_mail()
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Create new RGBA color types
        retval = dsl_display_type_rgba_color_custom_new('opaque-red', red=1.0, blue=0.5, green=0.5, alpha=0.7)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Action to Queue an email
        retval = dsl_ode_action_email_new('email-action', 
            mailer = 'mailer',
            subject = "Bicycle Occurrence!")
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a Fill-Area Action as a visual indicator to identify the frame that triggered the recording
        retval = dsl_ode_action_fill_frame_new('red-flash-action', 'opaque-red')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a new Capture Action to capture the full-frame to jpeg image, and save to file. 
        # The action will be triggered on firt occurrence of a bicycle and will be saved to the current dir.
        retval = dsl_ode_action_capture_frame_new('bicycle-capture-action', outdir="./", annotate=True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Next, create the Bicycle Occurrence Trigger. We will reset the trigger in the recording complete callback
        retval = dsl_ode_trigger_occurrence_new('bicycle-occurrence-trigger', 
            source=DSL_ODE_ANY_SOURCE, class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # Add the actions to our Bicycle Occurence Trigger.
        retval = dsl_ode_trigger_action_add_many('bicycle-occurrence-trigger', actions=[
            'email-action',
            'red-flash-action', 
            'bicycle-capture-action',
            None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # ````````````````````````````````````````````````````````````````````````````````````````````````````````
        # New ODE Handler for our Trigger
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pph_ode_trigger_add('ode-handler', trigger='bicycle-occurrence-trigger')
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
            'on-screen-display', 'window-sink', None])
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
    