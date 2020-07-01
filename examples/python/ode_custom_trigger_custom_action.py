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
sys.path.insert(0, "../../")
from dsl import *
import pyds

uri_file = "../../test/streams/sample_1080p_h264.mp4"

# Filespecs for the Primary GIE and IOU Trcaker
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b4_gpu0_fp16.engine'
tracker_config_file = '../../test/configs/iou_config.txt'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '':
        dsl_main_loop_quit()
 
## 
# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_main_loop_quit()

## 
# Function to be called on End-of-Stream (EOS) event
## 
def eos_event_listener(client_data):
    print('Pipeline EOS event')
    dsl_main_loop_quit()

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")
        
## 
# Function called by a Custom ODE Trigger to check for an ODE occurrence 
## 
def check_for_occurrence(buffer, frame_data, object_data, client_data):
    
    # increment our client trigger count Note... this could just as easily
    # be done by passing the count via the client_data
    global trigger_count
    trigger_count += 1
    
    # a contrived example of some type of occurrence criteria
    if (trigger_count % 100 == 0):
        
        # Note: not sure this is possible. Need to talk with Nvida
        # may need to cast and traverse the buffer pointer... less than ideal
        
        # cast the frame data to a pyds.NvDsFrameMeta
 #       frame_meta = pyds.glist_get_nvds_frame_meta(frame_data)

        # cast the object data to a pyds.NvDsObjectMeta
        #object_meta = pyds.glist_get_nvds_object_meta(object_data)
        
#        print("occurrence triggered: object-label=", object_meta.obj_label)
        print("occurrence triggered: trigger_count=", trigger_count)
        
        # return True to invoke all actions
        return True
    
    # return false, actions will NOT be invoked for this Trigger
    return False
    

## 
# Function called by a Callback ODE Action to handle an ODE occurrence 
## 
#def handle_occurrence(ode_id, trigger):
def handle_occurrence(ode_id, trigger, buffer, frame_data, objec_data, client_data):
    
    # Note: not sure this is possible. Need to talk with Nvida
    # may need to cast and traverse the buffer pointer... less than ideal

    # cast the frame data to a pyds.NvDsFrameMeta
#    frame_meta = pyds.glist_get_nvds_frame_meta(frame_data)

    # cast the object data to a pyds.NvDsObjectMeta
#    object_meta = pyds.glist_get_nvds_object_meta(object_data)
    
    print("Unique IDE Id=", ode_id)

##
# Trigger count, to track the number of times the Custom Trigger
# Calls the client "check-for-occurrence"
trigger_count = 0

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # This example creates two ODE triggers; one built-in Occurrence Trigger to call a Custom Action, and 
        # one Custom Trigger to call two built-in Actions, Fill and Print.
        
        # Create a new Custom Callback action that will print out the details of the ODE occurrence
        retval = dsl_ode_action_callback_new('callback-action', client_handler=handle_occurrence, client_data=None)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create a new Fill Action that will fill the Object's rectangle with a shade of red to indicate occurrence
        retval = dsl_ode_action_fill_object_new('red-fill-action', red=1.0, green=0.0, blue=0.0, alpha = 0.20)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a new Print Action to print out the ODE Details to the console winodw
        # The action will be triggered on every occurrence of a bicycle that meets the minimum criteria
        retval = dsl_ode_action_print_new('shared-print-action')
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Custom Trigger, filtering on the Vehical Class Id, with no limit on the number of occurrences
        retval = dsl_ode_trigger_custom_new('custom-trigger', class_id=PGIE_CLASS_ID_VEHICLE, limit=0,
            client_checker=check_for_occurrence, client_data=trigger_count)
        if retval != DSL_RETURN_SUCCESS:
            break
        # Add the red-fill and shared-print actions to be called when the "check-for-occurrence" returns true
        retval = dsl_ode_trigger_action_add_many('custom-trigger', actions=[
            'red-fill-action', 'shared-print-action', None])
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New Occurrence Trigger, filtering on the Bicycle ClassId, with a limit of one occurrence
        retval = dsl_ode_trigger_occurrence_new('bicycle-occurrence', class_id=PGIE_CLASS_ID_BICYCLE, limit=1)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('bicycle-occurrence', actions=[
            'red-fill-action', 'shared-print-action', 'callback-action', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # New ODE Handler to handle all ODE Triggers and their Actions    
        retval = dsl_ode_handler_new('ode-hanlder')
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_handler_trigger_add_many('ode-hanlder', triggers=[
            'custom-trigger', 'bicycle-occurrence', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # New URI File Source using the filespec defined above
        retval = dsl_source_uri_new('uri-source', uri_file, False, 0, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New KTL Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new('iou-tracker', tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiled Display, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new('tiler', 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break
 
        # New OSD with clock enabled... .
        retval = dsl_osd_new('on-screen-display', True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker', 'tiler', 'ode-hanlder', 'on-screen-display', 'window-sink', None])
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
