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

# ------------------------------------------------------------------------------------
# This example demonstrates the use of an ODE Monitor Action -- added to an 
# ODE Occurrence Trigger with the below criteria -- to monitor all 
# ODE Occurrences
#   - class id            = PGIE_CLASS_ID_PERSON
#   - inference-done-only = TRUE
#   - minimum confidience = PERSON_MIN_CONFIDENCE
#   - minimum width       = PERSON_MIN_WIDTH
#   - minimum height      = PERSON_MIN_HEIGHT


#!/usr/bin/env python

import sys
from dsl import *

uri_h265 = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"

# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
tracker_config_file = '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

WINDOW_WIDTH = DSL_DEFAULT_STREAMMUX_WIDTH // 4
WINDOW_HEIGHT = DSL_DEFAULT_STREAMMUX_HEIGHT // 4

# Minimum Inference confidence level to Trigger ODE Occurrence
PERSON_MIN_CONFIDENCE = 0.4 # 40%

PERSON_MIN_WIDTH = 120
PERSON_MIN_HEIGHT = 320

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'Q' or key_string == '' or key_string == '':
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
# Callback function for the ODE Monitor Action - illustrates how to
# dereference the ODE 'info_ptr' and access the data fields.
# Note: you would normally use the ODE Print Action to print the info
# to the console window if that is the only purpose of the Action.
## 
def ode_occurrence_monitor(info_ptr, client_data):
    info = info_ptr.contents
    print('Trigger Name        :', info.trigger_name)
    print('  Unique Id         :', info.unique_ode_id)
    print('  NTP Timestamp     :', info.ntp_timestamp)
    print('  Source Data       : ------------------------')
    print('    Id              :', info.source_info.source_id)
    print('    Batch Id        :', info.source_info.batch_id)
    print('    Pad Index       :', info.source_info.pad_index)
    print('    Frame Num       :', info.source_info.frame_num)
    print('    Frame Width     :', info.source_info.frame_width)
    print('    Frame Height    :', info.source_info.frame_height)
    print('    Infer Done      :', info.source_info.inference_done)

    if info.is_object_occurrence:
        print('  Object Data       : ------------------------')
        print('    Class Id        :', info.object_info.class_id)
        print('    Infer Comp Id   :', info.object_info.inference_component_id)
        print('    Tracking Id     :', info.object_info.tracking_id)
        print('    Label           :', info.object_info.label)
        print('    Persistence     :', info.object_info.persistence)
        print('    Direction       :', info.object_info.direction)
        print('    Infer Conf      :', info.object_info.inference_confidence)
        print('    Track Conf      :', info.object_info.tracker_confidence)
        print('    Left            :', info.object_info.left)
        print('    Top             :', info.object_info.top)
        print('    Width           :', info.object_info.width)
        print('    Height          :', info.object_info.height)
        
    else:
        print('  Accumulative Data : ------------------------')
        print('    Occurrences     :', info.accumulative_info.occurrences_total)
        print('    Occurrences In  :', info.accumulative_info.occurrences_total)
        print('    Occurrences Out :', info.accumulative_info.occurrences_total)

    print('  Trigger Criteria  : ------------------------')
    print('    Class Id        :', info.criteria_info.class_id)
    print('    Infer Comp Id   :', info.criteria_info.inference_component_id)
    print('    Min Infer Conf  :', info.criteria_info.min_inference_confidence)
    print('    Min Track Conf  :', info.criteria_info.min_tracker_confidence)
    print('    Infer Done Only :', info.criteria_info.inference_done_only)
    print('    Min Width       :', info.criteria_info.min_width)
    print('    Min Height      :', info.criteria_info.min_height)
    print('    Max Width       :', info.criteria_info.max_width)
    print('    Max Height      :', info.criteria_info.max_height)
    print('    Interval        :', info.criteria_info.interval)
    print('')
    

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
        
        #-----------------------------------------------------------------------------
        # First, we want to remove the Object labels and bounding boxes. This is
        # be adding Format-Label and Format-BBox actions to an Occurrence Trigger.

        # Create a Format Label Action to remove the Object Label from view
        # Note: the label can be disabled with the OSD API as well, however
        # that will disable all text/labels, not just object labels. 
        retval = dsl_ode_action_label_format_new('remove-label', 
            font=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Create a Format Bounding Box Action to remove the box border from view
        retval = dsl_ode_action_bbox_format_new('remove-bbox', border_width=0,
            border_color=None, has_bg_color=False, bg_color=None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create an Any-Class Occurrence Trigger for our remove Actions
        retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger', 
            source = DSL_ODE_ANY_SOURCE, 
            class_id = DSL_ODE_ANY_CLASS, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger', 
            actions=['remove-label', 'remove-bbox', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        #```````````````````````````````````````````````````````````````````````````````````
        # Next, create an Occurrence Trigger to filter on People - defined with
        # a minimuim confidence level to eleminate most false positives

        # New Occurrence Trigger, filtering on PERSON class_id,
        retval = dsl_ode_trigger_occurrence_new('person-occurrence-trigger', 
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON,
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set a minimum confidence level to avoid false positives.
        retval = dsl_ode_trigger_confidence_min_set('person-occurrence-trigger',
            min_confidence = PERSON_MIN_CONFIDENCE)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Set the inference done only filter. 
        retval = dsl_ode_trigger_infer_done_only_set('person-occurrence-trigger',
            True)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Set minimum bounding box dimensions to trigger.
        retval = dsl_ode_trigger_dimensions_min_set('person-occurrence-trigger',
            min_width = PERSON_MIN_WIDTH, min_height = PERSON_MIN_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_ode_action_monitor_new('person-occurrence-monitor',
            client_monitor = ode_occurrence_monitor,
            client_data = None)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the ODE Heat-Mapper to the Person Occurrence Trigger.
        retval = dsl_ode_trigger_action_add('person-occurrence-trigger', 
            action='person-occurrence-monitor')
        if retval != DSL_RETURN_SUCCESS:
            break

        #`````````````````````````````````````````````````````````````````````````````
        # New ODE Handler to handle all ODE Triggers with their Areas and Actions
        retval = dsl_pph_ode_new('ode-handler')
        if retval != DSL_RETURN_SUCCESS:
            break

        
        # Add the two Triggers to the ODE PPH to be invoked on every frame. 
        retval = dsl_pph_ode_trigger_add_many('ode-handler', 
            triggers=['every-occurrence-trigger', 'person-occurrence-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break
        
        
        ############################################################################################
        #
        # Create the remaining Pipeline components
        
        # New URI File Source using the filespec defined above
        retval = dsl_source_file_new('uri-source', uri_h265, True)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 3)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New KTL Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new('iou-tracker', tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

         # Add our ODE Pad Probe Handler to the Sink pad of the OSD
        retval = dsl_osd_pph_add('on-screen-display', handler='ode-handler', pad=DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['uri-source', 'primary-gie', 'iou-tracker', 'on-screen-display', 'window-sink', None])
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
