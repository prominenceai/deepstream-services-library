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

# 4K example image is located under "/deepstream-services-library/test/streams"
image_file = "../../test/streams/4K-image.jpg"

# Preprocessor config file is located under "/deepstream-services-library/test/configs"
preproc_config_file = \
    '../../test/configs/config_preprocess_4k_input_slicing.txt'
    
# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt'

# IMPORTANT! ensure that the model-engine was generated with the config from the Preprocessing example
#  - apps/sample_apps/deepstream-preprocess-test/config_infer.txt
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b3_gpu0_fp16.engine'
    
# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

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

#    dsl_pipeline_stop('pipeline')
#    dsl_main_loop_quit()

## 
# Function to be called on every change of Pipeline state
## 
def state_change_listener(old_state, new_state, client_data):
    print('previous state = ', old_state, ', new state = ', new_state)
    if new_state == DSL_STATE_PLAYING:
        dsl_pipeline_dump_to_dot('pipeline', "state-playing")

## 
# Function to be called on Object Capture (and file-save) complete
## 
def capture_complete_listener(capture_info_ptr, client_data):
    print(' ***  Object Capture Complete  *** ')
    
    capture_info = capture_info_ptr.contents
    print('capture_id: ', capture_info.capture_id)
    print('filename:   ', capture_info.filename)
    print('dirpath:    ', capture_info.dirpath)
    print('width:      ', capture_info.width)
    print('height:     ', capture_info.height)

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # --------------------------------------------------------------------------------
        # Step 1: We build the (final stage) Inference Pipeline with an Image-Source,
        # Preprocessor, Primary GIE, IOU Tracker, On-Screen Display, and Window Sink.

        # Create a new streaming image source to stream the UHD file at 10 frames/sec
        retval = dsl_source_image_stream_new('image-source', file_path=image_file,
            is_live=False, fps_n=10, fps_d=1, timeout=0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Preprocessor component using the config filespec defined above.
        retval = dsl_preproc_new('preprocessor', preproc_config_file)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # New Primary GIE using the filespecs above with interval = 0
        retval = dsl_infer_gie_primary_new('primary-gie', 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # **** IMPORTANT! for best performace we explicity set the GIE's batch-size 
        # to the number of ROI's defined in the Preprocessor configuraton file.
        # Otherwise, the Pipeline will set the GIE's batch size to the number of sources.
        retval = dsl_infer_batch_size_set('primary-gie', 3)
        if retval != DSL_RETURN_SUCCESS:
            break

        # **** IMPORTANT! we must set the input-meta-tensor setting to true when
        # using the preprocessor, otherwise the GIE will use its own preprocessor.
        retval = dsl_infer_gie_tensor_meta_settings_set('primary-gie',
            input_enabled=True, output_enabled=False);

        # New IOU Tracker, setting operational width and height
        retval = dsl_tracker_new('iou-tracker', iou_tracker_config_file, 640, 368)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New OSD with text and bbox display enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, 
            width=DSL_STREAMMUX_4K_UHD_WIDTH, height=DSL_STREAMMUX_4K_UHD_WIDTH)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add all the components to our pipeline
        retval = dsl_pipeline_new_component_add_many('pipeline', components=[
            'image-source', 'preprocessor', 'primary-gie', 'iou-tracker',
            'on-screen-display', 'window-sink', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Update the Pipeline's streammuxer dimensions 
        retval = dsl_pipeline_streammux_dimensions_set('pipeline', 
            width=DSL_STREAMMUX_4K_UHD_WIDTH, height= DSL_STREAMMUX_4K_UHD_HEIGHT)
        
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

        # --------------------------------------------------------------------------------
        # Step 2:

        # New Occurrence Trigger, filtering on PERSON class_id, and with no limit on
        # the number of occurrences. The Trigger will be added to an ODE Handler
        # that is added to the source pad (output) of the On-Screen Display. 
        retval = dsl_ode_trigger_occurrence_new('every-person-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Set the Trigger's frame-limit to 1. We only need/want to print out the
        # ODE Data for all objects detected in one frame. Image source streams
        # the same Image at a set framerate to constant tracking and viewing.
        retval = dsl_ode_trigger_limit_frame_set('every-person-trigger',
            DSL_ODE_TRIGGER_LIMIT_ONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Print Action to print each details of each ODE to the console.
        retval = dsl_ode_action_print_new('print-action', force_flush=False)        
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Print Action to the Every Person Trigger
        retval = dsl_ode_trigger_action_add('every-person-trigger', 
            action='print-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 3:

        # New Occurrence Trigger, filtering on PERSON class_id, with a limit of 1
        # The Trigger will be added to an ODE Handler that is added to the source 
        # pad (output) of the On-Screen Display. 
        retval = dsl_ode_trigger_occurrence_new('first-person-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_ONE)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Create a new Capture Action to capture the Frame to jpeg image, and save 
        # to file. The Action will be called once by the Trigger (trigger limit of 1).
        # Note: this call will fail if the output directory "outdir" does not exist.
        retval = dsl_ode_action_capture_frame_new('frame-capture-action',
            outdir = "./",
            annotate = False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Capture Action to the First Person Trigger
        retval = dsl_ode_trigger_action_add('first-person-trigger', 
            action='frame-capture-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 4:

        # New ODE Handler to handle the Every-Person and First Person Triggers
        retval = dsl_pph_ode_new('ode-handler-on-osd-src')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Add the Every-Person and First-Person Triggers to the ODE Handler
        retval = dsl_pph_ode_trigger_add_many('ode-handler-on-osd-src', 
            triggers = ['every-person-trigger', 'first-person-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the ODE Handler to the Source pad (output) of the On-Screen Display
        retval = dsl_osd_pph_add('on-screen-display', 
            'ode-handler-on-osd-src', DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 5: Create a Remove Object Action to remove an object (remove metadata)
        # from the current frame. The Action will be used by multiple Triggers.
        retval = dsl_ode_action_object_remove_new('remove-object-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 6: Remove the false positive person-objects caused by the tree. 
        #
        # Ths is done by creating Polygon Display type that will be used to create
        # an ODE Area of Inclusion - the Polygon will cover a small portion of the
        # tree, which is too high up to be a person, where the false positives occur.
        # Any object that touches or overlaps the ODE Area will be removed. 
        
        # New custom RGBA color that will be used to create an RGBA Polygon
        retval = dsl_display_type_rgba_color_custom_new('solid-blue', 
            red=0.2, green=0.2, blue=1.0, alpha=1.0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create a list of X,Y coordinates defining the points of the Polygon.
        # Polygon can have a minimum of 3, maximum of 16 points (sides).
        # This is done imperically by looking at the object data that is
        # printed to the cosole by the Every-Person Trigger and Print Action.
        coordinates = [dsl_coordinate(3165,1520), dsl_coordinate(3245,1516), 
            dsl_coordinate(3245,1540), dsl_coordinate(3165,1540)]
            
        # Create the Polygon display type using the coordinates and RGBA color.
        retval = dsl_display_type_rgba_polygon_new('polygon1', 
            coordinates=coordinates, num_coordinates=len(coordinates), 
            border_width=2, color='solid-blue')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Create the ODE inclusion area to use as criteria for ODE occurrence
        retval = dsl_ode_area_inclusion_new('tree-area', polygon='polygon1', 
            show=True, bbox_test_point=DSL_BBOX_POINT_ANY)    
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Occurrence Trigger, filtering on PERSON class_id, and with no limit 
        # on the number of occurrences. The Trigger will be added to an ODE Handler
        # which will be added to the Sink Pad (input) of the IOU Tracker
        retval = dsl_ode_trigger_occurrence_new('every-tree-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Inclusion Area to the Trigger as occurrence criteria.
        retval = dsl_ode_trigger_area_add('every-tree-trigger', 
            area='tree-area')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the Remove Object Action to the Every-Tree Trigger
        retval = dsl_ode_trigger_action_add('every-tree-trigger', 
            action='remove-object-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 7: Remove the person-objects with low inference confidence.

        # New Occurrence Trigger, filtering on PERSON class_id, and with no limit on 
        # the number of occurrences. The Trigger will use the Remove Object Action
        # to remove each object with low inference confidence.
        retval = dsl_ode_trigger_occurrence_new('every-low-conf-person-trigger',
            source = DSL_ODE_ANY_SOURCE,
            class_id = PGIE_CLASS_ID_PERSON, 
            limit = DSL_ODE_TRIGGER_LIMIT_NONE)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Set the Max Inference confidence as criteria for ODE occurrence.
        retval = dsl_ode_trigger_infer_confidence_max_set('every-low-conf-person-trigger',
            max_confidence=0.31)

        # Add the Remove Object
        retval = dsl_ode_trigger_action_add('every-low-conf-person-trigger', 
            action='remove-object-action')
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 8: Add the Trigger for removing false positives to a new ODE Handler.
        # The Handler will then be added to the Sink Pad (input) of the IOU Tracker.

        # New ODE Handler to handle the every-person     
        retval = dsl_pph_ode_new('ode-handler-on-iou-sink-pad')
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_pph_ode_trigger_add_many('ode-handler-on-iou-sink-pad', triggers=[
            'every-low-conf-person-trigger', 'every-tree-trigger', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the ODE Handler to the Sink Pad (input) of the IOU Tracker
        retval = dsl_tracker_pph_add('iou-tracker', 
            'ode-handler-on-iou-sink-pad', DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break

        # --------------------------------------------------------------------------------
        # Step 9: Play the Inference Pipeline and join the main loop.

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
