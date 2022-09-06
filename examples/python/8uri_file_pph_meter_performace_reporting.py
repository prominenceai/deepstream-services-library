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
sys.path.insert(0, "../../")
import time

from dsl import *

# Test URI used for all sources
uri = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4'
# Filespecs for the Primary GIE
inferConfigFile = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt'
modelEngineFile = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine'
tracker_config_file = '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

TILER_WIDTH = DSL_STREAMMUX_DEFAULT_WIDTH
TILER_HEIGHT = 720

WINDOW_WIDTH = TILER_WIDTH
WINDOW_HEIGHT = TILER_HEIGHT

## 
# Function to be called on XWindow KeyRelease event
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        
        # if we're able to pause the Pipeline (i.e. it's not already paused)
        if dsl_pipeline_pause('pipeline') == DSL_RETURN_SUCCESS:
        
            # then disable the sink meter from reporting metrics
            dsl_sink_meter_enabled_set('meter-sink', False)
            
    elif key_string.upper() == 'R':
    
        # if we're able to Resume the Pipeline 
        if dsl_pipeline_play('pipeline') == DSL_RETURN_SUCCESS:

            # then re-enable the sink meter to start reporting metrics again
            dsl_sink_meter_enabled_set('meter-sink', True)

    elif key_string.upper() == 'Q' or key_string == '' or key_string == '':
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
# To be used as client_data with our Meter Sink, and passed to our client_calback
##
class ReportData:
  def __init__(self, header_interval):
    self.m_report_count = 0
    self.m_header_interval = header_interval
    
## 
# Meter Sink client callback funtion
## 
def meter_sink_handler(session_avgs, interval_avgs, source_count, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    report_data = cast(client_data, POINTER(py_object)).contents.value

    # Print header on interval
    if (report_data.m_report_count % report_data.m_header_interval == 0):
        header = ""
        for source in range(source_count):
            subheader = f"FPS {source} (AVG)"
            header += "{:<15}".format(subheader)
        print()
        print(header)

    # Print FPS counters
    counters = ""
    for source in range(source_count):
        counter = "{:.2f} ({:.2f})".format(interval_avgs[source], session_avgs[source])
        counters += "{:<15}".format(counter)
    print(counters)

    # Increment reporting count
    report_data.m_report_count += 1
    
    return True
        

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        #
        # New Meter Pad Probe Handler that will measure the Pipeline's throughput. Our client callback will handle writing the 
        # Avg Session FPS and Avg Interval FPS measurements to the console. The Key-released-handler callback (above)
        # will disable the meter when pausing the Pipeline, and re-enable measurements when the Pipeline is resumed
        # Note: Session averages are reset each time a Meter is disabled and then re-enable.

        report_data = ReportData(header_interval=12)
        
        retval = dsl_pph_meter_new('meter-pph', interval=1, client_handler=meter_sink_handler, client_data=report_data)
        if retval != DSL_RETURN_SUCCESS:
            break
        #
        # Create the remaining Pipeline components
        # ... starting with eight URI File Sources
        
        retval = dsl_source_uri_new('Camera 1', uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        dsl_source_uri_new('Camera 2', uri, False, False, 0)
        dsl_source_uri_new('Camera 3', uri, False, False, 0)
        dsl_source_uri_new('Camera 4', uri, False, False, 0)
        dsl_source_uri_new('Camera 5', uri, False, False, 0)
        dsl_source_uri_new('Camera 6', uri, False, False, 0)
        dsl_source_uri_new('Camera 7', uri, False, False, 0)
        dsl_source_uri_new('Camera 8', uri, False, False, 0)

        # New Primary GIE using the filespecs above, with interval and Id
#        retval = dsl_infer_gie_primary_new('primary-gie', inferConfigFile, modelEngineFile, interval=4)
        retval = dsl_infer_gie_primary_new('primary-gie', inferConfigFile, None, interval=4)
        if retval != DSL_RETURN_SUCCESS:
            break

        #----------------------------------------------------------------------------------------------------
        # Create one of each Tracker Types, KTL and IOU, to test with each. We will only add one
        # at a time, but it's easier to create both and just update the Pipeline assembly below as needed.

        # New KTL Tracker, setting max width and height of input frame
        retval = dsl_tracker_ktl_new('ktl-tracker', 480, 288)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting max width and height of input frame
        retval = dsl_tracker_iou_new('iou-tracker', tracker_config_file, 480, 288)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiler, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new('tiler', TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Important: add the Meter to the Sink pad of the Tiler, while the stream is still batched and
        # measurements can be made for all sources. Adding downstream will measure the combined, tiled stream
        retval = dsl_tiler_pph_add('tiler', 'meter-pph', DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new('on-screen-display', 
            text_enabled=True, clock_enabled=True, bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        #----------------------------------------------------------------------------------------------------
        # Create one of each Render Sink Types, Overlay and Window, to test with each. We will only add one
        # at a time, but it's easier to create both and just update the Pipeline assembly below as needed.
        
        # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_overlay_new('overlay-sink', 0, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_new('window-sink', 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        #----------------------------------------------------------------------------------------------------
        # Pipeline assembly
        #
        # New Pipeline (trunk) with our Sources, Tracker, and Pre-Tiler  as last component
        # Note: *** change 'iou-tracker' to 'ktl-tracker' to try both. KTL => higher CPU load 
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            ['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4', 'Camera 5', 'Camera 6',  'Camera 7', 'Camera 8',
            'primary-gie', 'iou-tracker', 'tiler', 'on-screen-display', 'overlay-sink', None])
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
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_pipeline_delete_all()
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
