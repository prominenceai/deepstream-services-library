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
# This example demostrates how to use a  Source Meter Pad Probe Handler (PPH) 
# that will measure the Pipeline's throughput for each Source - while monitoring
# the depth of every component's Queue.
#   
# The Meter PPH is added to the sink (input) pad of the Tiler before tha batched
# stream is converted into a single stream as a 2D composite of all Sources.
#
# The "meter_pph_handler" callback added to the Meter PPH will handle writing 
# the Avg Session FPS and the Avg Interval FPS measurements to the console.
# # 
# The Key-released-handler callback (below) will disable the meter when pausing 
# the Pipeline, and # re-enable measurements when the Pipeline is resumed.
#  
# Note: Session averages are reset each time the Meter is disabled and 
# then re-enabled.
#
# The callback, called once per second as defined during Meter construction,
# is also responsible for polling the components for their queue depths - i.e
# using the "dsl_component_queue_current_level_print_many" service.
#  
# Additionally, a Queue Overrun Listener is added to each of the components to
# be notified on the event of a queue-overrun.
# 
# https://github.com/prominenceai/deepstream-services-library/blob/master/docs/api-component.md#component-queue-management
#
################################################################################

#!/usr/bin/env python

import sys
sys.path.insert(0, "../../")
import time

from dsl import *

# Unique names for each of the Pipeline components
URI_SOURCE_NAME_0 = 'uri-source-0'
URI_SOURCE_NAME_1 = 'uri-source-1'
URI_SOURCE_NAME_2 = 'uri-source-2'
URI_SOURCE_NAME_3 = 'uri-source-3'
URI_SOURCE_NAME_4 = 'uri-source-4'
URI_SOURCE_NAME_5 = 'uri-source-5'
URI_SOURCE_NAME_6 = 'uri-source-6'
URI_SOURCE_NAME_7 = 'uri-source-7'
PRIMARY_GIE_NAME = 'primary-gie'
IOU_TRACKER_NAME = 'iou-tracker'
TILER_NAME = 'tiler'
OSD_NAME = 'on-screen-display'
WINDOW_SINK_NAME = 'window-sink'

# Null terminated list of all Pipeline Component names - will be used when 
# calling Queue services collectively

COMPONENTS = [
    URI_SOURCE_NAME_0, URI_SOURCE_NAME_1, URI_SOURCE_NAME_2, URI_SOURCE_NAME_3, 
    URI_SOURCE_NAME_4, URI_SOURCE_NAME_5, URI_SOURCE_NAME_6, URI_SOURCE_NAME_7, 
    PRIMARY_GIE_NAME, IOU_TRACKER_NAME, TILER_NAME, OSD_NAME, WINDOW_SINK_NAME, None]

# Test URI used for all sources
uri = '/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4'

# Filespecs (Jetson and dGPU) for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt'
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine'

# Filespec for the IOU Tracker config file
iou_tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'

TILER_WIDTH = DSL_1K_HD_WIDTH
TILER_HEIGHT = 720

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
# Function to be called if any of the Pipeline Component Queues becomes full.
# A buffer is full if the total amount of data inside it (buffers, bytes, 
# or time) is higher than the max-size values set for each unit. Max-size values
# can be set by calling dsl_component_queue_max_size_set.
# ## 
def queue_overrun_listener(name, client_data):
    print('WARNING Queue Overrun occurred for component = ', name)

##
# To be used as client_data with our Source Meter PPH, and passed to our 
# client_calback
##
class ReportData:
  def __init__(self):
    self.m_report_count = 0
    
## 
# Meter Sink client callback funtion
## 
def meter_pph_handler(session_avgs, interval_avgs, source_count, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    report_data = cast(client_data, POINTER(py_object)).contents.value

    # Print header
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
    print()

    # Increment reporting count
    report_data.m_report_count += 1

    # Print (or log) out the current Component Queue levels
    dsl_component_queue_current_level_print_many(COMPONENTS,
        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
    # dsl_component_queue_current_level_log_many(COMPONENTS,
    #     DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS)
    
    return True
        

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
    
        report_data = ReportData()
        
        # New Source Meter Pad Probe handler to call the meter_pph_handler with an
        # interval of 1 second.
        retval = dsl_pph_meter_new('meter-pph', interval=1, 
            client_handler=meter_pph_handler, client_data=report_data)
        if retval != DSL_RETURN_SUCCESS:
            break
        #
        # Create the remaining Pipeline components
        # ... starting with eight URI File Sources
        
        retval = dsl_source_uri_new(URI_SOURCE_NAME_0, uri, False, False, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        dsl_source_uri_new(URI_SOURCE_NAME_1, uri, False, False, 0)
        dsl_source_uri_new(URI_SOURCE_NAME_2, uri, False, False, 0)
        dsl_source_uri_new(URI_SOURCE_NAME_3, uri, False, False, 0)
        dsl_source_uri_new(URI_SOURCE_NAME_4, uri, False, False, 0)
        dsl_source_uri_new(URI_SOURCE_NAME_5, uri, False, False, 0)
        dsl_source_uri_new(URI_SOURCE_NAME_6, uri, False, False, 0)
        dsl_source_uri_new(URI_SOURCE_NAME_7, uri, False, False, 0)

        # New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new(PRIMARY_GIE_NAME, 
            primary_infer_config_file, primary_model_engine_file, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new(IOU_TRACKER_NAME, iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Tiler, setting width and height, use default cols/rows set by source count
        retval = dsl_tiler_new(TILER_NAME, TILER_WIDTH, TILER_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # Important: add the Meter to the Sink pad of the Tiler, while the stream 
        # is still batched and measurements can be made for all sources. Adding 
        # downstream will measure the combined, tiled stream.
        retval = dsl_tiler_pph_add(TILER_NAME, 'meter-pph', DSL_PAD_SINK)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        # New OSD with text, clock and bbox display all enabled. 
        retval = dsl_osd_new(OSD_NAME, 
            text_enabled=True, clock_enabled=True, 
            bbox_enabled=True, mask_enabled=False)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, 0 x/y offsets and same dimensions as Tiled Display
        retval = dsl_sink_window_egl_new(WINDOW_SINK_NAME,
             0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the XWindow event handler functions defined above to the Window Sink
        retval = dsl_sink_window_key_event_handler_add(WINDOW_SINK_NAME, 
            xwindow_key_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_sink_window_delete_event_handler_add(WINDOW_SINK_NAME, 
            xwindow_delete_event_handler, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        #----------------------------------------------------------------------------------------------------
        # Pipeline assembly
        #
        retval = dsl_pipeline_new_component_add_many('pipeline', 
            COMPONENTS)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the queue-overrun callback funtion to each of the Pipeline Components
        retval = dsl_component_queue_overrun_listener_add_many(COMPONENTS,
            queue_overrun_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        ## Add the listener callback functions defined above
        retval = dsl_pipeline_state_change_listener_add('pipeline', 
            state_change_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_pipeline_eos_listener_add('pipeline', 
            eos_event_listener, None)
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
