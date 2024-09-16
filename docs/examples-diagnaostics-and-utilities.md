# Pipeline Diagnostics and Utilities
This page documents the following examples:
* [Pipeline with Source Meter Pad Probe Handler and Component Queue Management](#pipeline-with-source-meter-pad-probe-handler-and-component-queue-management)
* [Using a Simple DSL Player to test an RTSP Source Connection](#using-a-simple-dsl-player-to-test-an-rtsp-source-connection)
* [Running Inference on all MP4 files in a Folder](#running-inference-on-all-mp4-files-in-a-folder)

<br>

---

### Pipeline with Source Meter Pad Probe Handler and Component Queue Management

* [`8uri_file_pph_meter_performace_reporting.py`](/examples/python/8uri_file_pph_meter_performace_reporting.py)
* cpp example is still to be done

```python
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
```
 
Example of the metrics that are printed every second.
```
FPS 0 (AVG)    FPS 1 (AVG)    FPS 2 (AVG)    FPS 3 (AVG)    FPS 4 (AVG)    FPS 5 (AVG)    FPS 6 (AVG)    FPS 7 (AVG)    
30.00 (30.99)  30.00 (30.99)  30.00 (30.99)  30.00 (30.99)  30.00 (30.99)  30.00 (30.99)  30.00 (30.99)  30.00 (30.99)  

'current-level-buffers' = 0/200 for component 'uri-source-0'
'current-level-buffers' = 0/200 for component 'uri-source-1'
'current-level-buffers' = 0/200 for component 'uri-source-2'
'current-level-buffers' = 0/200 for component 'uri-source-3'
'current-level-buffers' = 0/200 for component 'uri-source-4'
'current-level-buffers' = 0/200 for component 'uri-source-5'
'current-level-buffers' = 0/200 for component 'uri-source-6'
'current-level-buffers' = 0/200 for component 'uri-source-7'
'current-level-buffers' = 0/200 for component 'primary-gie'
'current-level-buffers' = 0/200 for component 'iou-tracker'
'current-level-buffers' = 3/200 for component 'tiler'
'current-level-buffers' = 3/200 for component 'on-screen-display'
'current-level-buffers' = 3/200 for component 'window-sink'
```

<br>

---

### Using a Simple DSL Player to test an RTSP Source Connection

* [`rtsp_player_to_test_connections.py`](/examples/python/rtsp_player_to_test_connections.py)
* cpp example is still to be done

```python
#
# This example can be used to test your RTSP Source connection. It uses a simple 
# DSL Player with a single RTSP Source and Window Sink:
# 
# There two example camera URI's below. One for AMCREST and one for HIKVISION.
# update one of the URI's with your username and password, or add your own
# URI format as needed.
# 
# Ensure that the RTSP Source constructor is using the correct URI.  
# 
# The example registers handler callback functions to be notified of:
#   - key-release events
#   - delete-window events
#   - RTSP Source change-of-state events
#  
```
<br>

---

### Running Inference on all MP4 files in a Folder

* [`process_all_mp4_files_in_folder.py`](/examples/python/process_all_mp4_files_in_folder.py)
* cpp example is still to be done

```python
#
# This simple example demonstrates how to process (infer-on) all .mp4 files
# in a given folder. 
#
# The inference Pipeline is built with the following components:
#   - URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
# 
# A Custom Pad-Probe-Handler is added to the Sink-Pad of the OSD
# to process the frame and object meta-data for each buffer received
#
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events.
#
# IMPORTANT! it is the end-of-stream (EOS) listener "eos_event_listener"
# that stops the Pipeline on EOS, sets the URI to the next file in 
# the list, and starts the Pipeline again.
#  
```
