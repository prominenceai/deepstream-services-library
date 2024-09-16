# Pipelines with multiple Sources and Tiler or Demuxer
Examples using 
* [4 HTTP URI Sources, PGIE, Tracker, 2D Tiler, OSD, and Window Sink](#4-http-uri-sources-pgie-tracker-2d-tiler-osd-and-window-sink)
* [4 URI Sources with 2D Tiler Show Source Control](#4-uri-sources-with-2d-tiler-show-source-control)
* [2 URI Sources with Demuxer and 2 Branches](#2-uri-sources-with-demuxer-and-2-branches)

**Note:** there are other Tiler and Demuxer examples documented under
* [Advanced Inference Pipelies](/docs/examples-advanced-pipelines.md)
* [Dynamic Pipelines](/docs/examples-dynamic-pipelines.md)
* [Working with OpenCV](/docs/examples-opencv.md)
* [Diagnostics and Utilites](/docs/examples-diagnaostics-and-utilities.md)

<br> 

---

### 4 HTTP URI Sources, PGIE, Tracker, 2D Tiler, OSD, and Window Sink 

* [`4http_pgie_iou_tracker_tiler_osd_window.py`](/examples/python/4http_pgie_iou_tracker_tiler_osd_window.py)
* [`4http_pgie_iou_tracker_tiler_osd_window.cpp`](/examples/cpp/4http_pgie_iou_tracker_tiler_osd_window.cpp)

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - 4 HTTP URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - 2D Tiler
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - source-buffering messages
#   - key-release events
#   - delete-window events
#
# When using non-live streaming sources -- like the HTTP URI in this example --
# the application should pause the Pipeline when ever a Source is buffering. The 
# buffering_message_handler() callback funtion is added to the Pipeline to
# be called when a buffering-message is recieved on the Pipeline bus.
# The callback input parameters are 
#    - source - Source of the message == <source-name>-uridecodebin
#    - percent - the current buffer size as a percentage of the high watermark.
#    - client_data - unused in this simple example
# When a buffering message is received (percent < 100) the calback will pause
# the Pipeline. When a buffering message with 100% is received the callback
# resumes the Pipeline playback,
#
```

<br> 

---

### 4 URI Sources with 2D Tiler Show Source Control  

* [`4uri_file_tiler_show_source_control.py`](/examples/python/4uri_file_tiler_show_source_control.py)
* [`4uri_file_tiler_show_source_control.cpp`](/examples/cpp/4uri_file_tiler_show_source_control.cpp)

```python
#
# This example demonstrates how to manually control -- using key release and 
# button press events -- the 2D Tiler's output stream to: 
#   - show a specific source on key input (source No.) or mouse click on tile.
#   - to return to showing all sources on 'A' key input, mouse click, or timeout.
#   - to cycle through all sources on 'C' input showing each for timeout.
# 
# Note: timeout is controled with the global variable SHOW_SOURCE_TIMEOUT 
# 
# The example uses a basic inference Pipeline consisting of:
#   - 4 URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - 2D Tiler
#   - On-Screen Display
#   - Window Sink
#  
```

<br> 

---

### 2 URI Sources with Demuxer and 2 Branches  

* [`2uri_file_pgie_iou_tracker_demuxer_2osd_2window.py`](/examples/python/2uri_file_pgie_iou_tracker_demuxer_2osd_2window.py)
* [`2uri_file_pgie_iou_tracker_demuxer_2osd_2window.cpp`](/examples/cpp/2uri_file_pgie_iou_tracker_demuxer_2osd_2window.cpp)

```python
#
# This example demonstrates how to create an Inference Pipeline with two
# Sources, built-in Streammuxer, and Demuxer with two branches; one per demuxed
# Stream. Eaxh branch has an On-Screen-Display and EGL Window Sink.   
#
```