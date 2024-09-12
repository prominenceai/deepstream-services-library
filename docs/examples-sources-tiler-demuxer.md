# Pipelines with multiple Sources and Tiler or Demuxer
<br>

---

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
