# Basic Inference Pipelies with different Sources and Sinks

---

* [`1csi_pgie_osd_3dsink.py`](/examples/python/1csi_pgie_osd_3dsink.py)
* cpp example is still to be done

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - CSI Source
#   - Primary GST Inference Engine (PGIE)
#   - On-Screen Display
#   - 3D Sink
# ...and how to add them to a new Pipeline and play.
#
# IMPORTANT! this examples uses a CSI Camera Source and 3D Sink - Jetson only!
#
```
<br>

---

* [`1file_pgie_iou_tracker_osd_window_file.py`](/examples/python/1file_pgie_iou_tracker_osd_window_file.py)
* [`1file_pgie_iou_tracker_osd_window_file.cpp`](/examples/cpp/1file_pgie_iou_tracker_osd_window_file.cpp)

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
#   - File Sink to encode and save the stream to file.
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1file_pgie_iou_tracker_osd_window_rtsp.py`](/examples/python/1file_pgie_iou_tracker_osd_window_rtsp.py)
* [`1file_pgie_iou_tracker_osd_window_rtsp.cpp`](/examples/cpp/1file_pgie_iou_tracker_osd_window_rtsp.cpp)

```python
#
# This simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - A File Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#   - RTSP Sink
# ...and how to add them to a new Pipeline and play.
#
# The example registers handler callback functions for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1file_pgie_iou_tracker_osd_window_v4l2.py`](/examples/python/1file_pgie_iou_tracker_osd_window_v4l2.py)
* [`1file_pgie_iou_tracker_osd_window_v4l2.cpp`](/examples/cpp/1file_pgie_iou_tracker_osd_window_v4l2.cpp)

```python
#
# This simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - A File Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
#   - V4L2 Sink
# ...and how to add them to a new Pipeline and play.
#
# The V4L2 Sink is used to display video to v4l2 video devices. 
# 
# V4L2 Loopback can be used to create "virtual video devices". Normal (v4l2) 
# applications will read these devices as if they were ordinary video devices.
# See: https://github.com/umlaeute/v4l2loopback for more information.
#
# You can install v4l2loopback with
#    $ sudo apt-get install v4l2loopback-dkms
#
# Run the following to setup '/dev/video3'
#    $ sudo modprobe v4l2loopback video_nr=3
#
# When the script is running, you can use the following GStreamer launch 
# command to test the loopback
#    $ gst-launch-1.0 v4l2src device=/dev/video3 ! videoconvert  ! xvimagesink
#
# The example registers handler callback functions for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1rtsp_pgie_dcf_tracker_osd_window.py`](/examples/python/1rtsp_pgie_dcf_tracker_osd_window.py)
* [`1rtsp_pgie_dcf_tracker_osd_window.cpp`](/examples/cpp/1rtsp_pgie_dcf_tracker_osd_window.cpp)

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - RTSP Source
#   - Primary GST Inference Engine (PGIE)
#   - DCF Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - error-message events
#   - Pipeline change-of-state events
#   - RTSP Source change-of-state events.
#  
# IMPORTANT! The error-message-handler callback fucntion will stop the Pipeline 
# and main-loop, and then exit. If the error condition is due to a camera
# connection failure, the application could choose to let the RTSP Source's
# connection manager periodically reattempt connection for some length of time.
#
```
<br>

---

* [`1uri_http_pgie_iou_tracker_osd_window.py`](/examples/python/1uri_http_pgie_iou_tracker_osd_window.py)
* [`1uri_http_pgie_iou_tracker_osd_window.cpp`](/examples/cpp/1uri_http_pgie_iou_tracker_osd_window.cpp)

```python
################################################################################
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - HTTP URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - component-buffering messages
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
# IMPORTANT! The URI Source will send messages on the Pipeline bus when
# buffering is in progress.  The buffering_message_handler callback is 
# added to the Pipeline to be called with every buffer message received.
# The handler callback is required to pause the Pipeline while buffering
# is in progress. 
#
# The callback is called with the percentage of buffering done, with 
# 100% indicating that buffering is complete.
#
################################################################################
```

<br>

---

* [`1file_preproc_pgie_iou_tracker_osd_window.py`](/examples/python/1file_preproc_pgie_iou_tracker_osd_window.py)
* [`1file_preproc_pgie_iou_tracker_osd_window.cpp`](/examples/cpp/1file_preproc_pgie_iou_tracker_osd_window.cpp)

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - URI Source
#   - Preprocessor
#   - Primary GIE
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
#
# Specific services must be called for the PGIE to be able to receive tensor-meta
# buffers from the Preprocessor component.
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1file_ptis_dcf_tracker_osd_window.py`](/examples/python/1file_ptis_dcf_tracker_osd_window.py)
* cpp example is still to be done

```python
#
# The example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary Triton Inference Server (PTIS)
#   - NcDCF Low Level Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1file_ptis_iou_tracker_osd_window.py`](/examples/python/1file_ptis_iou_tracker_osd_window.py)
* [`1file_ptis_iou_tracker_osd_window.cpp`](/examples/cpp/1file_ptis_iou_tracker_osd_window.cpp)

```python
#
# The example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary Triton Inference Server (PTIS)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1file_ptis_iou_tracker_2stis_osd_window.py`](/examples/python/1file_ptis_iou_tracker_2stis_osd_window.py)
* cpp example is still to be done

```python
#
# The example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary Triton Inference Server (PTIS)
#   - 2 Secondary Triton Inference Servers(STIS)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1image_jpeg_pgie_osd_window.py`](/examples/python/1image_jpeg_pgie_osd_window.py)
* cpp example is still to be done

```python
#
# The example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - Image Source - single image to EOS
#   - Primary GST Inference Engine (PGIE)
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
<br>

---

* [`1uri_file_pgie_iou_tracker_app_sink.py`](/examples/python/1uri_file_pgie_iou_tracker_app_sink.py)
* cpp example is still to be done

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - URI Source
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - APP Sink
# ...and how to add them to a new Pipeline and play
# 
# A "new_buffer_handler_cb" is added to the APP Sink to process the frame
# and object meta-data for each buffer received
#
```
<br>

---

* [`1v4l2_pgie_iou_tracker_osd_window.py`](/examples/python/1v4l2_pgie_iou_tracker_osd_window.py)
* [`1v4l2_pgie_iou_tracker_osd_window.cpp`](/examples/cpp/1v4l2_pgie_iou_tracker_osd_window.cpp)

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - V4L2 Source - Web Camera
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - On-Screen Display
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#
# The key-release handler function will update the V4L2 device picture settings
# based on the key value as follows during runtime.
#   * brightness - or more correctly the black level. 
#                  enter 'B' to increase, 'b' to decrease
#   * contrast   - color contrast setting or luma gain.
#                  enter 'C' to increase, 'c' to decrease
#   * hue        - color hue or color balence.
#                  enter 'H' to increase, 'h' to decrease
#
# The Picture Settings are all integer values, range 
```

```python

```

