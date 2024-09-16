# Basic Inference Pipelines with different Sources and Sinks
This page documents the following "Basic Inference Pipelines" consiting of
* [CSI Source, Primary GIE, OSD, and 3D Window Sink](#csi-source-primary-gie-osd-and-3d-window-sink)
* [File Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink, and File Sink](#file-source-primary-gie-iou-tracker-osd-egl-window-sink-and-file-sink)
* [File Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink, and RTSP Sink](#file-source-primary-gie-iou-tracker-osd-egl-window-sink-and-rtsp-sink)
* [File Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink, and V4L2 Sink](#file-source-primary-gie-iou-tracker-osd-egl-window-sink-and-v4l2-sink)
* [RTSP Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink](#rtsp-source-primary-gie-iou-tracker-osd-egl-window-sink)
* [HTTP Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink](#http-source-primary-gie-iou-tracker-osd-egl-window-sink)
* [File Source, Preprocessor, Primary GIE, IOU Tracker, OSD, EGL Window Sink](#file-source-preprocessor-primary-gie-iou-tracker-osd-egl-window-sink)
* [File Source, Primary TIS, DSF Tracker, OSD, EGL Window Sink](#file-source-primary-tis-dsf-tracker-osd-egl-window-sink)
* [File Source, Primary TIS, IOU Tracker, OSD, EGL Window Sink](#file-source-primary-tis-iou-tracker-osd-egl-window-sink)
* [File Source, Primary TIS, IOU Tracker, 2 Secondary TIS, OSD, EGL Window Sink](#file-source-primary-tis-iou-tracker-2-secondary-tis-osd-egl-window-sink)
* [Image Source, Primary GIE, OSD, and EGL Window Sink](#image-source-primary-gie-osd-and-egl-window-sink)
* [URI Source, Primary GIE, IOU Tracker, and App Sink](#uri-source-primary-gie-iou-tracker-and-app-sink)
* [V4L2 Source, Primary GIE, IOU Tracker, OSD, and EGL Window Sink](#v4l2-source-primary-gie-iou-tracker-osd-and-egl-window-sink)
* [App Source, Primary TIE, IOU Tracker, OSD, and EGL Window Sink](#app-source-primary-tie-iou-tracker-osd-and-egl-window-sink)

---

### CSI Source, Primary GIE, OSD, and 3D Window Sink

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

### File Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink, and File Sink

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

### File Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink, and RTSP Sink

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

### File Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink, and V4L2 Sink

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

### RTSP Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink

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

### HTTP Source, Primary GIE, IOU Tracker, OSD, EGL Window Sink

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

### File Source, Preprocessor, Primary GIE, IOU Tracker, OSD, EGL Window Sink

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

### File Source, Primary TIS, DSF Tracker, OSD, EGL Window Sink

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

### File Source, Primary TIS, IOU Tracker, OSD, EGL Window Sink

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

### File Source, Primary TIS, IOU Tracker, 2 Secondary TIS, OSD, EGL Window Sink

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

### Image Source, Primary GIE, OSD, and EGL Window Sink

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

### URI Source, Primary GIE, IOU Tracker, and App Sink

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

### V4L2 Source, Primary GIE, IOU Tracker, OSD, and EGL Window Sink

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

---

### App Source, Primary TIE, IOU Tracker, OSD, and EGL Window Sink

* Python example is still to be done
* [`raw_i420_app_src_ptis_tracker_osd_window.cpp`](/examples/cpp/raw_i420_app_src_ptis_tracker_osd_window.cpp)

```python
# 
# This example illustrates how to push raw video buffers to a DSL Pipeline
# using an App Source component. The example application adds the following
# client handlers to control the input of raw buffers to the App Source
#   * need_data_handler   - called when the App Source needs data to process
#   * enough_data_handler - called when the App Source has enough data to process
# 
# The client handlers add/remove a callback function to read, map, and push data
# to the App Source called "read_and_push_data". 
# 
# The raw video file used with this example is created by executing the following 
# gst-launch-1.0 command.
# 
# gst-launch-1.0 uridecodebin \
#       uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
#       ! nvvideoconvert ! 'video/x-raw, format=I420, width=1280, height=720' \
#       ! filesink location=./sample_720p.i420
# 
```
