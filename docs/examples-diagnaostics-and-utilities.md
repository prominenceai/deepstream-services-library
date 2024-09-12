# Pipeline Diagnostics and Utilities

<br>

---

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
<br>

---

* [`process_all_mp4_files_in_folder.py`](/examples/python/process_all_mp4_files_in_folder.py)
* cpp example is still to be done

```python
