# Smart Recording Taps and Sinks
<br>

---

* [`smart_record_tap_start_session_on_ode_occurrence.py`](/examples/python/smart_record_tap_start_session_on_ode_occurrence.py)
* [`smart_record_tap_start_session_on_ode_occurrence.cpp`](/examples/cpp/smart_record_tap_start_session_on_ode_occurrence.cpp)

```python
# ````````````````````````````````````````````````````````````````````````````````````
# This example demonstrates the use of a Smart-Record Tap and how to start
# a recording session on the "occurrence" of an Object Detection Event (ODE).
# An ODE Occurrence Trigger, with a limit of 1 event, is used to trigger
# on the first detection of a Person object. The Trigger uses an ODE "Start 
# Recording Session Action" setup with the following parameters:
#   start:    the seconds before the current time (i.e.the amount of 
#             cache/history to include.
#   duration: the seconds after the current time (i.e. the amount of 
#             time to record after session start is called).
# Therefore, a total of start-time + duration seconds of data will be recorded.
# 
# **IMPORTANT!** 
# 1. The default max_size for all Smart Recordings is set to 600 seconds. The 
#    recording will be truncated if start + duration > max_size.
#    Use dsl_tap_record_max_size_set to update max_size. 
# 2. The default cache-size for all recordings is set to 60 seconds. The 
#    recording will be truncated if start > cache_size. 
#    Use dsl_tap_record_cache_size_set to update cache_size.
#
# Record Tap components tap into RTSP Source components pre-decoder to enable
# smart-recording of the incomming (original) H.264 or H.265 stream. 
#
# Additional ODE Actions are added to the Trigger to 1) print the ODE 
# data (source-id, batch-id, object-id, frame-number, object-dimensions, etc.)
# to the console and 2) to capture the object (bounding-box) to a JPEG file.
# 
# A basic inference Pipeline is used with PGIE, Tracker, Tiler, OSD, and Window Sink.
#
# DSL Display Types are used to overlay text ("REC") with a red circle to
# indicate when a recording session is in progress. An ODE "Always-Trigger" and an 
# ODE "Add Display Meta Action" are used to add the text's and circle's metadata
# to each frame while the Trigger is enabled. The record_event_listener callback,
# called on both DSL_RECORDING_EVENT_START and DSL_RECORDING_EVENT_END, enables
# and disables the "Always Trigger" according to the event received. 
#
# IMPORTANT: the record_event_listener is used to reset the one-shot Occurrence-
# Trigger when called with DSL_RECORDING_EVENT_END. This allows a new recording
# session to be started on the next occurrence of a Person. 
#
# IMPORTANT: this demonstrates a multi-source Pipeline, each with their own
# Smart-Recort Tap.

#!/usr/bin/env python    
```
<br>

---

* [`smart_record_tap_start_session_on_user_demand.py`](/examples/python/smart_record_tap_start_session_on_user_demand.py)
* [`smart_record_tap_start_session_on_user_demand.cpp`](/examples/cpp/smart_record_tap_start_session_on_user_demand.cpp)

```python
# ````````````````````````````````````````````````````````````````````````````````````
# This example demonstrates the use of a Smart-Record Tap and how
# to start a recording session on user/viewer demand - in this case
# by pressing the 'S' key.  The xwindow_key_event_handler calls
# dsl_tap_record_session_start with:
#   start:    the seconds before the current time (i.e.the amount of 
#             cache/history to include.
#   duration: the seconds after the current time (i.e. the amount of 
#             time to record after session start is called).
# Therefore, a total of start-time + duration seconds of data will be recorded.
# 
# **IMPORTANT!** 
# 1. The default max_size for all Smart Recordings is set to 600 seconds. The 
#    recording will be truncated if start + duration > max_size.
#    Use dsl_tap_record_max_size_set to update max_size. 
# 2. The default cache-size for all recordings is set to 60 seconds. The 
#    recording will be truncated if start > cache_size. 
#    Use dsl_tap_record_cache_size_set to update cache_size.
#
# Record Tap components tap into RTSP Source components pre-decoder to enable
# smart-recording of the incomming (original) H.264 or H.265 stream. 
# 
# A basic inference Pipeline is used with PGIE, Tracker, OSD, and Window Sink.
#
# DSL Display Types are used to overlay text ("REC") with a red circle to
# indicate when a recording session is in progress. An ODE "Always-Trigger" and an 
# ODE "Add Display Meta Action" are used to add the text's and circle's metadata
# to each frame while the Trigger is enabled. The record_event_listener callback,
# called on both DSL_RECORDING_EVENT_START and DSL_RECORDING_EVENT_END, enables
# and disables the "Always Trigger" according to the event received. 
```
<br>

---

* [`smart_record_sink_start_session_on_ode_occurrence.py`](/examples/python/smart_record_sink_start_session_on_ode_occurrence.py)
* [`smart_record_sink_start_session_on_ode_occurrence.cpp`](/examples/cpp/smart_record_sink_start_session_on_ode_occurrence.cpp)

```python
# ````````````````````````````````````````````````````````````````````````````````````
# This example demonstrates the use of a Smart-Record Sink and how to start
# a recording session on the "occurrence" of an Object Detection Event (ODE).
# An ODE Occurrence Trigger, with a limit of 1 event, is used to trigger
# on the first detection of a Person object. The Trigger uses an ODE "Start 
# Recording Session Action" setup with the following parameters:
#   start:    the seconds before the current time (i.e.the amount of 
#             cache/history to include.
#   duration: the seconds after the current time (i.e. the amount of 
#             time to record after session start is called).
# Therefore, a total of start-time + duration seconds of data will be recorded.
# 
# **IMPORTANT!** 
# 1. The default max_size for all Smart Recordings is set to 600 seconds. The 
#    recording will be truncated if start + duration > max_size.
#    Use dsl_sink_record_max_size_set to update max_size. 
# 2. The default cache-size for all recordings is set to 60 seconds. The 
#    recording will be truncated if start > cache_size. 
#    Use dsl_sink_record_cache_size_set to update cache_size.
#
# Additional ODE Actions are added to the Trigger to 1) to print the ODE 
# data (source-id, batch-id, object-id, frame-number, object-dimensions, etc.)
# to the console and 2) to capture the object (bounding-box) to a JPEG file.
# 
# A basic inference Pipeline is used with PGIE, Tracker, OSD, and Window Sink.
#
# DSL Display Types are used to overlay text ("REC") with a red circle to
# indicate when a recording session is in progress. An ODE "Always-Trigger" and an 
# ODE "Add Display Meta Action" are used to add the text's and circle's metadata
# to each frame while the Trigger is enabled. The record_event_listener callback,
# called on both DSL_RECORDING_EVENT_START and DSL_RECORDING_EVENT_END, enables
# and disables the "Always Trigger" according to the event received. 
#
# IMPORTANT: the record_event_listener is used to reset the one-shot Occurrence-
# Trigger when called with DSL_RECORDING_EVENT_END. This allows a new recording
# session to be started on the next occurrence of a Person. 
```
<br>

---

* [`smart_record_sink_start_session_on_user_demand.py`](/examples/python/smart_record_sink_start_session_on_user_demand.py)
* [`smart_record_sink_start_session_on_user_demand.cpp`](/examples/cpp/smart_record_sink_start_session_on_user_demand.cpp)

```python
# ````````````````````````````````````````````````````````````````````````````````````
# This example demonstrates the use of a Smart-Record Sink and how
# to start a recording session on user/viewer demand - in this case
# by pressing the 'S' key.  The xwindow_key_event_handler calls
# dsl_sink_record_session_start with:
#   start:    the seconds before the current time (i.e.the amount of 
#             cache/history to include.
#   duration: the seconds after the current time (i.e. the amount of 
#             time to record after session start is called).
# Therefore, a total of start-time + duration seconds of data will be recorded.
# 
# **IMPORTANT!** 
# 1. The default max_size for all Smart Recordings is set to 600 seconds. The 
#    recording will be truncated if start + duration > max_size.
#    Use dsl_sink_record_max_size_set to update max_size. 
# 2. The default cache-size for all recordings is set to 60 seconds. The 
#    recording will be truncated if start > cache_size. 
#    Use dsl_sink_record_cache_size_set to update cache_size.
#
# A basic inference Pipeline is used with PGIE, Tracker, OSD, and Window Sink.
#
# DSL Display Types are used to overlay text ("REC") with a red circle to
# indicate when a recording session is in progress. An ODE "Always-Trigger" and an 
# ODE "Add Display Meta Action" are used to add the text's and circle's metadata
# to each frame while the Trigger is enabled. The record_event_listener callback,
# called on both DSL_RECORDING_EVENT_START and DSL_RECORDING_EVENT_END, enables
# and disables the "Always Trigger" according to the event received. 
```

