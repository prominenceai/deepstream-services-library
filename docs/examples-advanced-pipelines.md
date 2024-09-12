# Advanced Inference Pipelies
<br>

---

* [`parallel_inference_on_selective_streams.py`](/examples/python/parallel_inference_on_selective_streams.py)
* [`parallel_inference_on_selective_streams.cpp`](/examples/cpp/parallel_inference_on_selective_streams.cpp)

```python
#
# This example shows how to use a Remuxer Component to create parallel branches,
# each with their own Inference Components (Preprocessors, Inference Engines, 
# Trackers, for example). 
# IMPORTANT! All branches are (currently) using the same model engine and config.
# files, which is not a valid use case. The actual inference components and 
# models to use for any specific use cases is beyond the scope of this example. 
#
# Each Branch added to the Remuxer can specify which streams to process or
# to process all. Use the Remuxer "branch-add-to" service to add to specific streams.
#
#    stream_ids = [0,1]
#    dsl_remuxer_branch_add_to('my-remuxer', 'my-branch-0', 
#        stream_ids, len[stream_ids])
#
# You can use the "branch-add" service if adding to all streams
#
#    dsl_remuxer_branch_add('my-remuxer', 'my-branch-0')
# 
# In this example, 4 RTSP Sources are added to the Pipeline:
#   - branch-1 will process streams [0,1]
#   - branch-2 will process streams [1,2]
#   - branch-3 will process streams [0,2,3]
#
# Three ODE Instance Triggers are created to trigger on new object instances
# events (i.e. new tracker ids). Each is filtering on a unique class-i
# (vehicle, person, and bicycle). 
#
# The ODE Triggers are added to an ODE Handler which is added to the src-pad
# (output) of the Remuxer.
#
# A single ODE Print Action is created and added to each Trigger (shared action).
# Using multiple Print Actions running in parallel -- each writing to the same 
# stdout buffer -- will result in the printed data appearing interlaced. A single 
# Action with an internal mutex will protect from stdout buffer reentrancy. 
# 
```
<br>

---

* [`interpipe_multiple_pipelines_listening_to_single_sink.py`](/examples/python/interpipe_multiple_pipelines_listening_to_single_sink.py)
* [`interpipe_multiple_pipelines_listening_to_single_sink.cpp`](/examples/cpp/interpipe_multiple_pipelines_listening_to_single_sink.cpp)

```python
#
# This script demonstrates how to run multple Pipelines, each with an Interpipe
# Source, both listening to the same Interpipe Sink.
#
# A single Player is created with a File Source and Interpipe Sink. Two Inference
# Pipelines are created to listen to the single Player - shared input stream. 
# The two Pipelines can be created with different configs, models, and/or Trackers
# for side-by-side comparison. Both Pipelines run in their own main-loop with their 
# own main-context, and have their own Window Sink for viewing and external control.
#
```
<br>

---

* [`interpipe_single_pipeline_dynamic_switching_between_multiple_sinks.py`](/examples/python/interpipe_single_pipeline_dynamic_switching_between_multiple_sinks.py)
* [`interpipe_single_pipeline_dynamic_switching_between_multiple_sinks.cpp`](/examples/cpp/interpipe_single_pipeline_dynamic_switching_between_multiple_sinks.cpp)

```python
# ------------------------------------------------------------------------------------
# This example demonstrates interpipe dynamic switching. Four DSL Players
# are created, each with a File Source and Interpipe Sink. A single
# inference Pipeline with an Interpipe Source is created as the single listener
# 
# The Interpipe Source's "listen_to" setting is updated based on keyboard input.
# The xwindow_key_event_handler (see below) is added to the Pipeline's Window Sink.
# The handler, on key release, sets the "listen_to" setting to the Interpipe Sink
# name that corresponds to the key value - 1 through 4.
```

