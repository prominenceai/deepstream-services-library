# Dynamic Pipelines
This page documents the following examples:
* [Dynamically Update Inference Model Engine File](#dynamically-update-inference-model-engine-file)
* [Dynamically Add/Remove Sources to/from a Pipeline with a Tiler and Window Sink](#dynamically-addremove-sources-tofrom-a-pipeline-with-a-tiler-and-window-sink)
* [Dynamically Move a Branch from One Demuxer Stream to Another](#dynamically-move-a-branch-from-one-demuxer-stream-to-another)

<br>

---

### Dynamically Update Inference Model Engine File
* [dynamically_update_inference_model.py](/examples/python/dynamically_update_inference_model.py)
* [dynamically_update_inference_model.cpp](/examples/cpp/dynamically_update_inference_model.cpp)

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - File Source
#   - Primary GST Inference Engine (PGIE)
#   - DCF Tracker
#   - Secondary GST Inference Engines (SGIEs)
#   - On-Screen Display (OSD)
#   - Window Sink
# ...and how to dynamically update an Inference Engine's config and model files.
#  
# The key-release handler function will dynamically update the Secondary
# Inference Engine's config-file on the key value as follows.
#
#    "1" = '../../test/config/config_infer_secondary_vehicletypes.yml'
#    "2" = '../../test/config/config_infer_secondary_vehiclemake.yml'
#
# The new model engine is loaded by the SGIE asynchronously. a client listener 
# (callback) function is added to the SGIE to be notified when the loading is
# complete. See the "model_update_listener" function defined below.
#   
# IMPORTANT! it is best to allow the config file to specify the model engine
# file when updating both the config and model. Set the model_engine_file 
# parameter to None when creating the Inference component.
#  
#        retval = dsl_infer_gie_secondary_new(L"secondary-gie", 
#            secondary_infer_config_file_1.c_str(), NULL, L"primary-gie", 0);
#
# The Config files used are located under /deepstream-services-library/test/config
# The files reference models created with the file 
#     /deepstream-services-library/make_trafficcamnet_engine_files.py
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events#   
#
```
<br>

---

### Dynamically Add/Remove Sources to/from a Pipeline with a Tiler and Window Sink

* [`dynamically_add_remove_sources_with_tiler_window_sink.py`](/examples/python/dynamically_add_remove_sources_with_tiler_window_sink.py)
* [`dynamically_add_remove_sources_with_tiler_window_sink.cpp`](/examples/cpp/dynamically_add_remove_sources_with_tiler_window_sink.cpp)

```python
#
# This example shows how to dynamically add and remove Source components
# while the Pipeline is playing. The Pipeline must have at least once source
# while playing. The Pipeline consists of:
#   - A variable number of File Sources. The Source are created/added and 
#       removed/deleted on user key-input.
#   - The Pipeline's built-in streammuxer muxes the streams into a
#       batched stream as input to the Inference Engine.
#   - Primary GST Inference Engine (PGIE).
#   - IOU Tracker.
#   - Multi-stream 2D Tiler - created with rows/cols to support max-sources.
#   - On-Screen Display (OSD)
#   - Window Sink - with window-delete and key-release event handlers.
# 
# A Source component is created and added to the Pipeline by pressing the 
#  "+" key which calls the following services:
#
#    dsl_source_uri_new(source_name, uri_h265, True, 0, 0)
#    dsl_pipeline_component_add('pipeline', source_name)
#
# A Source component (last added) is removed from the Pipeline and deleted by 
# pressing the "-" key which calls the following services
#
#    dsl_pipeline_component_remove('pipeline', source_name)
#    dsl_component_delete(source_name)
#  
```
<br>

---

### Dynamically Move a Branch from One Demuxer Stream to Another

* [`dynamically_move_branch_from_demuxer_stream_to_stream.py`](/examples/python/dynamically_move_branch_from_demuxer_stream_to_stream.py)
* [`dynamically_move_branch_from_demuxer_stream_to_stream.cpp`](/examples/cpp/dynamically_move_branch_from_demuxer_stream_to_stream.cpp)

```python

#
# This example shows how to use a single dynamic demuxer branch with a 
# multi-source Pipeline. The Pipeline trunk consists of:
#   - 5 Streaming Images Sources - each streams a single image at a given 
#       frame-rate with a number overlayed representing the stream-id.
#   - The Pipeline's built-in streammuxer muxes the streams into a
#       batched stream as input to the Inference Engine.
#   - Primary GST Inference Engine (PGIE).
#   - IOU Tracker.
#
# The dynamic branch will consist of:
#   - On-Screen Display (OSD)
#   - Window Sink - with window-delete and key-release event handlers.
# 
# The branch is added to one of the Streams when the Pipeline is constructed
# by calling:
#
#    dsl_tee_demuxer_branch_add_to('demuxer', 'branch-0', stream_id)
#
# Once the Pipeline is playing, the example uses a simple periodic timer to 
# call a callback function which advances/cycles the current stream_id 
# variable and moves the branch by calling.
#
#    dsl_tee_demuxer_branch_move_to('demuxer', 'branch-0', stream_id)
#  
```
