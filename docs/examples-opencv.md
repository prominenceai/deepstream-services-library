# Dynamic Pipelines
<br>

---

* [`4file_custom_pph_using_opencv.py`](/examples/python/4file_custom_pph_using_opencv.py)

```python
#
# This simple example demonstrates how to use OpenCV with NVIDIA's pyds.
# The Pipeline used in this example is built with :
#   - 4 URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - Tiler
#   - On-Screen Display (OSD)
#   - Window Sink
# 
# A Custom Pad-Probe-Handler is added to the Sink-Pad of the Tiler
# to process the frame meta-data for each buffer received. The handler
# demonstrates how to 
#   - use pyds.get_nvds_buf_surface() to get a buffer surface.
#   - convert a frame to numpy array format with np.array().
#   - convert the array into cv2 default BGRA format using cv2.cvtColor().
#   - save the array as an image using opencv cv2.imwrite().
#
# IMPORTANT! pyds.get_nvds_buf_surface() requires 
#   1. The color format of the buffer must be set to RGBA by calling
#      dsl_source_video_buffer_out_format_set()
#   2. The memory type must be set to DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED
#      if running on dGPU. This is done by calling 
#        * dsl_pipeline_streammux_nvbuf_mem_type_set() - if using old streammux
#        * dsl_component_nvbuf_mem_type_set_many() - with all sources if using 
#          new streammux. 
#     
# IMPORTANT! The output folders (1 per source) must be created first
#   ./stream_0, ./stream_1, ./stream_2, ./stream_3,
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

* [`4file_custom_pph_using_opencv.cpp`](/examples/cpp/4file_custom_pph_using_opencv.cpp)

```python
#
# This simple example demonstrates how to use OpenCV with the DSL Surface 
# Transform utility classes in /src/"DslSurfaceTransform.h"
#
# The Pipeline used in this example is built with :
#   - 4 URI Sources
#   - Primary GST Inference Engine (PGIE)
#   - IOU Tracker
#   - Tiler
#   - On-Screen Display (OSD)
#   - Window Sink
# 
# A Custom Pad-Probe-Handler is added to the Sink-Pad (input) of the Tiler
# to process the frame meta-data for each buffer received. The handler
# demonstrates how to 
#   - create an RGBA buffer-surface from a single frame in a batched buffer
#     using the utility classes in /src/"DslSurfaceTransform.h"
#   - convert the buffer serface to a jpeg image using OpenCV.
#     
# IMPORTANT! All captured images are save to the IMAGE_OUTDIR = "./";
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#   - end-of-stream EOS events
#   - Pipeline change-of-state events
#  
```
