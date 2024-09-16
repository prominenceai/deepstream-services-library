# Video Dewarping and Segmentation Viewing
This page documents the following Segmentation examples:
* [360 Degree Video Dewarping](#360-degree-video-dewarping)
* [Perspective Video Dewarping](#perspective-video-dewarping)
* [Industrial Segmentation and Viewing](#industrial-segmentation-and-viewing)
* [Semantic Segmentation and Viewing](#semantic-segmentation-and-viewing)

<br>

---

### 360 Degree Video Dewarping

* [`video_dewarper_360.py`](/examples/python/video_dewarper_360.py)
* cpp example is still to be done

```python
# This example shows the use of a Video Dewarper to dewarp a 360d camera stream 
#   - recorded from a 360d camera and provided by NVIDIA as a sample stream.
#
# The Dewarper component is created with the following parameters
#   - a config "file config_dwarper_txt" which tailors this 360d camera 
#     multi-surface use-case.
#   - and a camera-id which refers to the first column of the CSV files 
#     (i.e. csv_files/nvaisle_2M.csv & csv_files/nvspot_2M.csv).
#     The dewarping parameters for the given camera are read from CSV 
#     files and used to generate dewarp surfaces (i.e. multiple aisle 
#     and spot surface) from 360d input video stream.
# All files are located under:
#   /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/
```
<br>

---

### Perspective Video Dewarping

* [`video_dewarper_perspective.py`](/examples/python/video_dewarper_perspective.py)
* cpp example is still to be done

```python
#
# This example shows the use of a Video Dewarper to dewarp a perspective view.
#
# The Dewarper component is created with the following parameters:
#   - a config "config_dewarper_perspective.txt" which defines all dewarping 
#     parameters - i.e. the csv files are not used for this example. 
#   - and a camera-id which is NOT USED! Perspecitve dewarping requires that all
#     parameters be defined in the config file. 
# All files are located under:
#   /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/
#
```
<br>

---

### Industrial Segmentation and Viewing

* [`segmentation_industrial.py`](/examples/python/segmentation_industrial.py)
* cpp example is still to be done

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - URI Source to read a jpeg image file
#   - Primary GST Inference Engine (PGIE)
#   - Segmentation Visualizer
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#
# NOTE: The Primary GST Inference engine is configured for Industrial Segmentation.
#   The NVIDIA provided PGIE configuration file can be found at
#     /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-segmentation-test/
#
# The URI Source will push a single frame followed by an End of File (EOF) event.
# 
```

<br>

---

### Semantic Segmentation and Viewing

* [`segmentation_semantic.py`](/examples/python/segmentation_semantic.py)
* cpp example is still to be done

```python
#
# The simple example demonstrates how to create a set of Pipeline components, 
# specifically:
#   - URI Source to read a jpeg image file
#   - Primary GST Inference Engine (PGIE)
#   - Segmentation Visualizer
#   - Window Sink
# ...and how to add them to a new Pipeline and play
# 
# The example registers handler callback functions with the Pipeline for:
#   - key-release events
#   - delete-window events
#
# NOTE: The Primary GST Inference engine is configured for Semantic Segmentation.
#   The NVIDIA provided PGIE configuration file can be found at
#     /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-segmentation-test/
#
# The URI Source will push a single frame followed by an End of File (EOF) event.
# 
```
