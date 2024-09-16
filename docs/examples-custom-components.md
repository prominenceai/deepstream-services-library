# Pipelines with Custom Sources, Components, and Sinks
This page documents the following examples:
* [Pipeline with Custom Source Component](#pipeline-with-custom-source-component)
* [Pipeline with Custom Component - non Source or Sink](#pipeline-with-custom-component---non-source-or-sink)
* [Pipeline with Custom Sink Component](#pipeline-with-custom-sink-component)

<br>

---

### Pipeline with Custom Source Component

* [`pipeline_with_custom_source.py`](/examples/python/pipeline_with_custom_source.py)
* [`pipeline_with_custom_source.cpp`](/examples/cpp/pipeline_with_custom_source.cpp)

```python
#
# This example demonstrates how to create a custom DSL Source Component  
# using two GStreamer (GST) Elements created from two GST Plugins:
#   1. 'videotestsrc' as the source element.
#   2. 'capsfilter' to limit the video from the videotestsrc to  
#      'video/x-raw, framerate=15/1, width=1280, height=720'
#   
# Elements are constructed from plugins installed with GStreamer or 
# using your own proprietary plugin with a call to
#
#     dsl_gst_element_new('my-element', 'my-plugin-factory-name' )
#
# Multiple elements can be added to a Custom Source on creation be calling
#
#    dsl_source_custom_new_element_add_many('my-custom-source',
#        ['my-element-1', 'my-element-2', None])
#
# As with all DSL Video Sources, the Custom Souce will also include the 
# standard buffer-out-elements (queue, nvvideconvert, and capsfilter). 
# The Source in this example will be linked as follows:
#
#   videotestscr->capsfilter->queue->nvvideconvert->capsfilter
#
# See the GST and Source API reference sections for more information
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-source.md
#
```
<br>

---

### Pipeline with Custom Component - non Source or Sink

* [`pipeline_with_custom_component.py`](/examples/python/pipeline_with_custom_component.py)
* [`pipeline_with_custom_component.cpp`](/examples/cpp/pipeline_with_custom_component.cpp)

```python
#
# The example demonstrates how to create a custom DSL Pipeline Component with
# a custom GStreamer (GST) Element.  
#
# Elements are constructed from plugins installed with GStreamer or 
# using your own proprietary -- with a call to
#
#     dsl_gst_element_new('my-element', 'my-plugin-factory-name' )
#
# IMPORTANT! All DSL Pipeline Components, intrinsic and custom, include
# a queue element to create a new thread boundary for the component's element(s)
# to process in. 
#
# This example creates a simple Custom Component with two elements
#  1. The built-in 'queue' plugin - to create a new thread boundary.
#  2. An 'identity' plugin - a GST debug plugin to mimic our proprietary element.
#
# A single GST Element can be added to the Component on creation by calling
#
#    dsl_component_custom_new_element_add('my-custom-component',
#        'my-element')
#
# Multiple elements can be added to a Component on creation be calling
#
#    dsl_component_custom_new_element_add_many('my-bin',
#        ['my-element-1', 'my-element-2', None])
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-component.md
#
```
<br>

---

### Pipeline with Custom Sink Component

* [`pipeline_with_custom_sink.py`](/examples/python/pipeline_with_custom_source.py)
* [`pipeline_with_custom_sink.cpp`](/examples/cpp/pipeline_with_custom_sink.cpp)

```python
#
# The example demonstrates how to create a custom DSL Sink Component with
# using custom GStreamer (GST) Elements.  
#
# Elements are constructed from plugins installed with GStreamer or 
# using your own proprietary with a call to
#
#     dsl_gst_element_new('my-element', 'my-plugin-factory-name' )
#
# IMPORTANT! All DSL Pipeline Components, intrinsic and custom, include
# a queue element to create a new thread boundary for the component's element(s)
# to process in. 
#
# This example creates a simple Custom Sink with four elements in total
#  1. The built-in 'queue' element - to create a new thread boundary.
#  2. An 'nvvideoconvert' element -  to convert the buffer from 
#     'video/x-raw(memory:NVMM)' to 'video/x-raw'
#  3. A 'capsfilter' plugin - to filter the 'nvvideoconvert' caps to 
#     'video/x-raw'
#  4. A 'glimagesink' plugin - the actual Sink element for this Sink component.
#
# Multiple elements can be added to a Custom Sink on creation be calling
#
#    dsl_sink_custom_new_element_add_many('my-bin',
#        ['my-element-1', 'my-element-2', 'my-element-3', None])
#
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-gst.md
# https://github.com/prominenceai/deepstream-services-library/tree/master/docs/api-sink.md
#
```


