# Component API Reference
The Pipeline Component API services support:
* Custom Components
* Component Deletion
* Component Queue Management
* GPUID & NVIDIA Buffer MemType Settings

List of component types that are used with this API:
* [Sources](/docs/api-source.md)
* [Dewarpers](/docs/api-dewarper.md)
* [Taps](/docs/api-tap.md)
* [Preprocessors](/docs/api-preproc.md)
* [Inference Engines and Servers](/docs/api-infer.md)
* [Trackers](/docs/api-tracker.md)
* [Segmentation Visualizers](/dos/api-segvisual.md)
* [Splitters and Demuxers](/docs/api-tee.md)
* [Remuxers](/docs/api-remxer.md)
* [2D-Tilers](/docs/api-tiler.md)
* [On Screen Displays](/docs/api-osd.md)
* [Sinks](/docs/api-sink.md)
* [Custom Components](/docs/api-gst.md)

## Custom Components
The Custom Component API is used to create custom DSL Pipeline Components -- other than Sources and Sinks -- using [GStreamer (GST) Elements](/docs/api-gst.md) created from proprietary or built-in GStreamer plugins. See also [Custom Sources](/docs/api-source.md#custom-video-sources) and [Custom Sinks](/docs/api-sink.md#custom-video-sinks).

Custom Components are created by calling [`dsl_component_custom_new`](#dsl_component_custom_new), [`dsl_component_custom_new_element_add`](#dsl_component_custom_new_element_add) or [`dsl_component_custom_new_element_add_many`](#dsl_component_custom_new_element_add_many). As with all Pipeline Components, Custom Components are deleted by calling [`dsl_component_delete`](/docs/api-component.md#dsl_component_delete), [`dsl_component_delete_many`](/docs/api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](/docs/api-component.md#dsl_component_delete_all).

### Adding and Removing GST Elements
The relationship between Custom Components and GST Elements is one to many. Once added to a Custom Component, an Element must be removed before it can be used with another. Elements can be added to Custom Components when constructed by calling [`dsl_component_custom_new_element_add`](#dsl_component_custom_new_element_add) or [`dsl_component_custom_new_element_add_many`](#dsl_component_custom_new_element_add_many),  or after construction by calling [`dsl_component_custom_element_add`](#dsl_component_custom_element_add) and [`dsl_component_custom_element_add_many`](#dsl_component_custom_element_add_many). GST Elements can be removed from a Custom Component by calling [`dsl_component_custom_element_remove`](#dsl_component_custom_element_remove) or [`dsl_component_custom_element_remove_many`](#dsl_component_custom_element_remove_many).

**IMPORTANT!** Elements are linked in the order they are added.

### Adding and Removing Custom Components
The relationship between Pipelines/Branches and Custom Components is one to many. Once added to a Pipeline or Branch, a Custom Component must be removed before it can be used with another. Custom Components are added to a Pipeline by calling [`dsl_pipeline_component_add`](/docs/api-pipeline.md#dsl_pipeline_component_add) or [`dsl_pipeline_component_add_many`](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [`dsl_pipeline_component_remove`](/docs/api-pipeline.md#dsl_pipeline_component_remove), [`dsl_pipeline_component_remove_many`](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [`dsl_pipeline_component_remove_all`](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

A similar set of Services are used when adding/removing a to/from a branch: [`dsl_branch_component_add`](api-branch.md#dsl_branch_component_add), [`dsl_branch_component_add_many`](/docs/api-branch.md#dsl_branch_component_add_many), [`dsl_branch_component_remove`](/docs/api-branch.md#dsl_branch_component_remove), [`dsl_branch_component_remove_many`](/docs/api-branch.md#dsl_branch_component_remove_many), and [`dsl_branch_component_remove_all`](/docs/api-branch.md#dsl_branch_component_remove_all).

Below is a simple example that creates a GST Element and adds it to a new Custom Component.

```Python
# Create a new element from a proprietary plugin
retval = dsl_gst_element_new('my-element', 'my-plugin-name')
        
# Create a new Custom Component and add the element to it. If adding multiple
# elements they will be linked in the order they are added.
ret_val = dsl_component_custom_new_element_add('my-custom-component',
  'my-element')
# IMPORTANT! the element(s) added will be linked to the Custom Components queue

# The Custom Component can now be added to our Pipeline along with
# the other Pipeline components. Add Components in the order to be linked.
retval = dsl_pipeline_new_component_add_many('pipeline',
  ['my-source', 'my-primary-gie', 'my-iou-tracker', 'my-custom-component',
  'my-on-screen-display', 'my-egl-sink', None])
```
## Relevant Examples
For relevant examples see:
* [pipeline_with_custom_component.py](/examples/python/pipeline_with_custom_component.py)
* [pipeline_with_custom_component.cpp](/examples/python/pipeline_with_custom_component.cpp)

## Component Deletion
Components, once created with their type specific constructor, are deleted by calling [`dsl_component_delete`](#dsl_component_delete), [`dsl_component_delete_many`](#dsl_component_delete_many), or [`dsl_component_delete_all`](#dsl_component_delete_all).

## Component Queue Management
All DSL Pipeline Components are derived from the [GStreamer (GST) Bin](https://gstreamer.freedesktop.org/documentation/application-development/basics/bins.html?gi-language=c) container class. Bins are used to contain [GST Elements](https://gstreamer.freedesktop.org/documentation/application-development/basics/bins.html?gi-language=c). Bins combine multiple linked Elements into one logical Element.

The first element in all DSL Pipeline Components (except Sources), including Custom Components, is a [GStreamer queue](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html?gi-language=c). Queues create new threads to decouple the processing on the sink (input) an source (output) pads. Queues, therefore, decouple the processing of the Component's core plugin(s) from those upstream.

The image below shows a [GStreamer graph](/docs/debugging-dsl.md#creating-pipeline-graphs) of a DSL Tracker Component. Click the image to view full size.

![DSL Preprocessor Component](/Images/dsl-preprocessor.png)

**IMPORTANT!** From the GStreamer documentation:
> *"Data is queued until one of the limits specified by the max-size-buffers, max-size-bytes and/or max-size-time properties has been reached. Any attempt to push more buffers into the queue will block the pushing thread until more space becomes available."*

A Components current queue level (in buffers, bytes, or time) is queried by calling [`dsl_component_queue_current_level_get`](#dsl_component_queue_current_level_get). You can print the current level to stdout by calling [`dsl_component_queue_current_level_print`](#dsl_component_queue_current_level_print) or [`dsl_component_queue_current_level_print_many`](#dsl_component_queue_current_level_print_many), or you can log the current level by calling [`dsl_component_queue_current_level_log`](#dsl_component_queue_current_level_log)
[`dsl_component_queue_current_level_log_many`](#dsl_component_queue_current_level_log_many) at the log level of INFO(4).

The default queue size limits are 200 buffers, 10MB of data, or one second worth of data, whichever is reached first.

The queue's leaky property can be set by calling [`dsl_component_queue_leaky_set`](#dsl_component_queue_leaky_set) or [`dsl_component_queue_leaky_set_many`](#dsl_component_queue_leaky_set_many) so that it leaks (drops) new or old buffers instead of blocking the pushing thread as mentioned above. The current setting can be queried by calling [`dsl_component_queue_leaky_get`](#dsl_component_queue_leaky_get).

The client application can add callback functions to listen for:
* **Queue overrun**  (`max-size` reached) by calling [`dsl_component_queue_overrun_listener_add`](#dsl_component_queue_overrun_listener_add) or [`dsl_component_queue_overrun_listener_add_many`](#dsl_component_queue_overrun_listener_add_many). The queue's `max-size` (in buffers, bytes, and time) can be set by calling [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set) or [`dsl_component_queue_max_size_set_many`](#dsl_component_queue_max_size_set_many).
* Queue underrun (`min-threshold` reached) by calling [`dsl_component_queue_underrun_listener_add`](#dsl_component_queue_underrun_listener_add) or [`dsl_component_queue_underrun_listener_add_many`](#dsl_component_queue_underrun_listener_add_many). The queue's `min-threshold` (in buffers, bytes, and time) can be set by calling [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set) or [`dsl_component_queue_min_threshold_set_many`](#dsl_component_queue_min_threshold_set_many).

**IMPORTANT!** These callbacks are called from the context of the streaming thread.

The max-size and min-threshold settings can be queried by calling [`dsl_component_queue_max_size_get`](#dsl_component_queue_max_size_get) and [`dsl_component_queue_min_threshold_get`](#dsl_component_queue_min_threshold_get) respectively.

---

## Component API
**Client Callback Typedefs**
* [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb)
* [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb)

**Constructors**
* [`dsl_component_custom_new`](#dsl_component_custom_new)
* [`dsl_component_custom_new_element_add`](#dsl_component_custom_new_element_add)
* [`dsl_component_custom_new_element_add_many`](#dsl_component_custom_new_element_add_many)

**Destructors**
* [`dsl_component_delete`](#dsl_component_delete)
* [`dsl_component_delete_many`](#dsl_component_delete_many)
* [`dsl_component_delete_all`](#dsl_component_delete_all)

**Methods**
* [`dsl_component_custom_element_add`](#dsl_component_custom_element_add)
* [`dsl_component_custom_element_add_many`](#dsl_component_custom_element_add_many)
* [`dsl_component_custom_element_remove`](#dsl_component_custom_element_remove)
* [`dsl_component_custom_element_remove_many`](#dsl_component_custom_element_remove_many)
* [`dsl_component_list_size`](#dsl_component_list_size)
* [`dsl_component_queue_current_level_get`](#dsl_component_queue_current_level_get)
* [`dsl_component_queue_current_level_print`](#dsl_component_queue_current_level_print)
* [`dsl_component_queue_current_level_print_many`](#dsl_component_queue_current_level_print_many)
* [`dsl_component_queue_current_level_log`](#dsl_component_queue_current_level_log)
* [`dsl_component_queue_current_level_log_many`](#dsl_component_queue_current_level_log_many)
* [`dsl_component_queue_leaky_get`](#dsl_component_queue_leaky_get)
* [`dsl_component_queue_leaky_set`](#dsl_component_queue_leaky_set)
* [`dsl_component_queue_leaky_set_many`](#dsl_component_queue_leaky_set_many)
* [`dsl_component_queue_max_size_get`](#dsl_component_queue_max_size_get)
* [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set)
* [`dsl_component_queue_max_size_set_many`](#dsl_component_queue_max_size_set_many)
* [`dsl_component_queue_min_threshold_get`](#dsl_component_queue_min_threshold_get)
* [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set)
* [`dsl_component_queue_min_threshold_set_many`](#dsl_component_queue_min_threshold_set_many)
* [`dsl_component_queue_overrun_listener_add`](#dsl_component_queue_overrun_listener_add)
* [`dsl_component_queue_overrun_listener_add_many`](#dsl_component_queue_overrun_listener_add_many)
* [`dsl_component_queue_overrun_listener_remove`](#dsl_component_queue_overrun_listener_remove)
* [`dsl_component_queue_underrun_listener_add`](#dsl_component_queue_underrun_listener_add)
* [`dsl_component_queue_underrun_listener_add_many`](#dsl_component_queue_underrun_listener_add_many)
* [`dsl_component_queue_underrun_listener_remove`](#dsl_component_queue_underrun_listener_remove)
* [`dsl_component_gpuid_get`](#dsl_component_gpuid_get)
* [`dsl_component_gpuid_set`](#dsl_component_gpuid_set)
* [`dsl_component_gpuid_set_many`](#dsl_component_gpuid_set_many)
* [`dsl_component_nvbuf_mem_type_get`](#dsl_component_nvbuf_mem_type_get)
* [`dsl_component_nvbuf_mem_type_set`](#dsl_component_nvbuf_mem_type_set)
* [`dsl_component_nvbuf_mem_type_set_many`](#dsl_component_nvbuf_mem_type_set_many)

## Return Values
The following return codes are used by the Component API
```C++
#define DSL_RESULT_SUCCESS                                          0x00000000

#define DSL_RESULT_COMPONENT_RESULT                                 0x00010000
#define DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE                        0x00010001
#define DSL_RESULT_COMPONENT_NAME_NOT_FOUND                         0x00010002
#define DSL_RESULT_COMPONENT_NAME_BAD_FORMAT                        0x00010003
#define DSL_RESULT_COMPONENT_THREW_EXCEPTION                        0x00010004
#define DSL_RESULT_COMPONENT_IN_USE                                 0x00010005
#define DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE                   0x00010006
#define DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH                     0x00010007
#define DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE                   0x00010008
#define DSL_RESULT_COMPONENT_SET_GPUID_FAILED                       0x00010009
#define DSL_RESULT_COMPONENT_SET_NVBUF_MEM_TYPE_FAILED              0x0001000A
#define DSL_RESULT_COMPONENT_GET_QUEUE_PROPERTY_FAILED              0x0001000B
#define DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED              0x0001000C
#define DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED                    0x0001000D
#define DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED                 0x0001000E
#define DSL_RESULT_COMPONENT_ELEMENT_ADD_FAILED                     0x0001000F
#define DSL_RESULT_COMPONENT_ELEMENT_REMOVE_FAILED                  0x00010010
#define DSL_RESULT_COMPONENT_ELEMENT_NOT_IN_USE                     0x00010011
```

## Component Queue Leaky Constants
```C
#define DSL_COMPONENT_QUEUE_LEAKY_NO                                0
#define DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM                          1
#define DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM                        2
```

## Component Queue Units of Measurement
```C
#define DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS                         0
#define DSL_COMPONENT_QUEUE_UNIT_OF_BYTES                           1
#define DSL_COMPONENT_QUEUE_UNIT_OF_TIME                            2
```

## Component Queue Leaky Constants
```C
#define DSL_COMPONENT_QUEUE_LEAKY_NO                                0
#define DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM                          1
#define DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM                        2
```

## Component Queue Units of Measurement
```C
#define DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS                         0
#define DSL_COMPONENT_QUEUE_UNIT_OF_BYTES                           1
#define DSL_COMPONENT_QUEUE_UNIT_OF_TIME                            2
```

## NVIDIA Buffer Memory Types
```C
#define DSL_NVBUF_MEM_TYPE_DEFAULT                                  0
#define DSL_NVBUF_MEM_TYPE_PINNED                                   1
#define DSL_NVBUF_MEM_TYPE_DEVICE                                   2
#define DSL_NVBUF_MEM_TYPE_UNIFIED                                  3
```

---

## Client Callback Typedefs
### *dsl_component_queue_overrun_listener_cb*
```C++
typedef void (*dsl_component_queue_overrun_listener_cb)(const wchar_t* name,
  void* client_data);
```
Callback typedef for a client queue-overrun listener. The callback is registered with a call to [`dsl_component_queue_overrun_listener_add`](#dsl_component_queue_overrun_listener_add). Once added, the callback will be called if the Component's queue buffer becomes full (overrun). A buffer is full if the total amount of data inside it (buffers, bytes, or time) is higher than the max-size values set for each unit. Max-size values can be set by calling [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set).

**Parameters**
* `name` - [in] name of the Component that owns the Queue that has overrun.
* `client_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

<br>

### *dsl_component_queue_underrun_listener_cb*
```C++
typedef void (*dsl_component_queue_underrun_listener_cb)(const wchar_t* name,
  void* client_data);
```
Callback typedef for a client queue-overrun listener. The callback is registered with a call to [`dsl_component_queue_underrun_listener_add`](#dsl_component_queue_underrun_listener_add). Once added, the callback will be called if the Component's queue buffer becomes empty (underrun) A buffer is empty if the total amount of data inside it (buffers, bytes, or time) is less than the min-threshold values set for each unit. Min-threshold values can be set by calling [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set).

**Parameters**
* `name` - [in] name of the Component that owns the Queue that has underrun.
* `client_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add

---

## Constructors
### *dsl_component_custom_new*
```C++
DslReturnType dsl_component_custom_new(const wchar_t* name);
```
This constructor creates a uniquely named Custom Component. Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Custom Component to create.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_component_custom_new('my-custom-component')
```

<br>

### *dsl_component_custom_new_element_add*
```C++
DslReturnType dsl_component_custom_new_element_add(const wchar_t* name,
  const wchar_t* element);
```
This constructor creates a uniquely named Custom Component and adds a [GST Element](/docs/api-gst.md) to it. Construction will fail if the name is currently in use.

**IMPORTAT!** Elements added to Custom Component will be linked in the order they are added.

**Parameters**
* `name` - [in] unique name for the Custom Component to create.
* `element` - [in] unique name of the GST Element to add.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_component_custom_new_element_add('my-custom-component', 'my-element')
```

<br>


### *dsl_component_custom_new_element_add_many*
```C++
DslReturnType dsl_component_custom_new_element_add_many(const wchar_t* name,
  const wchar_t** elements);
```
This constructor creates a uniquely named Custom Component and adds a list of GST Elements to it. Construction will fail if the name is currently in use.

**IMPORTAT!** Elements added to Custom Component will be linked in the order they are added.

**Parameters**
* `name` - [in] unique name for the Custom Component to create.
* `elements` - [in] NULL terminated array of Element names to add.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_component_custom_new_element_add_many('my-component',
  ['my-element-1', 'my-element-2', 'my-element-3', None])
```

---

## Destructors
### *dsl_component_delete*
```c++
DslReturnType dsl_component_delete(const wchar_t* component);
```
This service deletes a single Pipeline Component of any type. The call will fail if the Component is currently `in-use` by a Pipeline.

**Parameters**
* `component` - [in] unique name of the component to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful delete. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_delete('my-uri-source')
```

<br>

### *dsl_component_delete_many*
```c++
DslReturnType dsl_component_delete_many(const wchar_t** component);
```
This service deletes a Null terminated list of named Components of any type. The call will fail if any Component is currently `in-use` by a Pipeline.

**Parameters**
* `components` - [in] Null terminated list of unique component names to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful delete. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_delete_many(['my-uri-source', 'my-primary-gie', 'my-osd', 'my-window-sink', None])
```

<br>

### *dsl_component_delete_all*
```c++
DslReturnType dsl_component_delete_all();
```
This service deletes all Components in memory. The call will fail if any Component is currently `in-use` by a Pipeline.

**Returns**
* `DSL_RESULT_SUCCESS` on successful delete. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_all()
```

<br>

---

## Methods
### *dsl_component_custom_element_add*
```C++
DslReturnType dsl_component_custom_element_add(const wchar_t* name,
   const wchar_t* element);
```
This service adds a single named GST Element to a named Custom Component. The add service will fail if the Element is currently `in-use` by any other Component. The Element's `in-use` state will be set to `true` on successful add.

**IMPORTAT!** Elements added to Custom Component will be linked in the order they are added.

**Parameters**
* `name` - [in] unique name of the Custom Component to update.
* `element` - [in] unique name of the GST Element to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful addition. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval = dsl_component_custom_element_add('my-component', 'my-element')
```

<br>

### *dsl_component_custom_element_add_many*
```C++
DslReturnType dsl_component_custom_element_add_many(const wchar_t* name, const wchar_t** elements);
```
Adds a list of named GST Elements to a named Custom Component. The add service will fail if any of the Elements are currently `in-use` by any other Component. All of the Element's `in-use` state will be set to true on successful add.

**IMPORTAT!** Elements added to Custom Component will be linked in the order they are added.

* `name` - [in] unique name of the Custom Component to update.
* `elements` - [in] a NULL terminated array of uniquely named GST Elements to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful  addition. One of the [Return Values](#return-values) defined above on failure.


**Python Example**
```Python
retval = dsl_component_custom_element_add_many('my-component',
  ['my-element-1', 'my-element-2', None])
```

<br>

---
### *dsl_component_custom_element_remove*
```C++
DslReturnType dsl_component_custom_element_remove(const wchar_t* name, const wchar_t* element);
```
This service removes a single named GST Element from a named Custom Component.

**Parameters**
* `name` - [in] unique name of the Custom Component to update.
* `element` - [in] unique name of the GST Element to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_component_custom_element_remove('my-component', 'my-element')
```

<br>


### *dsl_component_custom_element_remove_many*
```C++
DslReturnType dsl_component_custom_element_remove_many(const wchar_t* name, const wchar_t** elements);
```
This services removes a list of named Elements from a named Custom Component.

* `name` - [in] unique name for the Custom Component to update.
* `elements` - [in] a NULL terminated array of uniquely named GST Elements to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_custom_element_remove_many('my-component',
  ['my-element-1', 'my-element-2', None])
```

<br>

### *dsl_component_queue_current_level_get*
```c++
DslReturnType dsl_component_queue_current_level_get(const wchar_t* name,
  uint unit, uint64_t* current_level);
```
This service gets the queue-current-level by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `current_level` - [out] the current queue level for the specified unit.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, current_level = dsl_component_queue_current_level_get('my-primary-gie')
```

<br>


### *dsl_component_queue_current_level_print*
```c++
DslReturnType dsl_component_queue_current_level_print(const wchar_t* name,
  uint unit);
```
This service prints the queue-current-level by unit (buffers, bytes, or time) to stdout for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_print('my-primary-gie')
```

<br>

### *dsl_component_queue_current_level_print_many*
```c++
DslReturnType dsl_component_queue_current_level_print_many(const wchar_t** names,
  uint unit);
```
This service prints the queue-current-level by unit (buffers, bytes, or time) to stdout for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to query..
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_print_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None])
```

<br>

### *dsl_component_queue_current_level_log*
```c++
DslReturnType dsl_component_queue_current_level_log(const wchar_t* name,
  uint unit);
```
This service logs the queue-current-level by unit (buffers, bytes, or time) at a level of LOG_INFO.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_log('my-primary-gie')
```

<br>

### *dsl_component_queue_current_level_log_many*
```c++
DslReturnType dsl_component_queue_current_level_log_many(const wchar_t** names,
  uint unit);
```
This service logs the queue-current-level by unit (buffers, bytes, or time) at a level of LOG_INFO for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to query..
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_log_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None])
```

<br>

### *dsl_component_queue_leaky_get*
```c++
DslReturnType dsl_component_queue_leaky_get(const wchar_t* name, uint* leaky);
```
This service gets the queue-leaky setting for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `leaky` - [out] one of the [`DSL_COMPONENT_QUEUE_LEAKY`](#component-queue-leaky-constants) constant values. Default = `DSL_COMPONENT_QUEUE_LEAKY_NO`

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, leaky = dsl_component_queue_leaky_get('my-primary-gie')
```

<br>

### *dsl_component_queue_leaky_set*
```c++
DslReturnType dsl_component_queue_leaky_set(const wchar_t* name, uint leaky);
```
This service sets the queue-leaky setting for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `leaky` - [in] one of the [`DSL_COMPONENT_QUEUE_LEAKY`](#component-queue-leaky-constants) constant values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_leaky_set('my-primary-gie',
  DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM)
```

<br>

### *dsl_component_queue_leaky_set_many*
```c++
DslReturnType dsl_component_queue_leaky_set_many(const wchar_t** names, uint leaky);
```
This service sets the queue-leaky setting for a null terminated list of named Components.


**Parameters**
* `names` - [in] null terminated list of names of components to update.
* `leaky` - [in] one of the [`DSL_COMPONENT_QUEUE_LEAKY`](#component-queue-leaky-constants) constant values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_leaky_set(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM)
```

<br>

### *dsl_component_queue_max_size_get*
```c++
DslReturnType dsl_component_queue_max_size_get(const wchar_t* name,
  uint unit, uint64_t* max_size);
```
This service gets the current queue-max-size setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `max_size` - [out] current max-size setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, max_size = dsl_component_queue_max_size_get('my-primary-gie')
```

<br>

### *dsl_component_queue_max_size_set*
```c++
DslReturnType dsl_component_queue_max_size_set(const wchar_t* name,
  uint unit, uint64_t max_size);
```
This service sets the queue-max-size setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `max_size` - [out] new max-size setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_max_size_set('my-primary-gie',
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 10)
```

<br>

### *dsl_component_queue_max_size_set_many*
```c++
DslReturnType dsl_component_queue_max_size_set_many(const wchar_t** names,
  uint unit, uint64_t max_size);
```
This service sets the queue-max-size setting by unit (buffers, bytes, or time) for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `max_size` - [out] new max-size setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_max_size_set_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 10)
```

<br>

### *dsl_component_queue_min_threshold_get*
```c++
DslReturnType dsl_component_queue_min_threshold_get(const wchar_t* name,
  uint unit, uint64_t* min_threshold);
```
This service gets thus current queue-min-threshold setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `min_threshold` - [out] current min-threshold setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, min_threshold = dsl_component_queue_min_threshold_get('my-primary-gie')
```

<br>

### *dsl_component_queue_min_threshold_set*
```c++
DslReturnType dsl_component_queue_min_threshold_set(const wchar_t* name,
  uint unit, uint64_t min_threshold);
```
This service sets the queue-min-threshold setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `min_threshold` - [out] new min-threshold setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_min_threshold_set('my-primary-gie',
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 5)
```

<br>

### *dsl_component_queue_min_threshold_set_many*
```c++
DslReturnType dsl_component_queue_min_threshold_set_many(const wchar_t** names,
  uint unit, uint64_t min_threshold);
```
This service sets the queue-min-threshold setting by unit (buffers, bytes, or time) for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `min_threshold` - [out] new min-threshold setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_min_threshold_set_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 5)
```

<br>

### *dsl_component_queue_overrun_listener_add*
```c++
DslReturnType dsl_component_queue_overrun_listener_add(const wchar_t* name,
  dsl_component_queue_overrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a named Component to be called when the queue's buffer becomes full (overrun). A buffer is full if the total amount of data inside it (buffers, byte or time) is higher than the max-size values set for each unit. Max-size values can be set by calling [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set).

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_overrun_listener_cb(name, client_data):
  print('WARNING queue qverrun occurred for component = ', name)

retval = dsl_component_queue_overrun_listener_add('my-primary-gie',
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_queue_overrun_listener_add_many*
```c++
DslReturnType dsl_component_queue_overrun_listener_add_many(const wchar_t** names,
  dsl_component_queue_overrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a list of named Component to be called when any of the Component queue buffers becomes full (overrun). A buffer is full if the total amount of data inside it (buffers, byte or time) is higher than the max-size values set for each unit. Max-size values can be set by calling [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set).

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_overrun_listener_cb(name, client_data):
  print('WARNING queue overrun occurred for component = ', name)

retval = dsl_component_queue_overrun_listener_add_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_queue_overrun_listener_remove*
```c++
DslReturnType dsl_component_queue_overrun_listener_remove(const wchar_t* name,
  dsl_component_queue_overrun_listener_cb listener);
```
This service removes a queue-client-listener callback function from a named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_overrun_listener_remove('my-primary-gie',
  queue_overrun_listener_cb)
```

<br>

### *dsl_component_queue_overrun_listener_remove_many*
```c++
DslReturnType dsl_component_queue_overrun_listener_remove_many(const wchar_t** names,
  dsl_component_queue_overrun_listener_cb listener, void* client_data);
```
This service removes a queue-client-listener callback function from a list of named Components.

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python

retval = dsl_component_queue_overrun_listener_remove_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_queue_underrun_listener_add*
```c++
DslReturnType dsl_component_queue_underrun_listener_add(const wchar_t* name,
  dsl_component_queue_underrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a named Component to be called when the queue's buffer becomes empty (underrun). A buffer is empty if the total amount of data inside it (buffers, byte or time) is lower than the min-threshold values set for each unit. Min-threshold values can be set by calling [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set).

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to call on Queue underrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_underrun_listener_cb(name, client_data):
  print('INFO queue underrun occurred for component = ', name)

retval = dsl_component_queue_underrun_listener_add('my-primary-gie',
  queue_underrun_listener_cb, None)
```

### *dsl_component_queue_underrun_listener_add_many*
```c++
DslReturnType dsl_component_queue_underrun_listener_add_many(const wchar_t** names,
  dsl_component_queue_underrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a list of named Components to be called when any of the Component queue buffers becomes empty (underrun). A buffer is empty if the total amount of data inside it (buffers, byte or time) is lower than the min-threshold values set for each unit. Min-threshold values can be set by calling [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set).

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to call on Queue underrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_underrun_listener_cb(name, client_data):
  print('INFO queue underrun occurred for component = ', name)

retval = dsl_component_queue_underrun_listener_add_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_underrun_listener_cb, None)
```
<br>

### *dsl_component_queue_underrun_listener_remove*
```c++
DslReturnType dsl_component_queue_underrun_listener_remove(const wchar_t* name,
  dsl_component_queue_underrun_listener_cb listener);
```
This service removes a queue-client-listener callback function from a named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_underrun_listener_remove('my-primary-gie',
  queue_underrun_listener_cb)
```

### *dsl_component_queue_underrun_listener_remove_many*
```c++
DslReturnType dsl_component_queue_underrun_listener_remove_many(const wchar_t** names,
  dsl_component_queue_underrun_listener_cb listener, void* client_data);
```
This service removes a queue-client-listener callback function from a list of named Components.

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python

retval = dsl_component_queue_underrun_listener_remove_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_overrun_listener_cb, None)
```

<br>

---
### *dsl_component_custom_element_remove*
```C++
DslReturnType dsl_component_custom_element_remove(const wchar_t* name, const wchar_t* element);
```
This service removes a single named GST Element from a named Custom Component.

**Parameters**
* `name` - [in] unique name of the Custom Component to update.
* `element` - [in] unique name of the GST Element to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_component_custom_element_remove('my-component', 'my-element')
```

<br>


### *dsl_component_custom_element_remove_many*
```C++
DslReturnType dsl_component_custom_element_remove_many(const wchar_t* name, const wchar_t** elements);
```
This services removes a list of named Elements from a named Custom Component.

* `name` - [in] unique name for the Custom Component to update.
* `elements` - [in] a NULL terminated array of uniquely named GST Elements to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_custom_element_remove_many('my-component',
  ['my-element-1', 'my-element-2', None])
```

<br>

### *dsl_component_queue_current_level_get*
```c++
DslReturnType dsl_component_queue_current_level_get(const wchar_t* name,
  uint unit, uint64_t* current_level);
```
This service gets the queue-current-level by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `current_level` - [out] the current queue level for the specified unit.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, current_level = dsl_component_queue_current_level_get('my-primary-gie')
```

<br>


### *dsl_component_queue_current_level_print*
```c++
DslReturnType dsl_component_queue_current_level_print(const wchar_t* name,
  uint unit);
```
This service prints the queue-current-level by unit (buffers, bytes, or time) to stdout for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_print('my-primary-gie')
```

<br>

### *dsl_component_queue_current_level_print_many*
```c++
DslReturnType dsl_component_queue_current_level_print_many(const wchar_t** names,
  uint unit);
```
This service prints the queue-current-level by unit (buffers, bytes, or time) to stdout for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to query..
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_print_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None])
```

<br>

### *dsl_component_queue_current_level_log*
```c++
DslReturnType dsl_component_queue_current_level_log(const wchar_t* name,
  uint unit);
```
This service logs the queue-current-level by unit (buffers, bytes, or time) at a level of LOG_INFO.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_log('my-primary-gie')
```

<br>

### *dsl_component_queue_current_level_log_many*
```c++
DslReturnType dsl_component_queue_current_level_log_many(const wchar_t** names,
  uint unit);
```
This service logs the queue-current-level by unit (buffers, bytes, or time) at a level of LOG_INFO for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to query..
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_current_level_log_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None])
```

<br>

### *dsl_component_queue_leaky_get*
```c++
DslReturnType dsl_component_queue_leaky_get(const wchar_t* name, uint* leaky);
```
This service gets the queue-leaky setting for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `leaky` - [out] one of the [`DSL_COMPONENT_QUEUE_LEAKY`](#component-queue-leaky-constants) constant values. Default = `DSL_COMPONENT_QUEUE_LEAKY_NO`

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, leaky = dsl_component_queue_leaky_get('my-primary-gie')
```

<br>

### *dsl_component_queue_leaky_set*
```c++
DslReturnType dsl_component_queue_leaky_set(const wchar_t* name, uint leaky);
```
This service sets the queue-leaky setting for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `leaky` - [in] one of the [`DSL_COMPONENT_QUEUE_LEAKY`](#component-queue-leaky-constants) constant values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_leaky_set('my-primary-gie',
  DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM)
```

<br>

### *dsl_component_queue_leaky_set_many*
```c++
DslReturnType dsl_component_queue_leaky_set_many(const wchar_t** names, uint leaky);
```
This service sets the queue-leaky setting for a null terminated list of named Components.


**Parameters**
* `names` - [in] null terminated list of names of components to update.
* `leaky` - [in] one of the [`DSL_COMPONENT_QUEUE_LEAKY`](#component-queue-leaky-constants) constant values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_leaky_set(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM)
```

<br>

### *dsl_component_queue_max_size_get*
```c++
DslReturnType dsl_component_queue_max_size_get(const wchar_t* name,
  uint unit, uint64_t* max_size);
```
This service gets the current queue-max-size setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `max_size` - [out] current max-size setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, max_size = dsl_component_queue_max_size_get('my-primary-gie')
```

<br>

### *dsl_component_queue_max_size_set*
```c++
DslReturnType dsl_component_queue_max_size_set(const wchar_t* name,
  uint unit, uint64_t max_size);
```
This service sets the queue-max-size setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `max_size` - [out] new max-size setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_max_size_set('my-primary-gie',
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 10)
```

<br>

### *dsl_component_queue_max_size_set_many*
```c++
DslReturnType dsl_component_queue_max_size_set_many(const wchar_t** names,
  uint unit, uint64_t max_size);
```
This service sets the queue-max-size setting by unit (buffers, bytes, or time) for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `max_size` - [out] new max-size setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_max_size_set_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 10)
```

<br>

### *dsl_component_queue_min_threshold_get*
```c++
DslReturnType dsl_component_queue_min_threshold_get(const wchar_t* name,
  uint unit, uint64_t* min_threshold);
```
This service gets thus current queue-min-threshold setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to query.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `min_threshold` - [out] current min-threshold setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, min_threshold = dsl_component_queue_min_threshold_get('my-primary-gie')
```

<br>

### *dsl_component_queue_min_threshold_set*
```c++
DslReturnType dsl_component_queue_min_threshold_set(const wchar_t* name,
  uint unit, uint64_t min_threshold);
```
This service sets the queue-min-threshold setting by unit (buffers, bytes, or time) for the named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `min_threshold` - [out] new min-threshold setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_min_threshold_set('my-primary-gie',
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 5)
```

<br>

### *dsl_component_queue_min_threshold_set_many*
```c++
DslReturnType dsl_component_queue_min_threshold_set_many(const wchar_t** names,
  uint unit, uint64_t min_threshold);
```
This service sets the queue-min-threshold setting by unit (buffers, bytes, or time) for a null terminated list of named Components.

**Parameters**
* `names` - [in] null terminated list of names of components to update.
* `unit` - [in] one of the [`DSL_COMPONENT_QUEUE_UNIT_OF`](#component-queue-units-of-measurement) constants
* `min_threshold` - [out] new min-threshold setting for the specified unit. Default values: buffers=200, bytes=10485760, time=1000000000ns

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_min_threshold_set_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, 5)
```

<br>

### *dsl_component_queue_overrun_listener_add*
```c++
DslReturnType dsl_component_queue_overrun_listener_add(const wchar_t* name,
  dsl_component_queue_overrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a named Component to be called when the queue's buffer becomes full (overrun). A buffer is full if the total amount of data inside it (buffers, byte or time) is higher than the max-size values set for each unit. Max-size values can be set by calling [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set).

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_overrun_listener_cb(name, client_data):
  print('WARNING queue qverrun occurred for component = ', name)

retval = dsl_component_queue_overrun_listener_add('my-primary-gie',
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_queue_overrun_listener_add_many*
```c++
DslReturnType dsl_component_queue_overrun_listener_add_many(const wchar_t** names,
  dsl_component_queue_overrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a list of named Component to be called when any of the Component queue buffers becomes full (overrun). A buffer is full if the total amount of data inside it (buffers, byte or time) is higher than the max-size values set for each unit. Max-size values can be set by calling [`dsl_component_queue_max_size_set`](#dsl_component_queue_max_size_set).

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_overrun_listener_cb(name, client_data):
  print('WARNING queue overrun occurred for component = ', name)

retval = dsl_component_queue_overrun_listener_add_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_queue_overrun_listener_remove*
```c++
DslReturnType dsl_component_queue_overrun_listener_remove(const wchar_t* name,
  dsl_component_queue_overrun_listener_cb listener);
```
This service removes a queue-client-listener callback function from a named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_overrun_listener_remove('my-primary-gie',
  queue_overrun_listener_cb)
```

<br>

### *dsl_component_queue_overrun_listener_remove_many*
```c++
DslReturnType dsl_component_queue_overrun_listener_remove_many(const wchar_t** names,
  dsl_component_queue_overrun_listener_cb listener, void* client_data);
```
This service removes a queue-client-listener callback function from a list of named Components.

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_overrun_listener_cb`](#dsl_component_queue_overrun_listener_cb) to call on Queue overrun.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python

retval = dsl_component_queue_overrun_listener_remove_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_queue_underrun_listener_add*
```c++
DslReturnType dsl_component_queue_underrun_listener_add(const wchar_t* name,
  dsl_component_queue_underrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a named Component to be called when the queue's buffer becomes empty (underrun). A buffer is empty if the total amount of data inside it (buffers, byte or time) is lower than the min-threshold values set for each unit. Min-threshold values can be set by calling [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set).

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to call on Queue underrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_underrun_listener_cb(name, client_data):
  print('INFO queue underrun occurred for component = ', name)

retval = dsl_component_queue_underrun_listener_add('my-primary-gie',
  queue_underrun_listener_cb, None)
```

### *dsl_component_queue_underrun_listener_add_many*
```c++
DslReturnType dsl_component_queue_underrun_listener_add_many(const wchar_t** names,
  dsl_component_queue_underrun_listener_cb listener, void* client_data);
```
This service adds a queue-client-listener callback function to a list of named Components to be called when any of the Component queue buffers becomes empty (underrun). A buffer is empty if the total amount of data inside it (buffers, byte or time) is lower than the min-threshold values set for each unit. Min-threshold values can be set by calling [`dsl_component_queue_min_threshold_set`](#dsl_component_queue_min_threshold_set).

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to call on Queue underrun.
* `client_data` - [in] opaque pointer to user data to pass to the listener on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
def queue_underrun_listener_cb(name, client_data):
  print('INFO queue underrun occurred for component = ', name)

retval = dsl_component_queue_underrun_listener_add_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_underrun_listener_cb, None)
```
<br>

### *dsl_component_queue_underrun_listener_remove*
```c++
DslReturnType dsl_component_queue_underrun_listener_remove(const wchar_t* name,
  dsl_component_queue_underrun_listener_cb listener);
```
This service removes a queue-client-listener callback function from a named Component.

**Parameters**
* `name` - [in] unique name of the Component to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_queue_underrun_listener_remove('my-primary-gie',
  queue_underrun_listener_cb)
```

### *dsl_component_queue_underrun_listener_remove_many*
```c++
DslReturnType dsl_component_queue_underrun_listener_remove_many(const wchar_t** names,
  dsl_component_queue_underrun_listener_cb listener, void* client_data);
```
This service removes a queue-client-listener callback function from a list of named Components.

**Parameters**
* `names` - [in] names null terminated list of names of Components to update.
* `listener` - [in] pointer to the client's function of type [`dsl_component_queue_underrun_listener_cb`](#dsl_component_queue_underrun_listener_cb) to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python

retval = dsl_component_queue_underrun_listener_remove_many(
  ['my-primary-gie', 'my-tracker', 'my-tiler', 'my-osd', None],
  queue_overrun_listener_cb, None)
```

<br>

### *dsl_component_gpuid_get*
```c++
DslReturnType dsl_component_gpuid_get(const wchar_t* component, uint* gpuid);
```
This service returns the current GPU ID for the named Component. The default setting for all components is GPU ID = 0.

**Parameters**
* `component` - [in] unique name of the Component to query.
* `gpuid` - [out] current GPU ID in use by the Component.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, gpuid = dsl_component_gpuid_get('my-primary-gie')
```

<br>

### *dsl_component_gpuid_set*
```c++
DslReturnType dsl_component_gpuid_set(const wchar_t* component, uint gpuid);
```
This service sets the current GPU ID for the named Component to use. The call will fail if the Component is currently linked.

**Parameters**
* `component` - [in] unique name of the Component to query.
* `gpuid` - [in] new GPU ID to use by the Component.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_gpuid_set('my-primary-gie', 1)
```

<br>

### *dsl_component_gpuid_set_many*
```c++
DslReturnType dsl_component_gpuid_set_many(const wchar_t** component, uint gpuid);
```
This service sets the GPU ID for a Null terminated list of named components. The call will fail if any Component is currently linked, on first exception.

**Parameters**
* `components` - [in] Null terminated list of unique Component names to update.
* `gpuid` - [in] new GPU ID to use by all named Components.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_gpuid_set_many(['my-uri-source', 'my-primary-gie', 'my-osd', 'my-window-sink', None], 1)
```

<br>

### *dsl_component_nvbuf_mem_type_get*
```c++
DslReturnType dsl_component_nvbuf_mem_type_get(const wchar_t* name,
  uint* type);
```
This service returns the current NVIDIA buffer memory type for the named Component. The default setting for all components that support this property is  = `DSL_NVBUF_MEM_TYPE_DEFAULT`. Refer to the [NVIDIA Reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_FAQ.html#what-are-different-memory-types-supported-on-jetson-and-dgpu) for more information on the memory types supported on Jetson and dGPU.
<<<<<<< HEAD

=======
>>>>>>> b3b683b (Rename/move dsl_gst_bin_* to dsl_component_custom_*)

**Note:** Only Sources, Primary GIEs/TIEs, OSDs, and Window Sinks (on x86_64) support the NVIDIA buffer memory type setting.

**Parameters**
* `component` - [in] unique name of the Component to query.
* `type` - [out] one of the [NVIDIA buffer memory types](nvidia_buffer_memory_types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval, type = dsl_component_nvbuf_mem_type_get('my-primary-gie')
```

<br>

### *dsl_component_nvbuf_mem_type_set*
```c++
DslReturnType dsl_component_nvbuf_mem_type_set(const wchar_t* name,
  uint type);
```
This service sets the current NVIDIA buffer memory type for the named Component to use. The call will fail if the Component is currently linked. Refer to the [NVIDIA Reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_FAQ.html#what-are-different-memory-types-supported-on-jetson-and-dgpu) for more information on the memory types supported on Jetson and dGPU.

**Note:** Only Sources, Primary GIEs/TIEs, OSDs, and Window Sinks (on x86_64) support the NVIDIA buffer memory type setting.

**Parameters**
* `component` - [in] unique name of the Component to update.
* `type` - [in] one of the [NVIDIA buffer memory types](nvidia_buffer_memory_types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above otherwise.

**Python Example**
```Python
retval = dsl_component_nvbuf_mem_type_set('my-primary-gie', DSL_NVBUF_MEM_TYPE_DEVICE)
```

<br>

### *dsl_component_nvbuf_mem_type_set_many*
```c++
DslReturnType dsl_component_nvbuf_mem_type_set_many(const wchar_t** names,
  uint type);
```
This service sets the NVIDIA buffer memory type for a Null terminated list of named components. The call will fail if any Component is currently linked, on first exception. Refer to the [NVIDIA Reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_FAQ.html#what-are-different-memory-types-supported-on-jetson-and-dgpu) for more information on the memory types supported on Jetson and dGPU.

**Note:** Only Sources, Primary GIEs/TIEs, OSDs, and Window Sinks (on x86_64) support the NVIDIA buffer memory type setting.

**Parameters**
* `components` - [in] Null terminated list of unique Component names to update.
* `type` - [in] one of the [NVIDIA buffer memory types](nvidia_buffer_memory_types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_component_nvbuf_mem_type_set_many(
['my-uri-source', 'my-primary-gie', 'my-osd', 'my-window-sink', None], DSL_NVBUF_MEM_TYPE_DEVICE)
```

<br>

### *dsl_component_list_size*
```c++
uint dsl_component_list_size();
```
This service returns the current number of Components (all types) in memory. The number does not include Pipelines.

**Returns**
* The number of Components in memory

**Python Example**
```Python
number_of_components = dsl_component_list_size()
```

<br>

### *dsl_component_list_size*
```c++
uint dsl_component_list_size();
```
This service returns the current number of Components (all types) in memory. The number does not include Pipelines.

**Returns**
* The number of Components in memory

**Python Example**
```Python
number_of_components = dsl_component_list_size()
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter Tees](/docs/api-tee.md)
* [Remuxer](/docs/api-remxer.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* **Component**
* [GST Element](/docs/api-gst.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
