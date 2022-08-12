# Pad Probe Handler API Reference
Data flowing over a Pipeline Component’s Pads – link points between components – can be monitored and updated using a Pad Probe Handler. There are four types of Handlers supported in the current release.
* Custom PPH
* Source Meter PPH
* Object Detection Event PPH
* Non-Maximum Processor PPH

### Custom Pad Probe Handler
The Custom PPH allows the client to add a custom callback function to a Pipeline Component's sink or source pad. The custom callback will be called with each buffer that crosses over the Component's pad.

### Pipeline Meter Pad Probe Handler
The Pipeline Meter PPH measures a Pipeline's throughput in frames-per-second. Adding the Meter to the Tiler's sink-pad -- or any pad after the Stream-muxer and before the Tiler -- will measure all sources. Adding the Meter to the Tiler's source-pad -- or any component downstream of the Tiler -- will measure the throughput of the single tiled stream.

### Object-Detection-Event (ODE) Pad Probe Handler
The ODE PPH manages an ordered collection of [ODE Triggers](/docs/api-ode-trigger.md), each with their own ordered collections of [ODE Actions](/docs/api-ode-action.md) and (optional) [ODE Areas](/docs/api-ode-area.md). The Handler installs a pad-probe callback to handle each GST Buffer flowing over either the Sink (Input) Pad or the Source (output) pad of the named component; a 2D Tiler or On-Screen-Display as examples. The handler extracts the Frame and Object metadata iterating through its collection of ODE Triggers. Triggers, created with specific purpose and criteria, check for the occurrence of specific Object Detection Events (ODEs). On ODE occurrence, the Trigger iterates through its ordered collection of ODE Actions invoking their `handle-ode-occurrence` service. ODE Areas can be added to Triggers as additional criteria for ODE occurrence. Both Actions and Areas can be shared, or co-owned, by multiple Triggers. All options/settings can be updated at runtime while the Pipeline is playing.

### Non-Maximum Processor (NMP) Pad Probe Handler
The NMP PPH implements an inference cluster algorithm providing a more flexible alternative to the default non-maximum suppression (NMS) cluster algorithm performed by the NVIDIA Inference plugin. 

**Important:** The Primary GIE's `cluster-mode` configuration property must be set to `4` to disable the default processing.

See the [NVIDIA Gst-nvinfer plugin documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer) for details on the configuration property and more [cluster mode information](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#cluster-mode-info).

With the default post-processing disabled, the [Primary GIE](/docs/api-infer.md) will add object metadata for every prediction above a specified confidence level producing clusters of overlaping bounding boxes for each individual object. The image below illustrates the differrences in PGIE output between the default and disabled settings.

![gst-infer cluster mode comparison](/Images/gie_cluster_mode_comparison.png)

The PGIE output is post-processed by adding the NMP PPH to either the source pad of the [PGIE](/docs/api-infer.md) or sink pad of the [Multi-object Tracker (MOT)](/docs/api-tracker.md). Once added, the Handler installs a pad-probe callback to process each GST Buffer flowing over the parent component's pad. When processing the metadata from each buffer, the callback's cluster algorithm detects which of the predictions match the same object and which of the matching predictions has the maximum confidence. All non-maximum predictions are either suppressed or merged with the maximum depending on the process method selected.

The NMP PPH supports:
* two methods of non-maximum processing - non-maximum suppression (NMS) and non-maximum merge (NMM).
* two methods of sorting object predictions - class specific and class agnostic
* two methods of matching object predictions - intersection over union (IOU) and intersection over smallest (IOS).
* a definable matching threshold for the IOU/IOS results.

All options/settings can be updated at runtime while the Pipeline is playing.

#### Class Agnostic Non-Maximum Processing
TODO...

#### Input Source Slicing and Non-Maximum Merge.
TODO...

Credit and thanks to [@youngjae-avikus](https://github.com/youngjae-avikus) for developing the cluster algorithm.

**Important:** The Non-Maximum Processor PPH is dependent on third-party source code - [NumCpp: A Templatized Header Only C++ Implementation of the Python NumPy Library](https://github.com/dpilger26/NumCpp). For this reason the NMP PPH is released as an optional build component - disabled/excluded by default. 

Steps to include in DSL:

1. clone the NumCpp repository git clone https://github.com/dpilger26/NumCpp.git
2. set the [makefile](/Makefile) include variable to true - `BUILD_NMP_PPH:=true`
3. set the makefile path variable to the NumCpp `/include` folder - `NUM_CPP_PATH:=<path-to-numcpp-include-folder>`

### Pad Probe Handler Construction and Destruction
Pad Probe Handlers are created by calling their type specific constructor.  Handlers are deleted by calling [dsl_pph_delete](#dsl_pph_delete), [dsl_pph_delete_many](#dsl_pph_delete_many), or [dsl_pph_delete_all](#dsl_pph_delete_all).

The Pad Probe Handler's name must be unique from all other components. The relationship between Pipeline Components and Pad Probe Handlers is one-to-many and a Handler must be removed from a Pipeline/Branch before it can be used with another.

### Adding to a Pipeline Component
Pad Probe handlers are added to a Pipeline Component by calling the component's `dsl_<component>_pph_add` service and removed with `dsl_<component>_pph_remove`. Using the Tracker component for example; [dsl_tracker_pph_add](/docs/api-tracker.md#dsl_tracker_pph_add) and [dsl_tracker_pph_remove](/docs/api-tracker.md#dsl_tracker_pph_remove)

### Adding and Removing Triggers to/from an ODE Pad Probe handler
ODE Triggers are added to an ODE Pad Probe Handler by calling [dsl_pph_ode_trigger_add](#dsl_pph_ode_trigger_add) or [dsl_pph_ode_trigger_add_many](#dsl_pph_ode_trigger_add_many) and removed with [dsl_pph_ode_trigger_remove](#dsl_pph_ode_trigger_remove), [dsl_pph_ode_trigger_remove_many](#dsl_pph_ode_trigger_remove_many), or [dsl_pph_ode_trigger_remove_all](#dsl_pph_ode_trigger_remove_all).

---

## ODE Handler API
**Callback Types:**
* [dsl_pph_custom_client_handler_cb](#dsl_pph_custom_client_handler_cb)
* [dsl_pph_meter_client_handler_cb](#dsl_pph_meter_client_handler_cb)

**Constructors:**
* [dsl_pph_custom_new](#dsl_pph_custom_new)
* [dsl_pph_meter_new](#dsl_pph_meter_new)
* [dsl_pph_ode_new](#dsl_pph_ode_new)
* [dsl_pph_nmp_new](#dsl_pph_nmp_new)

**Destructors:**
* [dsl_pph_delete](#dsl_pph_delete)
* [dsl_pph_delete_many](#dsl_pph_delete_many)
* [dsl_pph_delete_all](#dsl_pph_delete_all)

**Methods:**
* [dsl_pph_meter_interval_get](#dsl_pph_meter_interval_get)
* [dsl_pph_meter_interval_set](#dsl_pph_meter_interval_set)
* [dsl_pph_ode_trigger_add](#dsl_pph_ode_trigger_add)
* [dsl_pph_ode_trigger_add_many](#dsl_pph_ode_trigger_add_many)
* [dsl_pph_ode_trigger_remove](#dsl_pph_ode_trigger_remove)
* [dsl_pph_ode_trigger_remove_many](#dsl_pph_ode_trigger_remove_many)
* [dsl_pph_ode_trigger_remove_all](#dsl_pph_ode_trigger_remove_all)
* [dsl_pph_ode_display_meta_alloc_size_get](#dsl_pph_ode_display_meta_alloc_size_get)
* [dsl_pph_ode_display_meta_alloc_size_set](#dsl_pph_ode_display_meta_alloc_size_set)
* [dsl_pph_nmp_label_file_get](#dsl_pph_nmp_label_file_get)
* [dsl_pph_nmp_label_file_set](#dsl_pph_nmp_label_file_set)
* [dsl_pph_nmp_process_method_get](#dsl_pph_nmp_process_method_get)
* [dsl_pph_nmp_process_method_set](#dsl_pph_nmp_process_method_set)
* [dsl_pph_nmp_match_settings_get](#dsl_pph_nmp_match_settings_get)
* [dsl_pph_nmp_match_settings_set](#dsl_pph_nmp_match_settings_set)
* [dsl_pph_enabled_get](#dsl_pph_enabled_get)
* [dsl_pph_enabled_set](#dsl_pph_enabled_set)
* [dsl_pph_list_size](#dsl_pph_list_size)

## Return Values
The following return codes are used by the Pad Probe Handler API
```C
#define DSL_RESULT_PPH_NAME_NOT_UNIQUE                              0x000D0001
#define DSL_RESULT_PPH_NAME_NOT_FOUND                               0x000D0002
#define DSL_RESULT_PPH_NAME_BAD_FORMAT                              0x000D0003
#define DSL_RESULT_PPH_THREW_EXCEPTION                              0x000D0004
#define DSL_RESULT_PPH_IS_IN_USE                                    0x000D0005
#define DSL_RESULT_PPH_SET_FAILED                                   0x000D0006
#define DSL_RESULT_PPH_ODE_TRIGGER_ADD_FAILED                       0x000D0007
#define DSL_RESULT_PPH_ODE_TRIGGER_REMOVE_FAILED                    0x000D0008
#define DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE                       0x000D0009
#define DSL_RESULT_PPH_METER_INVALID_INTERVAL                       0x0004000A
#define DSL_RESULT_PPH_PAD_TYPE_INVALID                             0x0004000B
```

## Symbolic Constants
The following constants are used by the Non-Maximum Processor (NMP) Pad Probe Handler API
#### Process Methods
```C
#define DSL_NMP_PROCESS_METHOD_SUPRESS                              0
#define DSL_NMP_PROCESS_METHOD_MERGE                                1
```
#### Object Match Determination Methods
```C
#define DSL_NMP_MATCH_METHOD_IOU                                    0
#define DSL_NMP_MATCH_METHOD_IOS                                    1
```

---

## Callback Types
### *dsl_pph_custom_client_handler_cb*
```C
typedef boolean (*dsl_pph_custom_client_handler_cb)(void* buffer, void* client_data);
```

This Type defines a Client Callback function that is added to a Custom Pad Probe Handler during handler construction (see [dsl_pph_custom_new](#dsl_pph_custom_new)). The same function can be added to multiple Custom Pad Probe Handlers.

**Parameters**
* `buffer` - [in] opaque pointer to a batched source buffer.
* `client_data` - [in] opaque pointer to the client's data, provided on Custom PPH construction

**Returns**
* `True` to continue handling Pad Probe buffers, false to stop and remove the Pad Probe Handler from the Pipeline component.

**Python Example**
```Python
def my_custom_pph_callback(buffer, client_data):

    # cast the opaque client data back to a python object and dereference
    my_app_data = cast(client_data, POINTER(py_object)).contents.value
   
    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break

        # do something with the frame meta

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
            except StopIteration:
                break
               
            # do something with the object meta
           
            try:
                l_obj=l_obj.next
            except StopIteration:
                break
     return True
```    

<br>

### *dsl_pph_meter_client_handler_cb*
```c++
typedef boolean (*dsl_pph_meter_client_handler_cb)(double* session_fps_averages, double* interval_fps_averages,
    uint source_count, void* client_data);
```

This Type defines a Client Callback function that is added to a Custom Pad Probe Handler during handler construction (see [dsl_pph_custom_new](#dsl_pph_custom_new)). The same function can be added to multiple Custom Pad Probe Handlers.

**Parameters**
* `session_fps_averages` - [in] array of average frames-per-second measurements for the current session, one per source, specified by `source_count`
* `interval_fps_averages` - [in] array of average frames-per-second measurements for the current interval, one per source, specified by `source_count`
* `source_count` - [in] number of sources - i.e. the number of measurements in each of the arrays
* `client_data` - [in] opaque pointer to the client's data, provided on Custom PPH construction

**Returns**
* `True` to continue handling source meter reports, false to stop and remove the Pad Probe Handler from the Pipeline component.

**Python Example**
```Python
##
# To be used as client_data with our Meter Sink, and passed to our client_calback
##
class ReportData:
  def __init__(self, header_interval):
    self.m_report_count = 0
    self.m_header_interval = header_interval
   
##
# Source Meter client callback funtion
##
def meter_pph_client_callback(session_avgs, interval_avgs, source_count, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    report_data = cast(client_data, POINTER(py_object)).contents.value

    # Print header on interval
    if (report_data.m_report_count % report_data.m_header_interval == 0):
        header = ""
        for source in range(source_count):
            subheader = f"FPS {source} (AVG)"
            header += "{:<15}".format(subheader)
        print()
        print(header)

    # Print FPS counters
    counters = ""
    for source in range(source_count):
        counter = "{:.2f} ({:.2f})".format(interval_avgs[source], session_avgs[source])
        counters += "{:<15}".format(counter)
    print(counters)

    # Increment reporting count
    report_data.m_report_count += 1
   
    return True  
```

---

<br>

## Constructors
### *dsl_pph_custom_new*
```C++
DslReturnType dsl_pph_custom_new(const wchar_t* name,
     dsl_pph_custom_client_handler_cb client_handler, void* client_data);
```
The constructor creates a uniquely named Custom Pad Probe Handler with a client callback function and client data to return on callback,

**Parameters**
* `name` - [in] unique name for the Custom Pad Probe Handler to create.
* `client_handler` - [in] client callback function of type [dsl_pph_custom_client_handler_cb](#dsl_pph_custom_client_handler_cb).
* `client_data` - [in] opaque pointer to the client's data.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_custom_new('my-custom-handler', my_client_callback, my_client_data)
```

<br>

### *dsl_pph_meter_new*
```C++
DslReturnType dsl_pph_meter_new(const wchar_t* name, uint interval,
    dsl_pph_meter_client_handler_cb client_handler, void* client_data);
```
The constructor creates a uniquely named source stream Meter Pad Probe Handler.

**Parameters**
* `name` - [in] unique name for the Meter Pad Probe Handler to create.
* `interval` - [in] interval at which to call the client handler with Meter data in units of seconds.
* `client_handler` - [in] client callback function of type [dsl_pph_meter_client_handler_cb](#dsl_pph_meter_client_handler_cb).
* `client_data` - [] opaque pointer to the client's data.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_meter_new('my-meter-pph', interval=1, client_handler=meter_sink_handler, client_data=report_data)
```

<br>

### *dsl_pph_ode_new*
```C++
DslReturnType dsl_pph_ode_new(const wchar_t* name);
```
The constructor creates a uniquely named Object Detection Event (ODE) Pad Probe Handler. [ODE Triggers](/docs/api-ode-triggers.md) can be added to the ODE Pad Probe Handler prior and after adding the Handler to a Pipeline component

**Parameters**
* `name` - [in] unique name for the ODE Pad Probe Handler to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_new('my-ode-pph')
```
<br>

### *dsl_pph_nmp_new*
```C++
DslReturnType dsl_pph_nmp_new(const wchar_t* name, const wchar_t* label_file,
    uint process_method, uint match_method, float match_threshold);
```
The constructor creates a new, uniquely named Non-Maximum Processor (NMP) Pad Probe Handler (PPH).

**Parameters**
* `name` - [in] unique name for the NMP Pad Probe Handler to create.
* `label_file` - [in] absolute or relative path to inference model label file. Set to NULL to perform class agnostic non maximum processing.
* `process_method` - [in] method of processing non-maximum predictions, either `DSL_NMP_PROCESS_METHOD_SUPRESS` or `DSL_NMP_PROCESS_METHOD_MERGE`.
* `match_method` - [in] method for object match determination, either `DSL_NMP_MATCH_METHOD_IOU` or `DSL_NMP_MATCH_METHOD_IOS`.
* `match_threshold` - [in] IOU or IOS threshold for object match determination, between 0.0 and 1.0.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_new('my-nmp-pph', path_to_label_file,
    DSL_NMP_PROCESS_METHOD_SUPRESS, DSL_NMP_MATCH_METHOD_IOU, 0.5)
```

---

## Destructors
### *dsl_pph_delete*
```C++
DslReturnType dsl_pph_delete(const wchar_t* name);
```
This destructor deletes a single, uniquely named Pad Probe handler. The destructor will fail if the Pad Probe Handler is currently `in-use` by a Pipeline Component.

**Parameters**
* `name` - [in] unique name for the Pad Probe Handler to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_delete('my-ode-handler')
```

<br>

### *dsl_pph_delete_many*
```C++
DslReturnType dsl_pph_delete_many(const wchar_t** names);
```
This destructor deletes a null terminated list of uniquely named Pad Probe Handlers. Each name is checked for existence with the service returning on first failure. The destructor will also fail if one of the Pad Probe Handlers is currently `in-use` by a Pipeline Component.

**Parameters**
* `names` - [in] a NULL terminated array of uniquely named Pad Probe Handlers to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_delete_many(['my-custom-handler', 'my-meter-handler', 'my-ode-handler', None])
```

<br>

### *dsl_pph_delete_all*
```C++
DslReturnType dsl_pph_delete_all();
```
This destructor deletes all Pad Probe Handlers currently in memory. The destructor will fail if any one of the Pad Probe Handlers is currently `in-use` by a Pipeline Component.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_delete_all()
```

<br>

---

## Methods
### *dsl_pph_meter_interval_get*
```c++
DslReturnType dsl_pph_meter_interval_get(const wchar_t* name, uint* interval);
```

This service gets the current reporting interval - the interval at which results are reported by callback -- for the named Source Meter Pad Probe Handler.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `interval` - [out] reporting interval in seconds.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, interval = dsl_pph_meter_interval_get('my-meter')
```

<br>

### *dsl_pph_meter_interval_set*
```c++
DslReturnType dsl_pph_meter_interval_set(const wchar_t* name, uint* interval);
```

This service sets the current reporting interval - the interval at which results are reported by callback -- for the named Source Meter Pad Probe Handler. This service will fail if the Meter is currently enabled. Disable by calling [dsl_pph_disable](#dsl_pph_disable) and then re-enable with [dsl_pph_enable](#dsl_pph_enable).

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `interval` - [in] reporting interval in seconds.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_meter_interval_set('my-meter', 2)
```

<br>

### *dsl_pph_ode_trigger_add*
```c++
DslReturnType dsl_pph_ode_trigger_add(const wchar_t* name, const wchar_t* trigger);
```

This service adds a named ODE Trigger to a named ODE Handler. The relationship between Handler and Trigger is one-to-many.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `trigger` - [in] unique name of the ODE Trigger to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_trigger_add('my-handler', 'my-trigger')
```

<br>

### *dsl_pph_ode_trigger_add_many*
```c++
DslReturnType dsl_pph_ode_trigger_add_many(const wchar_t* name, const wchar_t** triggers);
```

This service adds a Null terminated list of named ODE Triggers to a named ODE Pad Probe Handler. The relationship between Handler and Trigger is one-to-many.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `triggers` - [in] a Null terminated list of unique names of the ODE Triggers to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_trigger_add_many('my-handler', ['my-trigger-a', 'my-trigger-b', 'my-trigger-c', None])
```

<br>

### *dsl_pph_ode_trigger_remove*
```c++
DslReturnType dsl_pph_ode_trigger_remove(const wchar_t* name, const wchar_t* trigger);
```

This service removes a named ODE Trigger from a named ODE Pad Probe Handler. The services will fail if the Trigger is not currently in-use by the named Handler.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `Trigger` - [in] unique name of the ODE Trigger to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_trigger_remove('my-handler', 'my-trigger')
```

<br>

### *dsl_pph_ode_trigger_remove_many*
```c++
DslReturnType dsl_pph_ode_trigger_remove_many(const wchar_t* name, const wchar_t** triggers);
```

This service removes a Null terminated list of named ODE Triggers from a named ODE Pad Probe Handler. The service will fail if any one of the named Triggers is not currently in-use by the named Handler.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `triggers` - [in] a Null terminated list of unique names of the ODE Triggers to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_trigger_remove_many('my-handler', ['my-trigger-a', 'my-trigger-b', 'my-trigger-c', None])
```

<br>

### *dsl_pph_ode_trigger_remove_all*
```c++
DslReturnType dsl_pph_ode_trigger_remove_all(const wchar_t* name);
```

This service removes all ODE Triggers from a named ODE Pad Probe Handler.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_trigger_remove_all('my-handler')
```

<br>

### *dsl_pph_ode_display_meta_alloc_size_get*
```c++
DslReturnType dsl_pph_ode_display_meta_alloc_size_get(const wchar_t* name, uint* size);
```

This service gets the current setting for the number of Display Meta structures that are allocated for each frame. Each structure can hold up to 16 display elements for each display type (lines, arrows, rectangles, etc.). The default size is one. Note: each allocation adds overhead to the processing of each frame.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to query.
* `size` - [out] current allocation size = number of structures allocated per frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, size = dsl_pph_ode_display_meta_alloc_size_get('my-handler')
```

<br>

### *dsl_pph_ode_display_meta_alloc_size_set*
```c++
DslReturnType dsl_pph_ode_display_meta_alloc_size_set(const wchar_t* name, uint size);
```

This service sets the setting for the number of Display Meta structures that are allocated for each frame. Each structure can hold up to 16 display elements for each display type (lines, arrows, rectangles, etc.). The default size is one. Note: each allocation adds overhead to the processing of each frame.

**Parameters**
* `name` - [in] unique name of the ODE Pad Probe Handler to update.
* `size` - [in] new allocation size = number of structures allocated per frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_ode_display_meta_alloc_size_set('my-handler', 3)
```

<br>

### *dsl_pph_nmp_label_file_get*
```c++
DslReturnType dsl_pph_nmp_label_file_get(const wchar_t* name,
     const wchar_t** label_file);
```

This service gets the current inference model class label file in use by the Non-Maximum Processor (NMP) Pad Probe Handler component.  

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to query.
* `label_file` - [out] path to the inference model label file in use. NULL indicates class agnostic non maximum processing.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, label_file = dsl_pph_nmp_label_file_get('my-nmp-pph')
```

<br>

### *dsl_pph_nmp_label_file_set*
```c++
DslReturnType dsl_pph_nmp_label_file_set(const wchar_t* name,
     const wchar_t* label_file);
```

This service sets the inference model label file for the Non-Maximum Processor (NMP) Pad Probe Handler to use.

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to update.
* `label_file` - [in] absolute or relative path to the inference model label file to use. Set to NULL to perform class agnostic non maximum processing.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_nmp_label_file_set('my-nmp-pph', path_to_label_file)
```

<br>

### *dsl_pph_nmp_process_method_get*
```c++
DslReturnType dsl_pph_nmp_process_method_get(const wchar_t* name,
     uint* process_method);
```

This service gets the current process mode in use by the Non-Maximum Processor (NMP) Pad Probe Handler.

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to query.
* `process_method` - [out] current method of processing non-maximum predictions. One of `DSL_NMP_PROCESS_METHOD_SUPRESS` or `DSL_NMP_PROCESS_METHOD_MERGE`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, process_method = dsl_pph_nmp_process_method_get('my-nmp-pph')
```

<br>

### *dsl_pph_nmp_process_method_set*
```c++
DslReturnType dsl_pph_nmp_process_method_set(const wchar_t* name,
     uint process_method);
```

This service sets the process mode for the Non-Maximum Processor (NMP) Pad Probe Handler to use.  

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to update.
* `process_method` - [in] new method of processing non-maximum predictions. One of `DSL_NMP_PROCESS_METHOD_SUPRESS` or `DSL_NMP_PROCESS_METHOD_MERGE`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_nmp_process_method_set('my-nmp-pph', DSL_NMP_PROCESS_METHOD_MERGE)
```

<br>

### *dsl_pph_enabled_get*
```c++
DslReturnType dsl_pph_enabled_get(const wchar_t* name, boolean* enabled);
```

This service returns the current enabled setting for the named Pad Probe Handler. Note: Handlers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the Pad Probe Handler to query.
* `enabled` - [out] true if the Pad Probe Handler is currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_pph_enabled_get('my-handler')
```

<br>

### *dsl_pph_nmp_process_method_get*
```c++
DslReturnType dsl_pph_nmp_process_method_get(const wchar_t* name,
     uint* process_method);
```

This service gets the current process mode in use by the Non-Maximum Processor (NMP) Pad Probe Handler.

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to query.
* `process_method` - [out] current method of processing non-maximum predictions. One of `DSL_NMP_PROCESS_METHOD_SUPRESS` or `DSL_NMP_PROCESS_METHOD_MERGE`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, process_method = dsl_pph_nmp_process_method_get('my-nmp-pph')
```

<br>

### *dsl_pph_nmp_process_method_set*
```c++
DslReturnType dsl_pph_nmp_process_method_set(const wchar_t* name,
     uint process_method);
```

This service sets the process mode for the Non-Maximum Processor (NMP) Pad Probe Handler to use.  

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to update.
* `process_method` - [in] new method of processing non-maximum predictions. One of `DSL_NMP_PROCESS_METHOD_SUPRESS` or `DSL_NMP_PROCESS_METHOD_MERGE`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_nmp_process_method_set('my-nmp-pph', DSL_NMP_PROCESS_METHOD_MERGE)
```

<br>

### *dsl_pph_nmp_match_settings_get*
```c++
DslReturnType dsl_pph_nmp_match_settings_get(const wchar_t* name,
    uint* match_method, float* match_threshold);
```

This service gets the current match settings in use by the named Non-Maximum Processor (NMP) Pad Probe Handler.

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to query.
* `process_method` - [out] current method of object match determination, either `DSL_NMP_MATCH_METHOD_IOU` or `DSL_NMP_MATCH_METHOD_IOS`.
* `match_threshold` - [out] current threshold for object match determination currently in use, between 0.0 and 1.0.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, match_method, match_threshold = dsl_pph_nmp_match_settings_get('my-nmp-pph')
```

<br>

### *dsl_pph_nmp_match_settings_set*
```c++
DslReturnType dsl_pph_nmp_match_settings_set(const wchar_t* name,
    uint match_method, float match_threshold);
```

This service sets the match settings for the named Non-Maximum Processor (NMP) Pad Probe Handler to use.

**Parameters**
* `name` - [in] unique name of the NMP Pad Probe Handler to update.
* `match_method` - [in] new method for object match determination, either `DSL_NMP_MATCH_METHOD_IOU` or `DSL_NMP_MATCH_METHOD_IOS` `    .
* `match_threshold` - [in] new threshold for object match determination.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pph_nmp_match_settings_set('my-nmp-pph', DSL_NMP_MATCH_METHOD_IOU, 0.5)
```

<br>

### *dsl_pph_enabled_get*
```c++
DslReturnType dsl_pph_enabled_get(const wchar_t* name, boolean* enabled);
```

This service returns the current enabled setting for the named Pad Probe Handler. Note: Handlers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the Pad Probe Handler to query.
* `enabled` - [out] true if the Pad Probe Handler is currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_pph_enabled_get('my-handler')
```

<br>

### *dsl_pph_enabled_set*
```c++
DslReturnType dsl_pph_enabled_set(const wchar_t* name, boolean enabled);
```

This service sets the enabled setting for the named Pad Probe Handler. Note: Handlers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the Pad Probe Handler to update.
* `enabled` - [in] set to true to enable the Pad Probe Handler, false to disable

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_pph_enabled_set('my-handler', False)
```

<br>


### *dsl_pph_list_size*
```C++
uint dsl_pph_list_size();
```
This service returns the size of the current list of Pad Probe Handlers in memory

**Returns**
* The number of Pad Probe Handlers currently in memory

**Python Example**
```Python
pph_count = dsl_pph_list_size()
```

<br>---

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
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* **Pad Probe Handler**
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-types)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
