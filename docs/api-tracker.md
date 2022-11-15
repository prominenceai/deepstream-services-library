# Multi-Object Tracker API Reference
The DeepStream Services Library (DSL) supports Nvidia's three reference low-level trackers (*Note: the below bullets are copied from the Nvidia DeepStream* [*Gst-nvtracker plugin-guide*](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)).

1. **NvDCF:** The NvDCF tracker is an NVIDIA®-adapted Discriminative Correlation Filter (DCF) tracker that uses a correlation filter-based online discriminative learning algorithm for visual object tracking capability, while using a data association algorithm and a state estimator for multi-object tracking.
2. **DeepSORT:** The DeepSORT tracker is a re-implementation of the official DeepSORT tracker, which uses the deep cosine metric learning with a Re-ID neural network. This implementation allows users to use any Re-ID network as long as it is supported by NVIDIA’s TensorRT™ framework.
3. **IOU Tracker:** The Intersection-Over-Union (IOU) tracker uses the IOU values among the detector’s bounding boxes between the two consecutive frames to perform the association between them or assign a new target ID if no match found. This tracker includes a logic to handle false positives and false negatives from the object detector; however, this can be considered as the bare-minimum object tracker, which may serve as a baseline only.

The three reference implementations are provided in a single low-level library `libnvds_nvmultiobjecttracker.so` which is specified as the default library for the DSL Tracker to use as defined in the Makefile. The default library path can be updated -- by updating the Makefile or by calling [dsl_tracker_lib_file_set](#dsl_tracker_lib_file_set) -- to reference a custom library that implements the [NvDsTracker API](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#how-to-implement-a-custom-low-level-tracker-library).

**Important!** The DeepSORT tracker requires additional installation and setup steps which can be found in the **README** file located under
```bash
/opt/nvidia/deepstream/deepstream/sources/tracker_DeepSORT
```

## Construction and Destruction
A Tracker component is created by calling [dsl_tracker_new](#dsl_tracker_new) with a type specific configuration file.

**Important!** NVIDIA® provides reference configuration files for the three Tracker implementations under
```bash
/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/
```

Tracker components are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all). Calling a delete service on a Tracker `in-use` by a Pipeline will fail.


## Adding and Removing
The relationship between Pipelines/Branches and Trackers is one-to-one. Once added to a Pipeline or Branch, a Tracker must be removed before it can be used with another.
A Tracker is added to a Pipeline by calling [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other components) and removed with [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

A similar set of Services are used when adding/removing a Tracker to/from a branch: [dsl_branch_component_add](api-branch.md#dsl_branch_component_add), [dsl_branch_component_add_many](/docs/api-branch.md#dsl_branch_component_add_many), [dsl_branch_component_remove](/docs/api-branch.md#dsl_branch_component_remove), [dsl_branch_component_remove_many](/docs/api-branch.md#dsl_branch_component_remove_many), and [dsl_branch_component_remove_all](/docs/api-branch.md#dsl_branch_component_remove_all).

Pipelines with a Tracker component require a [Primary GIE/TIS](/docs/api-infer.md) component in order to Play.

## Tracker API
**Constructors:**
* [dsl_tracker_new](#dsl_tracker_new)

**Methods:**
* [dsl_tracker_lib_file_get](#dsl_tracker_lib_file_get)
* [dsl_tracker_lib_file_set](#dsl_tracker_lib_file_set)
* [dsl_tracker_config_file_get](#dsl_tracker_config_file_get)
* [dsl_tracker_config_file_set](#dsl_tracker_config_file_set)
* [dsl_tracker_dimensions_get](#dsl_tracker_dimensions_get)
* [dsl_tracker_dimensions_set](#dsl_tracker_dimensions_set)
* [dsl_tracker_batch_processing_enabled_get](#dsl_tracker_batch_processing_enabled_get)
* [dsl_tracker_batch_processing_enabled_set](#dsl_tracker_batch_processing_enabled_set)
* [dsl_tracker_past_frame_reporting_enabled_get](#dsl_tracker_past_frame_reporting_enabled_get)
* [dsl_tracker_past_frame_reporting_enabled_set](#dsl_tracker_past_frame_reporting_enabled_set)
* [dsl_tracker_pph_add](#dsl_tracker_pph_add)
* [dsl_tracker_pph_remove](#dsl_tracker_pph_remove)

<br>

## Return Values
The following return codes are used specifically by the Tracker API
```C++
#define DSL_RESULT_TRACKER_NAME_NOT_UNIQUE                          0x00030001
#define DSL_RESULT_TRACKER_NAME_NOT_FOUND                           0x00030002
#define DSL_RESULT_TRACKER_NAME_BAD_FORMAT                          0x00030003
#define DSL_RESULT_TRACKER_THREW_EXCEPTION                          0x00030004
#define DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND                    0x00030005
#define DSL_RESULT_TRACKER_IS_IN_USE                                0x00030006
#define DSL_RESULT_TRACKER_SET_FAILED                               0x00030007
#define DSL_RESULT_TRACKER_HANDLER_ADD_FAILED                       0x00030008
#define DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED                    0x00030009
#define DSL_RESULT_TRACKER_PAD_TYPE_INVALID                         0x0003000A
```

## Constructors
### *dsl_tracker_new*
```C++
DslReturnType dsl_tracker_new(const wchar_t* name, const wchar_t* config_file,
    uint width, uint height);
```
This service creates a uniquely named Tracker component using the default NVIDIA NvMultiObjectTracker low-level library. Construction will fail if the name is currently in use. The `config_file` parameter is optional. If not specified, the low-level library will proceed with default IOU settings.

Note a custom implementation of the NvDsTracker API can be used by setting the Tracker's low-level library by calling [dsl_tracker_lib_file_set](#dsl_tracker_lib_file_set).

**Parameters**
* `name` - [in] unique name for the IOU Tracker to create.
* `config_file` - [in] relative or absolute pathspec to a valid config text file. Set to NULL or empty string to omit.
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_new('my-iou-tracker', './test/configs/iou_config.txt', 640, 384)
```

<br>

## Methods
### *dsl_tracker_lib_file_get*
```C++
DslReturnType dsl_tracker_lib_file_get(const wchar_t* name,
    const wchar_t** lib_file);
```
This service returns the absolute path to the low-level library in use by the named Tracker.

**Important** the default path to low-level library is defined in the Makefile as `$(LIB_INSTALL_DIR)/libnvds_nvmultiobjecttracker.so`

**Parameters**
* `name` - [in] unique name for the Tracker to query.
* `lib_file` - [out] absolute pathspec to the low-level library in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, lib_file = dsl_tracker_lib_file_get('my-tracker')
```

<br>

### *dsl_tracker_lib_file_set*
```C++
DslReturnType dsl_tracker_lib_file_set(const wchar_t* name,
    const wchar_t* lib_file);
```
This service updates the named Tracker with a new low-level library to use.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to update.
* `lib_file` - [in] absolute or relative pathspec to the new low-level library to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_lib_file_set('my-tracker', path_to_ll_lib)
```

<br>

### *dsl_tracker_config_file_get*
```C++
DslReturnType dsl_tracker_config_file_get(const wchar_t* name,
    const wchar_t** config_file);
```
This service returns the absolute path to the (optional) Tracker Config File in use by the named Tracker. This service returns an empty string if the configuration file was omitted on construction, or removed by calling [dsl_tracker_config_file_set](#dsl_tracker_config_file_set) with a NULL pointer.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to query.
* `config_file` - [out] absolute pathspec to the IOU config text file in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_tracker_config_file_get('my-tracker')
```

<br>

### *dsl_tracker_config_file_set*
```C++
DslReturnType dsl_tracker_iou_config_file_set(const wchar_t* name,
    const wchar_t* config_file);
```
This service updates the named Tracker with a new config file to use.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to update.
* `config_file` - [in] absolute pathspec to the IOU config text file in use. Set config_file to NULL to clear the optional configuration file setting.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_config_file_set('my-tracker', './test/configs/iou_config.txt')
```

<br>

### *dsl_tracker_dimensions_get*
```C++
DslReturnType dsl_tracker_dimensions_get(const wchar_t* name, uint* width, uint* height);
```

This service returns the operational dimensions in use by the named Tracker.

**Parameters**
* `name` - [in] unique name of the Tracker to query.
* `width` - [out] Current frame width at which the tracker is to operate, in pixels.
* `height` - [out] Current frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_tracker_dimensions_get('my-tracker')
```

<br>

### *dsl_tracker_dimensions_set*
```C++
DslReturnType dsl_tracker_dimensions_set(const wchar_t* name, uint max_width, uint max_height);
```
This Service sets the operational dimensions for the name Tracker.

**Parameters**
* `name` - [in] unique name of the Tracker to update.
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_dimensions_set('my-tracker', 640, 368)
```

<br>

### *dsl_tracker_batch_processing_enabled_get*
```C++
DslReturnType dsl_tracker_batch_processing_enabled_get(const wchar_t* name,
    boolean* enabled);
```

This service gets the current `enable-batch-process` setting for the named Tracker. The Tracker's low-level library must support batch-processing to use this setting. The NVIDIA reference low-level trackers enable this setting by default.

**Parameters**
* `name` - [in] unique name of the Tracker to query.
* `enabled` - [out] true if batch processing is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_tracker_batch_processing_enabled_get('my-tracker')
```

<br>

### *dsl_tracker_batch_processing_enabled_set*
```C++
DslReturnType dsl_tracker_batch_processing_enabled_set(const wchar_t* name,
    boolean enabled);
```
This service sets the `enable-batch-process` setting for the named Tracker object.  The Tracker's low-level library must support batch-processing to use this setting.

**Parameters**
* `name` - [in] unique name of the Tracker to update.
* `enabled` - [in] set to true to enabled batch processing, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_batch_processing_enabled_set('my-tracker', True)
```

<br>

### *dsl_tracker_past_frame_reporting_enabled_get*
```C++
DslReturnType dsl_tracker_past_frame_reporting_enabled_get(const wchar_t* name,
    boolean* enabled);
```
This service gets the current `enable-past-frame` setting for the named Tracker object. The Tracker's low-level library must support past-frame-reporting to use this setting.  If the past-frame data is retrieved from the low-level tracker, it will be reported as a user-meta, called `NvDsPastFrameObjBatch`.  See the [NVIDIA Tracker reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#gst-nvtracker) for more information.

**Parameters**
* `name` - [in] unique name of the Tracker to query.
* `enabled` - [out] true if past frame reporting is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_tracker_past_frame_reporting_enabled_get('my-tracker')
```

<br>

### *dsl_tracker_past_frame_reporting_enabled_set*
```C++
DslReturnType dsl_tracker_past_frame_reporting_enabled_set(const wchar_t* name,
    boolean enabled);
```
This service sets the `enable-past-frame` setting for the named Tracker object. The Tracker's low-level library must support past-frame-reporting to use this setting.  If the past-frame data is retrieved from the low-level tracker, it will be reported as a user-meta, called `NvDsPastFrameObjBatch`.  See the [NVIDIA Tracker reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#gst-nvtracker) for more information.

**Parameters**
* `name` - [in] unique name of the Tracker to update.
* `enabled` - [in] set to true to enable "past frame reporting", false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_past_frame_reporting_enabled_set('my-tracker', True)
```

<br>


### *dsl_tracker_pph_add*
```C++
DslReturnType dsl_tracker_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named Tracker.

**Parameters**
* `name` - [in] unique name of the Tracker to update.
* `handler` - [in] unique name of Pad Probe Handler to add
* `pad` - [in] to which of the two pads to add the handler: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tracker_pph_add('my-tracker', 'my-pph-handler', `DSL_PAD_SINK`)
```

<br>

### *dsl_tracker_pph_remove*
```C++
DslReturnType dsl_tracker_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named Tracker. The service will fail if the named handler is not owned by the Tracker

**Parameters**
* `name` - [in] unique name of the Tracker to update.
* `handler` - [in] unique name of Pad Probe Handler to remove
* `pad` - [in] to which of the two pads to remove the handler from: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tracker_pph_remove('my-tracker', 'my-pph-handler', `DSL_PAD_SINK`)
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
* **Tracker**
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)

