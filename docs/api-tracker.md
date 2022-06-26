# Multi-Object Tracker API Reference
The DeepStream Services Liberary supports Nvidia's three reference low-level trackers (*Note: the below bullets are copied from the Nvidia DeepStream* [*Gst-nvtracker plugin-guide*](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)). 

1. The KLT tracker uses a CPU-based implementation of the Kanade-Lucas-Tomasi (KLT) tracker algorithm. This library requires no configuration file.
2. The Intersection-Over-Union (IOU) tracker uses the IOU values among the detector’s bounding boxes between the two consecutive frames to perform the association between them or assign a new ID. This library takes an optional configuration file.
3. The NVIDIA®-adapted Discriminative Correlation Filter (NvDCF) tracker uses a correlation filter-based online discriminative learning algorithm as a visual object tracker, while using a data association algorithm for multi-object tracking. This library accepts an optional configuration file.

Tracker components are created by calling their type specific constructor, [dsl_tracker_ktl_new](#dsl_tracker_ktl_new), [dsl_tracker_iou_new](#dsl_tracker_iou_new), and [dsl_tracker_dcf_new](#dsl_tracker_dcf_new)

**Important Note: NVIDIA has removed the KLT Tracker in DeepStream 6.0.** Calling [dsl_tracker_ktl_new](#dsl_tracker_ktl_new) will create a a new IOU tracker with default configuration values. This is done to allow the examples and DSL to work with both DeepStream 5 and 6.  The KLT Tracker will be removed from DSL in a future release.

A Tracker is added to a Pipeline by calling [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other components) and removed with [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

The relationship between Pipelines and Trackers is one-to-one. Once added to a Pipeline, a Tracker must be removed before it can used with another. Tracker components are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all). Calling a delete service on a Tracker `in-use` by a Pipeline will fail.

Pipelines with a Tracker component require a [Primary GIE/TIS](/docs/api-infer.md) component in order to Play. 

## Tracker API
**Constructors:**
* [dsl_tracker_ktl_new](#dsl_tracker_ktl_new)
* [dsl_tracker_iou_new](#dsl_tracker_iou_new)
* [dsl_tracker_dcf_new](#dsl_tracker_dcf_new)

**Methods:**
* [dsl_tracker_dimensions_get](#dsl_tracker_dimensions_get)
* [dsl_tracker_dimensions_set](#dsl_tracker_dimensions_set)
* [dsl_tracker_config_file_get](#dsl_tracker_config_file_get)
* [dsl_tracker_config_file_set](#dsl_tracker_config_file_set)
* [dsl_tracker_dcf_batch_processing_enabled_get](#dsl_tracker_dcf_batch_processing_enabled_get)
* [dsl_tracker_dcf_batch_processing_enabled_set](#dsl_tracker_dcf_batch_processing_enabled_set)
* [dsl_tracker_dcf_past_frame_reporting_enabled_get](#dsl_tracker_dcf_past_frame_reporting_enabled_get)
* [dsl_tracker_dcf_past_frame_reporting_enabled_set](#dsl_tracker_dcf_past_frame_reporting_enabled_set)
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
#define DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER                 0x0003000B
```

## Constructors
### *dsl_tracker_ktl_new*
```C++
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint width, uint height);
```
This service creates a uniquely named KTL Tracker component. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the KTL Tracker to create.
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_ktl_new('my-ktl-tracker', 640, 384)
```

<br>

### *dsl_tracker_iou_new*
```C++
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, 
    uint width, uint height);
```
This service creates a uniquely named IOU Tracker component. Construction will fail if the name is currently in use. The `config_file` parameter for the IOU Tracker is optional.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to create.
* `config_file` - [in] relative or absolute pathspec to a valid IOU config text file. Set to NULL or empty string to omit. 
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_iou_new('my-iou-tracker', './test/configs/iou_config.txt', 640, 384)
```

<br>

### *dsl_tracker_dcf_new*
```C++
DslReturnType dsl_tracker_dcf_new(const wchar_t* name, const wchar_t* config_file, 
    uint width, uint height, boolean batch_processing_enabled, boolean past_frame_reporting_enabled);
```
This service creates a uniquely named DCF Tracker component. Construction will fail if the name is currently in use. The `config_file` parameter for the DCF Tracker is optional.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to create.
* `config_file` - [in] relative or absolute pathspec to a valid IOU config text file. Set to NULL or empty string to omit. 
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.
* `batch_processing_enabled` - [in] set to true to enable batch_mode processing, false for single stream mode.
* `past_frame_reporting_enabled` - [in] set to true to enable reporting of past frame data when available, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_dcf_new('my-dcf-tracker', 
    './test/configs/tracker_config.yml', 640, 384, True, False)
```

<br>

## Methods
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

### *dsl_tracker_config_file_get*
```C++
DslReturnType dsl_tracker_config_file_get(const wchar_t* name, const wchar_t** config_file);
```
This service returns the absolute path to the (optional) Tracker Config File in use by the named IOU or DCF Tracker. This service returns an empty string if the configuration file was omitted on construction, or removed by calling [dsl_tracker_config_file_set](#dsl_tracker_config_file_set) with a NULL pointer.

NOTE: Calling this service on a KTL Tracker will return an empty string..

**Parameters**
* `name` - [in] unique name for the IOU Tracker to query.
* `config_file` - [out] absolute pathspec to the IOU config text file in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_tracker_config_file_get('my-iou-tracker')
```

<br>

### *dsl_tracker_config_file_set*
```C++
DslReturnType dsl_tracker_iou_config_file_set(const wchar_t* name, const wchar_t* config_file);
```
This service updates the named IOU or DCF tracker with a new config file to use. 

NOTE: Calling this service on a KTL Tracker will have no affect.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to update.
* `config_file` - [in] absolute pathspec to the IOU config text file in use. Set config_file to NULL or an empty string to clear the optional configuration file setting

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_config_file_set('my-iou-tracker', './test/configs/iou_config.txt')
```

<br>

### *dsl_tracker_dcf_batch_processing_enabled_get*
```C++
DslReturnType dsl_tracker_dcf_batch_processing_enabled_get(const wchar_t* name, 
    boolean* enabled);
```

This service gets the current `enable-batch-process` setting for the named DCF Tracker object. 

**Parameters**
* `name` - [in] unique name of the DCF Tracker to query.
* `enabled` - [out] true if batch processing is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_tracker_dcf_batch_processing_enabled_get('my-tracker')
```

<br>

### *dsl_tracker_dcf_batch_processing_enabled_set*
```C++
DslReturnType dsl_tracker_dcf_batch_processing_enabled_set(const wchar_t* name, 
    boolean enabled);
```
This service sets the `enable-batch-process` setting for the named DCF Tracker object. 

**Parameters**
* `name` - [in] unique name of the DCF Tracker to update.
* `enabled` - [in] set to true to enabled batch processing, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_dcf_batch_processing_enabled_set('my-tracker', True)
```

<br>

### *dsl_tracker_dcf_past_frame_reporting_enabled_get*
```C++
DslReturnType dsl_tracker_dcf_past_frame_reporting_enabled_get(const wchar_t* name, 
    boolean* enabled);
```
This service gets the current `enable-past-frame` setting for the named DCF Tracker object. 

**Parameters**
* `name` - [in] unique name of the DCF Tracker to query.
* `enabled` - [out] true if past frame reporting is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_tracker_dcf_past_frame_reporting_enabled_get('my-tracker')
```

<br>

### *dsl_tracker_dcf_past_frame_reporting_enabled_set*
```C++
DslReturnType dsl_tracker_dcf_past_frame_reporting_enabled_set(const wchar_t* name, 
    boolean enabled);
```
This service sets the `enable-past-frame` setting for the named DCF Tracker object. 

**Parameters**
* `name` - [in] unique name of the DCF Tracker to update.
* `enabled` - [in] set to true to enable "past frame reporting", false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_dcf_past_frame_reporting_enabled_set('my-tracker', True)
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
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

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
