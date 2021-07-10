# Multi-Object Tracker API Reference
KTL and IOU Tracker components are created by calling their type specific constructor, [dsl_tracker_ktl_new](#dsl_tracker_ktl_new) and [dsl_tracker_iou_new](#dsl_tracker_iou_new)

A Tracker is added to a Pipeline by calling [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other compnents) and removed with [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

The relationship between Pipelines and Trackers is one-to-one. Once added to a Pipeline, a Tracker must be removed before it can used with another. Tracker components are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all). Calling a delete service on a Tracker `in-use` by a Pipeline will fail.

Pipelines with a Tracker component requirie a [Primary GIE](/docs/api-gie.md) component in order to Play. 

## Tracker API
**Constructors:**
* [dsl_tracker_ktl_new](#dsl_tracker_ktl_new)
* [dsl_tracker_iou_new](#dsl_tracker_iou_new)

**Methods:**
* [dsl_tracker_dimensions_get](#dsl_tracker_dimensions_get)
* [dsl_tracker_dimensions_set](#dsl_tracker_dimensions_set)
* [dsl_tracker_iou_config_file_get](#dsl_tracker_iou_config_file_get)
* [dsl_tracker_iou_config_file_set](#dsl_tracker_iou_config_file_set)
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
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint max_width, uint max_height);
```
This service creates a unqiuely named KTL Tracker component. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the KTL Tracker to create.
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_ktl_new('my-ktl-tracker', 640, 368)
```

<br>

### *dsl_tracker_iou_new*
```C++
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, 
    uint width, uint height);
```
This service creates a unqiuely named IOU Tracker component. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the IOU Tracker to create.
* `config_file` - [in] relative or absolute pathspec to a valid IOU config text file
* `width` - [in] Frame width at which the tracker is to operate, in pixels.
* `height` - [in] Frame height at which the tracker is to operate, in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_iou_new('my-', './test/configs/iou_config.txt', 368, 368)
```

<br>

## Methods
### *dsl_tracker_dimensions_get*
```C++
DslReturnType dsl_tracker_dimensions_get(const wchar_t* name, uint* width, uint* height);
```

This service returns the max input frame dimensions to track on in use by the named Tracker.

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
This Service sets the max input frame dimensions for the name Tracker.

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

### *dsl_tracker_iou_config_file_get*
```C++
DslReturnType dsl_tracker_iou_config_file_get(const wchar_t* name, const wchar_t** config_file);
```
This service returns the absolute path to the IOU Tracker Config File in use by the named Tracker.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to query.
* `config_file` - [out] absolute pathspec to the IOU config text file in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_tracker_iou_config_file_get('my-iou-tracker')
```

<br>

### *dsl_tracker_iou_config_file_set*
```C++
DslReturnType dsl_tracker_iou_config_file_set(const wchar_t* name, const wchar_t* config_file);
```
This service updates the named IOU tracker with a new config file to use.

**Parameters**
* `name` - [in] unique name for the IOU Tracker to update.
* `config_file` - [in] absolute pathspec to the IOU config text file in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_iou_config_file_set('my-iou-tracker', './test/configs/iou_config.txt')
```

<br>

### *dsl_tracker_pph_add*
```C++
DslReturnType dsl_tracker_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named of a Tracker.

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
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named On-Screen Display. The service will fail if the named handler is not owned by the Tiler

**Parameters**
* `name` - [in] unique name of the On-Screen Display to update.
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
* [Inference Engine and Server](/docs/api-infer.md)
* **Tracker**
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
