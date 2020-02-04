# Multi-Object Tracker API Reference

A Tracker is added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other compnents) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all]((api-pipeline.md#dsl_pipeline_component_remove_all).

The relationship between Pipelines and Trackers is one-to-one. Once added to a Pipeline, a Tracker must be removed before it can used with another. Trakers are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all). Calling a delete service on a Tracker `in-use` by a Pipeline will fail.

## Tracker API
* [dsl_tracker_ktl_new](#dsl_tracker_ktl_new)
* [dsl_tracker_iou_new](#dsl_tracker_iou_new)
* [dsl_tracker_max_dimensions_get](#dsl_tracker_dimensions_get)
* [dsl_tracker_max_dimensions_set](#dsl_tracker_dimensions_set)
* [dsl_tracker_iou_config_file_get](#dsl_tracker_iou_config_file_get)
* [dsl_tracker_iou_config_file_set](#dsl_tracker_iou_config_file_set)
* [dsl_tracker_meta_batch_handler_add](#dsl_tracker_meta_batch_handler_add)
* [dsl_tracker_meta_batch_handler_remove](#dsl_tracker_meta_batch_handler_remove)

<br>

## Return Values
The following return codes are used specifically by the Tracker API
```C++
#define DSL_RESULT_TRACKER_NAME_NOT_UNIQUE                          0x00030001
#define DSL_RESULT_TRACKER_NAME_NOT_FOUND                           0x00030002
#define DSL_RESULT_TRACKER_NAME_BAD_FORMAT                          0x00030003
#define DSL_RESULT_TRACKER_THREW_EXCEPTION                          0x00030004
#define DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND                    0x00030005
#define DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID                   0x00030006
#define DSL_RESULT_TRACKER_IS_IN_USE                                0x00030007
#define DSL_RESULT_TRACKER_SET_FAILED                               0x00030008
#define DSL_RESULT_TRACKER_HANDLER_ADD_FAILED                       0x00030009
#define DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED                    0x0003000A
#define DSL_RESULT_TRACKER_PAD_TYPE_INVALID                         0x0003000B
#define DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER                 0x0003000C
```

## Constructors
### *dsl_tracker_ktl_new*
```C++
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint max_width, uint max_height);
```
**Parameters**
* `name` - [in] unique name for the KTL Tracker to create.
* `max_width` - [in] maximum width of each frame for the input transform
* `max_height` - [in] maximum height of each frame for the input transform

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_ktl_new('my-ktl-tracker', 480, 270)
```

<br>

### *dsl_tracker_iou_new*
```C++
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, 
    uint width, uint height);
```
**Parameters**
* `name` - [in] unique name for the IOU Tracker to create.
* `config_file` - [in] relative or absolute pathspec to a valid IOU config text file
* `max_width` - [in] maximum width of each frame for the input transform
* `max_height` - [in] maximum height of each frame for the input transform

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_iou_new('my-', './test/configs/iou_config.txt', 480, 270)
```

<br>

## Methods
### *dsl_tracker_dimensions_get*
```C++
DslReturnType dsl_tracker_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
**Parameters**
* `name` - [in] unique name of the Tracker to query.
* `max_width` - [out] current maximum width of each frame for the input transform
* `max_height` - [out] current maximum height of each frame for the input transform

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, max_width, max_height = dsl_tracker_dimensions_get('my-tracker')
```

<br>

### *dsl_tracker_dimensions_set*
```C++
DslReturnType dsl_tracker_dimensions_set(const wchar_t* name, uint width, uint height);
```
**Parameters**
* `name` - unique name of the Tracker to update.
* `max_width` - new maximum width of each frame for the input transform.
* `max_height` - new maximum height of each frame for the input transform.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_dimensions_set('my-tracker', 480, 270)
```

<br>

### *dsl_tracker_iou_config_file_get*
```C++
DslReturnType dsl_tracker_iou_config_file_get(const wchar_t* name, const wchar_t** config_file);
```
This service return the absolute path to the IOU Tracker Config File in use

**Parameters**
* `name` - [in] unique name for the IOU Tracker to query.
* `config_file` - [out] absolute pathspec to the IOU config text file in use.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_tracker_iou_config_file_get('my-iou-tracker')
```

<br>

### *dsl_tracker_iou_config_file_set*
```C++
DslReturnType dsl_tracker_iou_config_file_set(const wchar_t* name, const wchar_t* config_file);
```
**Parameters**
* `name` - unique name for the IOU Tracker to update.
* `config_file` - absolute pathspec to the IOU config text file in use.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_iou_config_file_set('my-iou-tracker', './test/configs/iou_config.txt')
```

<br>

### *dsl_tracker_meta_batch_handler_add*
```C++
DslReturnType dsl_tracker_meta_batch_handler_add(const wchar_t* name, 
    dsl_meta_batch_handler_cb handler, void* user_data);
```
**Parameters**
* `name` - unique name for the Tracker to update.
* `handler` - unique meta batch handler (callback function) to add
* `user_data` - opaque pointer to callers userdata, provided on callback

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure
<br>

### *dsl_tracker_meta_batch_handler_remove*
```C++
DslReturnType dsl_tracker_meta_batch_handler_remove(const wchar_t* name, 
    dsl_meta_batch_handler_cb handler);
```
**Parameters**
* `name` - unique name for the Tracker to update.
* `handler` - unique meta batch handler (callback function) to remove

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure
<br>

---

## API Reference
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Seconday GIE](/docs/api-gie.md)
* **Tracker**
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Sink](docs/api-sink.md)
* [Component](/docs/api-component.md)
* [Pipeline](/docs/api-pipeline.md)
