# Multi-Object Tracker API Reference

## Tracker API
* [dsl_tracker_ktl_new](#dsl_tracker_ktl_new)
* [dsl_tracker_iou_new](#dsl_tracker_iou_new)
* [dsl_tracker_object_ids_get](#dsl_tracker_object_ids_get)
* [dsl_tracker_object_ids_set](#dsl_tracker_object_ids_set)
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
#define DSL_RESULT_TRACKER_NAME_NOT_UNIQUE                          0x00110001
#define DSL_RESULT_TRACKER_NAME_NOT_FOUND                           0x00110010
#define DSL_RESULT_TRACKER_NAME_BAD_FORMAT                          0x00110011
#define DSL_RESULT_TRACKER_THREW_EXCEPTION                          0x00110100
#define DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND                    0x00110101
#define DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID                   0x00110110
```

## Constructors
### *dsl_tracker_ktl_new*
```C++
DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint max_width, uint max_height);
```
**Parameters**
* `name` - unique name for the KTL Tracker to create.
* `max_width` - maximum width of each frame for the input transform
* `max_height` - maximum height of each frame for the input transform

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_tracker_iou_new*
```C++
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, 
    uint width, uint height);
```
**Parameters**
* `name` - unique name for the IOU Tracker to create.
* `config_file` - relative or absolute pathspec to a valid IOU config text file
* `max_width` - maximum width of each frame for the input transform
* `max_height` - maximum height of each frame for the input transform

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

## Methods
### *dsl_tracker_object_ids_get*
```C++
DslReturnType dsl_tracker_object_ids_get(const wchar_t* name, uint** object_ids);
```

### *dsl_tracker_object_ids_set*
```C++
DslReturnType dsl_tracker_object_ids_set(const wchar_t* name, uint* object_ids);
```
**Parameters**
* `name` - unique name of the Tracker to update.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_tracker_dimensions_get*
```C++
DslReturnType dsl_tracker_dimensions_get(const wchar_t* name, uint& width, uint& height);
```
**Parameters**
* `name` - unique name of the Tracker to query.
* `max_width` - current maximum width of each frame for the input transform
* `max_height` - current maximum height of each frame for the input transform

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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

<br>

### *dsl_tracker_iou_config_file_get*
```C++
DslReturnType dsl_tracker_iou_config_file_get(const wchar_t* name, const wchar_t** config_file);
```
**Parameters**
* `name` - unique name for the IOU Tracker to query.
* `config_file` - absolute pathspec to the IOU config text file in use.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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

