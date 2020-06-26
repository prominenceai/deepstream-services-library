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
* [dsl_tracker_max_dimensions_get](#dsl_tracker_max_dimensions_get)
* [dsl_tracker_max_dimensions_set](#dsl_tracker_max_dimensions_set)
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
This service creates a unqiuely named KTL Tracker component. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the KTL Tracker to create.
* `max_width` - [in] maximum width of each frame for the input transform
* `max_height` - [in] maximum height of each frame for the input transform

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

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
This service creates a unqiuely named IOU Tracker component. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the IOU Tracker to create.
* `config_file` - [in] relative or absolute pathspec to a valid IOU config text file
* `max_width` - [in] maximum width of each frame for the input transform
* `max_height` - [in] maximum height of each frame for the input transform

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_iou_new('my-', './test/configs/iou_config.txt', 480, 270)
```

<br>

## Methods
### *dsl_tracker_max_dimensions_get*
```C++
DslReturnType dsl_tracker_max_dimensions_get(const wchar_t* name, uint* max_width, uint* max_height);
```

This service returns the max input frame dimensions to track on in use by the named Tracker.

**Parameters**
* `name` - [in] unique name of the Tracker to query.
* `max_width` - [out] current maximum width of each frame for the input transform
* `max_height` - [out] current maximum height of each frame for the input transform

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, max_width, max_height = dsl_tracker_dimensions_get('my-tracker')
```

<br>

### *dsl_tracker_max_dimensions_set*
```C++
DslReturnType dsl_tracker_max_dimensions_set(const wchar_t* name, uint max_width, uint max_height);
```
This Service sets the max input frame dimensions for the name Tracker.

**Parameters**
* `name` - [in] unique name of the Tracker to update.
* `max_width` - [in] new maximum width of each frame for the input transform.
* `max_height` - [in] new maximum height of each frame for the input transform.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tracker_dimensions_set('my-tracker', 480, 270)
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

### *dsl_tracker_meta_batch_handler_add*
```C++
DslReturnType dsl_tracker_meta_batch_handler_add(const wchar_t* name, 
    dsl_meta_batch_handler_cb handler, void* user_data);
```
This service adds a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb) to either the `sink-pad` (on input to the Tracker) or `src-pad` (on output from the Tracker). Once added, the handler will be called to handle batch-meta data for each meta buffer. A Tracker can have more than one `sink-pad` and `src-pad` batch meta handler, and each handler can be added to more than one Tracker. Adding the same handler function to the same `sink` or `src` pad more than once will fail. 

**Parameters**
* `name` - [in] unique name for the Tracker to update.
* `handler` - [in] unique meta batch handler (callback function) to add
* `user_data` - [in] opaque pointer to callers userdata, provided on callback

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
* Example using Nvidia's pyds lib to handle batch-meta data

```Python
##
# Callback function to handle batch-meta data
##
def tracker_batch_meta_handler_cb(buffer, user_data):

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break    

        # Handle the frame_meta data, typically more than just printing to console
        
        print("Frame Number is ", frame_meta.frame_num)
        print("Source id is ", frame_meta.source_id)
        print("Batch id is ", frame_meta.batch_id)
        print("Source Frame Width ", frame_meta.source_frame_width)
        print("Source Frame Height ", frame_meta.source_frame_height)
        print("Num object meta ", frame_meta.num_obj_meta)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return True

##
# Create a new tracker component and add the batch-meta handler function above to the Source (output) Pad.
##
retval = dsl_tracker_ktl_new('my-ktl-tracker', 480, 270)
retval += dsl_tracker_batch_meta_handler_add('my-ktl-tracker', 
    DSL_PAD_SRC, tracker_batch_meta_handler_cb, None)

if retval != DSL_RESULT_SUCCESS:
    # Tracker setup failed
```

<br>

### *dsl_tracker_meta_batch_handler_remove*
```C++
DslReturnType dsl_tracker_meta_batch_handler_remove(const wchar_t* name, 
    dsl_meta_batch_handler_cb handler);
```
This function removes a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb), previously added to the Tracker with [dsl_tracker_batch_meta_handler_add](#dsl_tracker_batch_meta_handler_add). 

**Parameters**
* `name` - [in] unique name for the Tracker to update.
* `handler` - [in] unique meta batch handler (callback function) to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tracker_batch_meta_handler_remove('my-ktl-tracker', 
    DSL_PAD_SRC, tracker_batch_meta_handler_cb)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* **Tracker**
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
