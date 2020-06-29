# Multi-Stream Tiler API
Tiler components perform frame-rendering from multiple-sources into a 2D grid array with one tile per source.  As with all components, Tilers must be uniquely named from all other components created. Tiler components have dimensions, `width` and `height`, and a number-of-tiles expressed in `rows` and `cols`. Dimension must be set on creation, whereas `rows` and `cols` default to 0 indicating best-fit based on the number of sources. Both dimensions and tiles can be updated after Tiler creation, as long as the Tiler is not currently `in-use` by a Pipeline.

#### Tiler Construction and Destruction
Tilers are constructued usThe relationship between Branches and Tilers is one-to-one. Once added to a Pipeline or Branch, a Tiler must be removed before it can used with another. Tilers are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

#### Adding to a Pipeline
A Multi-Stream Tilers are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other components) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](api-pipeline.md#dsl_pipeline_component_remove_all).

#### Adding to a Branch
Tilers are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other components) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](api-pipeline.md#dsl_pipeline_component_remove_all).

## Tiler API
**Constructors**
* [dsl_tiler_new](#dsl_tiler_new)

**Methods**
* [dsl_tiler_dimensions_get](#dsl_tiler_dimensions_get)
* [dsl_tiler_dimensions_set](#dsl_tiler_dimensions_set)
* [dsl_tiler_tiles_get](#dsl_tiler_tiles_get)
* [dsl_tiler_tiles_set](#dsl_tiler_tiles_set)
* [dsl_tiler_batch_meta_handler_add](#dsl_tiler_batch_meta_handler_add).
* [dsl_tiler_batch_meta_handler_remove](#dsl_tiler_batch_meta_handler_remove).

## Return Values
The following return codes are used by the Tiler API
```C++
#define DSL_RESULT_TILER_NAME_NOT_UNIQUE                            0x00070001
#define DSL_RESULT_TILER_NAME_NOT_FOUND                             0x00070002
#define DSL_RESULT_TILER_NAME_BAD_FORMAT                            0x00070003
#define DSL_RESULT_TILER_THREW_EXCEPTION                            0x00070004
#define DSL_RESULT_TILER_IS_IN_USE                                  0x00070005
#define DSL_RESULT_TILER_SET_FAILED                                 0x00070006
#define DSL_RESULT_TILER_HANDLER_ADD_FAILED                         0x00070007
#define DSL_RESULT_TILER_HANDLER_REMOVE_FAILED                      0x00070008
#define DSL_RESULT_TILER_PAD_TYPE_INVALID                           0x00070009
#define DSL_RESULT_TILER_COMPONENT_IS_NOT_TILER                     0x0007000A
```

## Constructors
### *dsl_tiler_new*
```C++
DslReturnType dsl_tiler_new(const wchar_t* name, uint width, uint height);
```
The constructor creates a uniquely named Tiler with given dimensions. Construction will fail if the name is currently in use. The Tiler is created using the default `rows` and `cols` settings of 0 allowing the Tiler to select a best-fit for the number of [Sources] upstream in the Pipeline. The default values can be updated be calling [dsl_tiler_tiles_set](#dsl_display_tiles_set).

**Parameters**
* `name` - [in] unique name for the Tiler to create.
* `width` - [in] width of the Tilded Display in pixels
* `height` - [in] height of the Tilded Display in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tiler_new('my-tiler', 1280, 720)
```

<br>

### *dsl_demuxer_new*
```C++
DslReturnType dsl_demuxer_new(const wchar_t* name);
```
The constructor creates a uniquely named Demuxer. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Demuxer to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_demuxer_new('my-demuxer')
```

<br>

## Methods
### *dsl_tiler_dimensions_get*
```C++
DslReturnType dsl_tiler_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
This function returns the current width and height of the named Tiler.

**Parameters**
* `name` - [in] unique name for the Tiler to query.
* `width` - [out] width of the Tiler in pixels.
* `height` - [out] height of the Tiler in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_tiler_dimensions_get('my-tiler')
```

<br>

### *dsl_tiler_dimensions_set*
```C++
DslReturnType dsl_tiler_dimensions_set(const wchar_t* name, uint width, uint height);
```
This function sets the width and height of the named Tiler. The call will fail if the Tiler is currently `in-use`.

**Parameters**
* `name` - [in] unique name for the Tiler to update.
* `width` - [in] current width of the Tiler in pixels.
* `height` - [in] current height of the Tiler in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tiler_dimensions_get('my-tiler', 1280, 720)
```

<br>

### *dsl_tiler_tiles_get*
```C++
DslReturnType dsl_tiler_tiles_get(const wchar_t* name, uint* cols, uint* rows);
```
This function returns the current cols and rows settings in use by the named Tiler. Values of 0 - the default - allow the Tiler to determine a best-fit based on the number of Sources upstream in the Pipeline

**Parameters**
* `name` - [in] unique name for the Tiler to query.
* `cols` - [out] current columns setting for the Tiler.
* `rows` - [out] current rows setting for the Tiler.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, cols, rows = dsl_tiler_tiles_get('my-tiler')
```

<br>

### *dsl_tiler_tiles_set*
```C++
DslReturnType dsl_tiler_tiles_set(const wchar_t* name, uint cols, uint rows);
```
This function sets the number of columns and rows for the named Tiler. Setting both values to 0 - the default - allows the Tiler to determine a best-fit based on the number of Sources upstream. The number of rows must be at least one half that of columns or the call will fail (e.g. 4x1). The call will also fail if the Tiler is currently `in-use`.

**Parameters**
* `name` - [in] unique name for the Tiler to update.
* `cols` - [in] new columns setting for the Tiler.
* `rows` - [in] new rows setting for the Tiler.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tiler_tiles_set('my-tiler', 3, 2)
```

<br>

### *dsl_tiler_batch_meta_handler_add*
```C++
DslReturnType dsl_tiler_batch_meta_handler_add(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function adds a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb) to either the `sink-pad`(on input to the Tiler) or `src-pad` (on ouput from the Tiler). Once added, the handler will be called to handle batch-meta data for each frame buffer. A Tiler can have more than one `sink-pad` and `src-pad` batch meta handler, and each handler can be added to more than one Tiler.

**Parameters**
* `name` - [in] unique name of the Tiler to update.
* `pad` - [in] to which of the two pads to add the handler; `DSL_PAD_SIK` | `DSL_PAD SRC`
* `handler` - [in] callback function to process batch meta data
* `user_data` [in] opaque pointer to the the caller's user data - passed back with each callback call.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
* Example using Nvidia's pyds lib to handle batch-meta data

```Python

##
# Create a new Tiler component and add the batch-meta handler function to the Sink (input) Pad.
##
retval = dsl_tiler_new('my-tiler', 1280, 720)
retval += dsl_tiler_batch_meta_handler_add('my-tiler', DSL_PAD_SINK, my_tiler_batch_meta_handler_cb, None)

if retval != DSL_RESULT_SUCCESS:
    # Tiler setup failed
```    

<br>

### *dsl_tiler_batch_meta_handler_remove*
```C++
DslReturnType dsl_tiler_batch_meta_handler_remove(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function removes a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb), previously added to the Tiler with [dsl_tiler_batch_meta_handler_add](#dsl_tiler_batch_meta_handler_add). A Tiler can have more than one Sink and Source batch meta handler, and each handler can be added to more than one Tiler. Each callback added to a single pad must be unique.

**Parameters**
* `name` - [in] unique name of the Tiler to update.
* `pad` - [in] which of the two pads to remove the handler to; DSL_PAD_SINK | DSL_PAD SRC
* `handler` - [in] callback function to remove

**Returns**
*`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
    retval = dsl_tiler_batch_meta_handler_remove('my-tiler',  DSL_PAD_SINK, my_tiler_batch_meta_handler_cb)
```

<br>

### *dsl_demuxer_batch_meta_handler_add*
```C++
DslReturnType dsl_demuxer_batch_meta_handler_add(const wchar_t* name,
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function adds a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb) to the `sink-pad`, on the single stream input to the Demuxer. Once added, the handler will be called to handle batch-meta data for each frame buffer. A Demuxer can have more than one `sink-pad` (only) batch meta handler, and each handler can be added to more than one Tiler.

**Parameters**
* `name` - [in] unique name of the Tiler to update.
* `handler` - [in] callback function to process batch meta data
* `user_data` [in] opaque pointer to the the caller's user data - passed back with each callback call.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
##
# Create a new Tiler component and add the batch-meta handler function.
##
retval = dsl_demuxer_new('my-demuxer')
retval += dsl_demuxer_batch_meta_handler_add('my-demuxer', my_demuxer_batch_meta_handler_cb, None)

if retval != DSL_RESULT_SUCCESS:
    # Demuxer setup failed
```    

<br>

### *dsl_demuxer_batch_meta_handler_remove*
```C++
DslReturnType dsl_demuxer_batch_meta_handler_remove(const wchar_t* name, 
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function removes a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb), previously added to the Demuer with [dsl_demuxer_batch_meta_handler_add](#dsl_demuxer_batch_meta_handler_add). A Tiler can have more than one Sink (only) batch meta handler, and each handler can be added to more than one Tiler. Each callback added to a single pad must be unique.

**Parameters**
* `name` - [in] unique name of the Tiler to update.
* `handler` - [in] callback function to remove

**Returns**
*`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
    retval = dsl_demuxer_batch_meta_handler_remove('my-demuxer', my_demuxer_batch_meta_handler_cb)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [ODE Handler](/docs/api-ode-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* **Tiler**
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
