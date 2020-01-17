# Tiled Display (Tiler) API

## Tiled Display API
* [dsl_tiler_new](#dsl_tiler_new)
* [dsl_tiler_dimensions_get](#dsl_tiler_dimensions_get)
* [dsl_tiler_dimensions_set](#dsl_tiler_dimensions_set)
* [dsl_tiler_tiles_get](#dsl_display_tiles_get)
* [dsl_tiler_tiles_set](#dsl_display_tiles_set)
* [dsl_tiler_batch_meta_handler_add](#dsl_tiler_batch_meta_handler_add).
* [dsl_tiler_batch_meta_handler_remove](#dsl_tiler_batch_meta_handler_remove).

## Return Values
The following return codes are used by the Tiler API
```C++
#define DSL_RESULT_TILER_NAME_NOT_UNIQUE                          0x00070001
#define DSL_RESULT_TILER_NAME_NOT_FOUND                           0x00070002
#define DSL_RESULT_TILER_NAME_BAD_FORMAT                          0x00070003
#define DSL_RESULT_TILER_THREW_EXCEPTION                          0x00070004
#define DSL_RESULT_TILER_IS_IN_USE                                0x00070005
#define DSL_RESULT_TILER_SET_FAILED                               0x00070006
#define DSL_RESULT_TILER_HANDLER_ADD_FAILED                       0x00070007
#define DSL_RESULT_TILER_HANDLER_REMOVE_FAILED                    0x00070008
#define DSL_RESULT_TILER_PAD_TYPE_INVALID                         0x00070009
```

## Constructors
### *dsl_tiler_new*
```C++
DslReturnType dsl_tiler_new(const wchar_t* name, uint width, uint height);
```
The constructor creates a uniquely named Tiled Display with given dimensions. Construction will fail if the name is currently in use. The Tiler is created using the default `rows` and `cols` settings of 0 allowing the Tiler to select a best-fit for the number of [Sources] upstream in the Pipeline. The default values can be updated be calling [dsl_tiler_tiles_set](#dsl_display_tiles_set).

**Parameters**
* `name` - [in] unique name for the Tiled Display to create.
* `width` - [in] width of the Tilded Display in pixels
* `width` - [in] height of the Tilded Display in pixels

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

## Methods
### *dsl_tiler_dimensions_get*
This function returns the current width and height of the named Tiled Display.
```C++
DslReturnType dsl_tiler_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
**Parameters**
* `name` - [in] unique name for the Tiled Display to query.
* `width` - [out] width of the Tiled Display in pixels.
* `height` - [out] height of the Tiled Display in pixels.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_tiler_dimensions_set*
This function sets the width and height of the named Tiled Display. The call will fail if the Tiler is currently `in-use`.
```C++
DslReturnType dsl_tiler_dimensions_set(const wchar_t* name, uint width, uint height);
```
**Parameters**
* `name` - [in] unique name for the Tiled Display to update.
* `width` - [in] current width of the Tiled Display in pixels.
* `height` - [in] current height of the Tiled Display in pixels.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_tiler_tiles_get*
```C++
DslReturnType dsl_tiler_tiles_get(const wchar_t* name, uint* cols, uint* rows);
```
**Parameters**
* `name` - unique name for the Tiled Display to query.
* `cols` - [out] current columns setting for the Tiled Display.
* `rows` - [out] current rows setting for the Tiled Display.

**Returns**
`DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

<br>

### *dsl_tiler_tiles_set*
This function sets the number of columns and rows for the named Tiled Display. Setting both values to 0 - the fault - allows the Tiler to determine a bit-fit based on the number of Sources upstream. The number of rows must be at least one half that of columns or the call will fail (e.g. 4x1). The call will also fail if the Tiler is currently `in-use`.
```C++
DslReturnType dsl_tiler_tiles_set(const wchar_t* name, uint cols, uint rows);
```
**Parameters**
* `name` - unique name for the Tiled Display to update.
* `cols` - new columns setting for the Tiled Display.
* `rows` - new rows setting for the Tiled Display.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

<br>

# *dsl_tiler_batch_meta_handler_add*
```C++
DslReturnType dsl_tiler_batch_meta_handler_add(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function adds a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb) to either the `sink-pad`(on input to the Tiled Display) or `src-pad` (on ouput from the Tiled Display). Once added, the handler will be called to handle batch-meta data for each frame buffer. A Tiled Display can have more than one `sink-pad` and `src-pad` batch meta handler, and each handler can be added to more than one Tiled Display.

**Parameters**
* `name` - [in] unique name of the Tiled Display to update.
* `pad` - [in] to which of the two pads to add the handler; `DSL_PAD_SIK` | `DSL_PAD SRC`
* `handler` - [in] callback function to process batch meta data
* `user_data` [in] opaque pointer to the the caller's user data - passed back with each callback call.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

<br>

# *dsl_tiler_batch_meta_handler_remove*
```C++
DslReturnType dsl_tiler_batch_meta_handler_remove(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function removes a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb), previously added to the Tiled Display with [dsl_tiler_batch_meta_handler_add](#dsl_tiler_batch_meta_handler_add). A Tiled Display can have more than one Sink and Source batch meta handler, and each handler can be 
added to more than one Tiled Display.

**Parameters**
* `name` - [in] unique name of the Tiled Display to update.
* `pad` - [in] which of the two pads to remove the handler to; DSL_PAD_SINK | DSL_PAD SRC
* `handler` - [in] callback function to remove

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

<br>

---

## API Reference
* [Source](/docs/source-api.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Seconday GIE](/docs/api-git.md)
* [Tracker](/docs/api-tracker)
* [On-Screen Display](/docs/api-osd.md)
* **Tiler**
* [Sink](/docs/api-sink.md)
* [Component](/docs/api-component.md)
* [Pipeline](/docs/api-pipeline.md)
