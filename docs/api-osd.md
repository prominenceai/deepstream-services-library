# On-Screen Display (OSD)


---
## On-Screen API
**Constructors:**
* [dsl_osd_new](#dsl_osd_new)

**Methods:**
* [dsl_osd_clock_enabled_get](#dsl_osd_clock_enabled_get)
* [dsl_osd_clock_enabled_set](#dsl_osd_clock_enabled_set)
* [dsl_osd_clock_offsets_get](#dsl_osd_clock_offsets_get)
* [dsl_osd_clock_offsets_set](#dsl_osd_clock_offsets_set)
* [dsl_osd_clock_font_get](#dsl_osd_clock_font_get)
* [dsl_osd_clock_font_set](#dsl_osd_clock_font_set)
* [dsl_osd_clock_color_get](#dsl_osd_clock_color_get)
* [dsl_osd_clock_color_set](#dsl_osd_clock_color_set)

## Return Values
The following return codes are used by the On-Screen Display API
```C++
#define DSL_RESULT_OSD_NAME_NOT_UNIQUE                              0x00050001
#define DSL_RESULT_OSD_NAME_NOT_FOUND                               0x00050002
#define DSL_RESULT_OSD_NAME_BAD_FORMAT                              0x00050003
#define DSL_RESULT_OSD_THREW_EXCEPTION                              0x00050004
#define DSL_RESULT_OSD_MAX_DIMENSIONS_INVALID                       0x00050005
#define DSL_RESULT_OSD_IS_IN_USE                                    0x00050006
#define DSL_RESULT_OSD_SET_FAILED                                   0x00050007
#define DSL_RESULT_OSD_HANDLER_ADD_FAILED                           0x00050008
#define DSL_RESULT_OSD_HANDLER_REMOVE_FAILED                        0x00050009
#define DSL_RESULT_OSD_PAD_TYPE_INVALID                             0x0005000A
#define DSL_RESULT_OSD_COMPONENT_IS_NOT_OSD                         0x0005000B
```

## Constructors
### *dsl_osd_new*
```c++
DslReturnType dsl_osd_new(const wchar_t* name, boolean clock_enabled);
```
The constructor creates a uniquely named On-Screen Display with an optional clock. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the On-Screen Display to create.
* `clock_enable` - [in] set to true to enable On-Screen clock, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_new('my-on-screen-display', True)
```

<br>

## Methods
### *dsl_osd_clock_enabled_get*
```c++
DslReturnType dsl_osd_clock_enabled_get(const wchar_t* name, boolean* enabled);
```
This service returns the current clock enabled setting for the named On-Screen Display
**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `enabled` - [out] true if the On-Screen clock is currently enabled, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_osd_clock_enabled_get('my-on-screen-display')
```

<br>

### *dsl_osd_clock_enabled_set*
```c++
DslReturnType dsl_osd_clock_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the current clock enabled setting for the named On-Screen Display

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `enabled` - [in] set to true to enable the On-Screen clock, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_clock_enabled_set('my-on-screen-display', False)
```

<br>

### *dsl_osd_clock_offsets_get*
```c++
DslReturnType dsl_osd_clock_offsets_get(const wchar_t* name, uint* x_offset, uint* y_offset);
```
This service gets the current clock X and Y offsets for the named On-Screen Display. The offsets start from the upper left corner of the bounding box for each object detected and tracked.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `x_offset` - [out] On-Screen clock offset in the X direction
* `y_offset` - [out] On-Screen clock offset in the Y direction

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, x_offset, y_offset = dsl_osd_clock_offsets_get('my-on-screen-display')
```

<br>

### *dsl_osd_clock_offsets_set*
```c++
DslReturnType dsl_osd_clock_offsets_set(const wchar_t* name, uint offsetX, uint offsetY);
```
This service sets the current clock X and Y offsets for the named On-Screen Display. The offsets start from the upper left corner of the bounding box for each object detected and tracked.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `x_offset` - [in] On-Screen clock offset in the X direction
* `y_offset` - [in] On-Screen clock offset in the Y direction

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, x_offset, y_offset = dsl_osd_clock_offsets_get('my-on-screen-display')
```

<br>

### *dsl_osd_clock_font_get*
```c++
DslReturnType dsl_osd_clock_font_get(const wchar_t* name, const wchar_t** font, uint size);
```
This service gets the current clock font name and size for the named On-Screen Display

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `font` - [out] Name of the clock font type currently in use.
* `size` - [out] Size of the On-Screen clock

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, font, size = dsl_osd_clock_font_get('my-on-screen-display')
```

<br>

### *dsl_osd_clock_font_set*
```c++
DslReturnType dsl_osd_clock_font_set(const wchar_t* name, const wchar_t* font, uint size);
```
This service sets the current clock font type and size for the named On-Screen Display

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `font` - [in] Name of the clock font type currently in use.
* `size` - [in] Size of the On-Screen clock

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_clock_font_set('my-on-screen-display', 'arial', 12)
```

<br>

### *dsl_osd_clock_color_get*
```c++
DslReturnType dsl_osd_clock_color_get(const wchar_t* name, uint* red, uint* green, uint* blue);
```
This service gets the current clock color for the named On-Screen Display. The color is represented as weights for the three RGB values.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `red` - [out] red color weight `[0..255]`
* `green` - [out] green color weight `[0..255]`
* `blue` - [out] blue color weight `[0..255]`

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, red, green, blue = dsl_osd_clock_color_get('my-on-screen-display')
```

<br>


### *dsl_osd_clock_color_set*
```c++
DslReturnType dsl_osd_clock_color_set(const wchar_t* name, uint red, uint green, uint blue);
```
This service gets the current clock color for the named On-Screen Display. The color is represented as weights for the three RGB values.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `red` - [in] red color weight `[0..255]`
* `green` - [in] green color weight `[0..255]`
* `blue` - [in] blue color weight `[0..255]`

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_clock_color_set('my-on-screen-display', 255, 0, 0)
```

<br>

### *dsl_osd_batch_meta_handler_add*
```c++
DslReturnType dsl_osd_batch_meta_handler_add(const wchar_t* name, uint type, 
    dsl_batch_meta_handler_cb handler, void* user_data);
```
This function adds a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb) to either the `sink-pad`(on input to the On-Screen Display) or `src-pad` (on ouput from the On-Screen Display). Once added, the handler will be called to handle batch-meta data for each frame buffer. An On-Screen Display can have more than one `sink-pad` and `src-pad` batch meta handler, and each handler can be added to more than one On-Screen Display.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to update.
* `pad` - [in] to which of the two pads to add the handler; `DSL_PAD_SIK` | `DSL_PAD SRC`
* `handler` - [in] callback function to process batch meta data
* `user_data` [in] opaque pointer to the the caller's user data - passed back with each callback call.

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

    
### *dsl_osd_batch_meta_handler_remove*
```c++
DslReturnType dsl_osd_batch_meta_handler_remove(const wchar_t* name, uint pad);
```
```
This function removes a batch meta handler callback function of type [dsl_batch_meta_handler_cb](#dsl_batch_meta_handler_cb), previously added to the On-Screen Display with [dsl_osd_batch_meta_handler_add](#dsl_osd_batch_meta_handler_add). A Tiled Display can have more than one Sink and Source batch meta handler, and each handler can be added to more than one On-Screen.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to update.
* `pad` - [in] which of the two pads to remove the handler to; DSL_PAD_SINK | DSL_PAD SRC
* `handler` - [in] callback function to remove

**Returns**
`DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

<br>

---

## API Reference
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Seconday GIE](/docs/api-gie)
* [Tracker](/docs/api-tracker.md)
* **On-Screen Display**
* [Tiler](/docs/api-tiler.md)
* [Sink](docs/api-sink.md)
* [Component](/docs/api-component.md)
* [Pipeline](/docs/api-pipeline.md)
