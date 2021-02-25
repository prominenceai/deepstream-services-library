# On-Screen Display (OSD)
On-Screen Display components provide visualization of object categorization and tracking. OSDs add bounding boxes, labels, and clocks to objects detected in the video stream. As with all components, OSDs must be uniquely named from all other components created. 

#### OSD Construction and Destruction
The constructor [dsl_osd_new](#dsl_osd_new) is used to create an OSD with a single option of enabling the on-screen clock. Once created, the OSD's clock parameters -- fonts, color and offsets -- can be modified from their default values. OSDs are deleted by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)

#### Adding and Removing 
A single OSD can be added to Pipeline trunk or inidividual branch. OSD are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all). 

A similar set of Services are used when adding/removing an OSD to/from a branch: [dsl_branch_component_add](api-branch.md#dsl_branch_component_add), [dsl_branch_component_add_many](/docs/api-branch.md#dsl_branch_component_add_many), [dsl_branch_component_remove](/docs/api-branch.md#dsl_branch_component_remove), [dsl_branch_component_remove_many](/docs/api-branch.md#dsl_branch_component_remove_many), and [dsl_branch_component_remove_all](/docs/api-branch.md#dsl_branch_component_remove_all). 

Once added to a Pipeline or Branch, an OSD must be removed before it can be used with another. 

---
## On-Screen API
**Constructors:**
* [dsl_osd_new](#dsl_osd_new)

**Methods:**
* [dsl_osd_text_enabled_get](#dsl_osd_clock_enabled_get)
* [dsl_osd_text_enabled_set](#dsl_osd_clock_enabled_set)
* [dsl_osd_clock_enabled_get](#dsl_osd_clock_enabled_get)
* [dsl_osd_clock_enabled_set](#dsl_osd_clock_enabled_set)
* [dsl_osd_clock_offsets_get](#dsl_osd_clock_offsets_get)
* [dsl_osd_clock_offsets_set](#dsl_osd_clock_offsets_set)
* [dsl_osd_clock_font_get](#dsl_osd_clock_font_get)
* [dsl_osd_clock_font_set](#dsl_osd_clock_font_set)
* [dsl_osd_clock_color_get](#dsl_osd_clock_color_get)
* [dsl_osd_clock_color_set](#dsl_osd_clock_color_set)
* [dsl_osd_pph_add](#dsl_osd_pph_add)
* [dsl_osd_pph_remove](#dsl_osd_pph_remove)

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
#define DSL_RESULT_OSD_COLOR_PARAM_INVALID                          0x0005000C
```

## Constructors
### *dsl_osd_new*
```c++
DslReturnType dsl_osd_new(const wchar_t* name, 
    boolean text_enabled, boolean clock_enabled);
```
The constructor creates a uniquely named On-Screen Display with an optional clock. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the On-Screen Display to create.
* `text_enable` - [in] set to true to enable On-Screen object text - class label and tracking number -  false otherwise
* `clock_enable` - [in] set to true to enable On-Screen clock, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_new('my-on-screen-display', True, True)
```

<br>

## Methods
### *dsl_osd_text_enabled_get*
```c++
DslReturnType dsl_osd_text_enabled_get(const wchar_t* name, boolean* enabled);
```
This service returns the current object text enabled setting for the named On-Screen Display

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `enabled` - [out] true if the On-Screen object text - class label and tracking number -  is currently enabled, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_osd_text_enabled_get('my-on-screen-display')
```

<br>

### *dsl_osd_text_enabled_set*
```c++
DslReturnType dsl_osd_text_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the current clock enabled setting for the named On-Screen Display

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `enable` - [in] set to true to enable On-Screen object text - class label and tracking number -  false otherwise


**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_text_enabled_set('my-on-screen-display', False)
```

<br>

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
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

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
DslReturnType dsl_osd_clock_color_get(const wchar_t* name, 
    double* red, double* green, double* blue, double alpha);
```
This service gets the current clock color for the named On-Screen Display. The color is represented in RGBA format, each with weights between 0.0 and 1.0.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `red` - [out] red color weight `[0.0...1.0]`
* `green` - [out] green color weight `[0.0...1.0]`
* `blue` - [out] blue color weight `[0.0...1.0]`
* `alpha` - [out] alpha color weight `[0.0...1.0]`

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, red, green, blue, alpha = dsl_osd_clock_color_get('my-on-screen-display')
```

<br>


### *dsl_osd_clock_color_set*
```c++
DslReturnType dsl_osd_clock_color_set(const wchar_t* name, double red, uint double, uint blue);
```
This service gets the current clock color for the named On-Screen Display. The color is  The color is specified in RGBA format.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to query.
* `red` - [in] red color weight `[0.0...1.0]`
* `green` - [in] green color weight `[0.0...1.0]`
* `blue` - [in] blue color weight `[0.0...1.0]`
* `alpha` - [in] alpha color weight `[0.0...1.0]`

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_osd_clock_color_set('my-on-screen-display', 0.6, 0.0, 0.6, 1.0)
```

<br>

### *dsl_osd_pph_add*
```C++
DslReturnType dsl_osd_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named On-Screen Display.

**Parameters**
* `name` - [in] unique name of the On-Screen Display to update.
* `handler` - [in] unique name of Pad Probe Handler to add
* `pad` - [in] to which of the two pads to add the handler: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
* Example using Nvidia's pyds lib to handle batch-meta data

```Python
retval = dsl_osd_pph_add('my-osd', 'my-pph-handler', `DSL_PAD_SINK`)
```

<br>

### *dsl_osd_pph_remove*
```C++
DslReturnType dsl_osd_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);
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
retval = dsl_osd_pph_remove('my-osd', 'my-pph-handler', `DSL_PAD_SINK`)
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
* [Tiler](/docs/api-tiler.md)
* **On-Screen Display**
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
