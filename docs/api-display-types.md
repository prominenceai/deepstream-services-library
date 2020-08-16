## Display Types API
Display Types are used to add Display metadata to a Frame's metadata to be displayed downstream by an [On-Screen Display](/docs/api-osd.md). Display Types, once created, can be added to an [ODE Action](/docs/api-ode-action.md) in turn added to one or more [ODE Triggers](/docs/api-ode-trigger.md).  Each Trigger, on ODE occurrence, invokes the action to add the Display metadata to the current Frame metadata that triggerd the event.

Further control of the Display Types can be achieved by enabling/disabling the Action or Trigger in a Client callback function when other events occur.  The start and end of a recording session for example. 

#### Construction and Destruction
There are two base types used when creating other complete types for actual display. 
* RGBA Color
* RGBA Font

There are four types for displaying text and shapes. 
* RGBA Line
* RGBA Arrow
* RGBA Rectangle
* RGBA Circle

And three types for displaying source information specific to each frame. 
* Source Number
* Source Name
* Source Dimensions

Display Types are created by calling their type specific constructor. 

Display Typess are deleted by calling [dsl_display_type_delete](#dsl_display_type_delete), [dsl_display_type_delete_many](#dsl_display_type_delete_many), or [dsl_display_type_delete_all](#dsl_display_type_delete_all).

#### Adding to an ODE Action. 
Display Types are added to a Display Action when the action is created by calling [dsl_ode_action_display_meta_add_new](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_new).

Note: Adding a Base Display Type to an ODE Action will fail. 

#### Adding Rectangles to ODE Areas.
RGBA Rectangles are used to define [ODE Areas](/docs/api-ode-area.md) of criteria, either inclussion or exclusion, for one or more [ODE Triggers](/docs/api-ode-trigger.md). Rectangles are added when the ODE Area is created by calling [dsl_ode_area_new](/docs/api-ode-area.md#dsl_ode_area_new).

---

### Display Types API

**Constructors:**
* [dsl_display_type_rgba_color_new](#dsl_display_type_rgba_color_new)
* [dsl_display_type_rgba_font_new](#dsl_display_type_rgba_font_new)
* [dsl_display_type_rgba_line_new](#dsl_display_type_rgba_line_new) 
* [dsl_display_type_rgba_arrow_new](#dsl_display_type_rgba_arrow_new)
* [dsl_display_type_rgba_rectangle_new](#dsl_display_type_rgba_rectangle_new)
* [dsl_display_type_rgba_circle_new](#dsl_display_type_rgba_circle_new)
* [dsl_display_type_source_number_new](#dsl_display_type_source_number_new)
* [dsl_display_type_source_name_new](#dsl_display_type_source_name_new)
* [dsl_display_type_source_dimensions_new](#dsl_display_type_source_dimensions_new)

**Destructors:**
* [dsl_display_type_delete](#dsl_display_type_delete)
* [dsl_display_type_delete_many](#dsl_display_type_delete_many)
* [dsl_display_type_delete_all](#dsl_display_type_delete_all)
* [dsl_display_type_delete_all](#dsl_display_type_delete_all)

**Methods:**
* [dsl_display_type_list_size](#dsl_display_type_list_size)
* [dsl_display_type_meta_add](#dsl_display_type_meta_add)

---
## Return Values
The following return codes are used by the Display Type API
```C++
#define DSL_RESULT_DISPLAY_TYPE_RESULT                              0x00100000
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE                     0x00100001
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND                      0x00100002
#define DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION                     0x00100003
#define DSL_RESULT_DISPLAY_TYPE_IN_USE                              0x00100004
#define DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE                0x00100005
#define DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE                        0x00100006
#define DSL_RESULT_DISPLAY_RGBA_COLOR_NAME_NOT_UNIQUE               0x00100007
#define DSL_RESULT_DISPLAY_RGBA_FONT_NAME_NOT_UNIQUE                0x00100008
#define DSL_RESULT_DISPLAY_RGBA_TEXT_NAME_NOT_UNIQUE                0x00100009
#define DSL_RESULT_DISPLAY_RGBA_LINE_NAME_NOT_UNIQUE                0x0010000A
#define DSL_RESULT_DISPLAY_RGBA_ARROW_NAME_NOT_UNIQUE               0x0010000B
#define DSL_RESULT_DISPLAY_RGBA_ARROW_HEAD_INVALID                  0x0010000C
#define DSL_RESULT_DISPLAY_RGBA_RECTANGLE_NAME_NOT_UNIQUE           0x0010000D
#define DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE              0x0010000E
```

---

## Constructors
### *dsl_display_type_rgba_color_new* 
```C++
DslReturnType dsl_display_type_rgba_color_new(const wchar_t* name, 
    double red, double green, double blue, double alpha);
```
The constructor creates an RGBA Color Display Type. The RGBA Color is a base type used to create other RGBA types that can be added as display metadata to a frame's metadata.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* ` red` - [in]red level for the RGB color [0..1]
* ` blue` - [in] blue level for the RGB color [0..1]
* `green` - [in] green level for the RGB color [0..1]
* `alpha` - [in] alpha level for the RGB color [0..1]
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-red', 1.0, 0.0, 0.0, 1.0)
```

<br>

### *dsl_display_type_rgba_font_new* 
```C++
DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, 
    const wchar_t* font, uint size, const wchar_t* color);
```
The constructor creates an RGBA Font Display Type. The RGBA Font is a base type used to create other RGBA types that can be added as display metadata to a frame's metadata.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `fount` - [in] standard, unique string name of the actual font type (eg. 'arial')
* `size` - [in] size of the font
* `color` - [in] name of the RGBA Color for the RGBA font
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_font_new('red-arial-18', 'arial', 18, 'full-red')
```

<br>

### *dsl_display_type_rgba_text_new* 
```C++
DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, const wchar_t* text, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);    
```
The constructor creates an RGBA Text Display Type. The RGBA Text can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `text` - [in] text string to display
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` [in] - RGBA font to use for the display dext
* `hasBgColor` - [in] set to true to enable bacground color, false otherwise
* `bgColor` [in] RGBA Color for the Text background if set

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_text_new('recording-on', 'REC 0', 30, 30, 'full-red', False, None)
```

<br>

### *dsl_display_type_rgba_line_new* 
```C++
DslReturnType dsl_display_type_rgba_line_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, const wchar_t* color); 
```
The constructor creates an RGBA Line Display Type. The RGBA Line can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x1` - [in] starting x positional offest
* `y1` - [in] starting y positional offest
* `x2` - [in] ending x positional offest
* `y2` - [in] ending y positional offest
* `width` - [in] width of the line in pixels
* `color` - [in] RGBA Color for the RGBA Line
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_line_new('dividing-line', 400, 10, 400, 700, 2, 'full-red')
```

<br>

### *dsl_display_type_rgba_arrow_new* 
```C++
DslReturnType dsl_display_type_rgba_arrow_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, const wchar_t* color);   
```

The constructor creates an RGBA Arrow Display Type. The RGBA Arrow can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x1` - [in] starting x positional offest
* `y1` - [in] starting y positional offest
* `x2` - [in] ending x positional offest
* `y2` - [in] ending y positional offest
* `width` - [in] width of the line in pixels
* `head` - [in] one of `DSL_ARROW_START_HEAD`, `DSL_ARROW_END_HEAD`, `DSL_ARROW_BOTH_HEAD`
* `color` - [in] RGBA Color for the RGBA Arrow

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_arrow_new('arrow-pointer', 220, 165, 370, 235, DSL_ARROW_END_HEAD)
```

<br>

### *dsl_display_type_rgba_rectangle_new* 
```C++
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, uint left, uint top, uint width, uint height, 
    uint border_width, const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);
```

The constructor creates an RGBA Rectangle Display Type. The RGBA Rectangle can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `left` - [in] left positional offest
* `top` - [in] positional offest
* `width` - [in] width of the rectangle in Pixels
* `height` - [in] height of the rectangle in Pixels
* `border_width` - [in] width of the rectangle border in pixels
* `color` - [in] RGBA Color for thIS RGBA Line
* `hasBgColor` - [in] set to true to enable bacground color, false otherwise
* `bgColor` - [in] RGBA Color for the Circle background if set
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_rectangle_new('black-rectangle', 240, 370, 1200, 940, 2, 'full-black', False, None)
```

<br>

### *dsl_display_type_rgba_circle_new* 
```C++
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, uint x_center, uint y_center, uint radius,
    const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);
```

The constructor creates an RGBA Circle Display Type. The RGBA Circle can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).


**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_center` - [in] X positional offset to center of Circle
* `y_center` - [in] y positional offset to center of Circle
* `radius` - [in] radius of the RGBA Circle in pixels 
* `color` - [in] RGBA Color for the RGBA Circle
* `hasBgColor` - [in] set to true to enable bacground color, false otherwise
* `bgColor` - [in] RGBA Color for the Circle background if set

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_circle_new('blue-circle', 220, 220, 20, 'my-blue', True, 'my-blue')
```

<br>

### *dsl_display_type_source_number_new* 
```C++
DslReturnType dsl_display_type_source_number_new(const wchar_t* name, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
```

The constructor creates a uniquely name Source Nuumber Display Type. The Source Number can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` - [in] RGBA font to use for the display text
* `hasBgColor` - [in] set to true to enable bacground color, false otherwise
* `bgColor` - [in] RGBA Color for the Text background if set
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_source_number_new('display-source-number', 10, 10, 'arial-blue-14', False, None)
```

<br>

### *dsl_display_type_source_name_new* 
```C++
DslReturnType dsl_display_type_source_name_new(const wchar_t* name, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
```

The constructor creates a uniquely name Source Name Display Type. The Source Name can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` - [in] RGBA font to use for the display text
* `hasBgColor` - [in] set to true to enable bacground color, false otherwise
* `bgColor` - [in] RGBA Color for the Text background if set
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_source_name_new('display-source-name', 10, 10, 'arial-blue-14', False, None)
```

<br>

### *dsl_display_type_source_dimensions_new* 
```C++
DslReturnType dsl_display_type_source_dimensions_new(const wchar_t* name, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
```

The constructor creates a uniquely name Source Dimensions Display Type. The Source Dimensions can be added as display metadata to a frame's metadata when using a [Pad Probe Handler](/docs/api-pph.md).

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` - [in] RGBA font to use for the display text
* `hasBgColor` - [in] set to true to enable bacground color, false otherwise
* `bgColor` - [in] RGBA Color for the Text background if set
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_source_dimensions_new('display-source-dimensions', 10, 10, 'arial-blue-14', False, None)
```

<br>

---

## Destructors
### *dsl_display_type_delete*
```C++
DslReturnType dsl_display_type_delete(const wchar_t* name);
```
This destructor deletes a single, uniquely named Display Type. The destructor will fail if the Display Type is currently `in-use` by an ODE Action

**Parameters**
* `name` - [in] unique name for the Display Type to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_delete('blue-circle')
```

<br>

### *dsl_display_type_delete_many*
```C++
DslReturnType dsl_display_type_delete_many(const wchar_t** names);
```
This destructor deletes multiple uniquely named Display Types. Each name is checked for existence, with the function returning `DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND` on first occurrence of failure. The destructor will fail if one of the Display Types is currently `in-use` by one or more ODE Actions

**Parameters**
* `names` - [in] a NULL terminated array of uniquely named Display Types to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_delete_many(['my-blue-circle', 'my-blue-color', None])
```

<br>

### *dsl_display_type_delete_all*
```C++
DslReturnType dsl_display_type_delete_all();
```
This destructor deletes all Display Types currently in memory. The destructor will fail if any one of the Display Types is currently `in-use` by an ODE Action. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_delete_all()
```

<br>

---

## Methods
### *dsl_display_type_list_size*
```c++
uint dsl_display_type_list_size();
```
This service returns the size of the display_type container, i.e. the number of Display Types currently in memory. 

**Returns**
* The size of the Display Types container

**Python Example**
```Python
size = dsl_display_type_list_size()
```

<br>

### *dsl_display_type_meta_add*
```c++
DslReturnType dsl_display_type_meta_add(const wchar_t* name, void* buffer, void* frame_meta);
```
This service, when called from a custom [Pad Probe Handler](/docs/api-pph.md), adds the named Display Type as display-meta to the frame meta for the associated buffer.

**Parmeters**
* `name` - [in] unique name for the Display Type to add
* `display_meta` - [in] opaque pointer to the aquired display meta to to add the Display Type to
* `frame_meta` - [in] opaque pointer to a Frame's meta data to add the Display Type.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_meta_add('blue-circle', buffer, frame_meta)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [Tiler](/docs/api-tiler.md)
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE-Trigger](/docs/api-ode-trigger.md)
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* **Display Types**
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)


