# Display Type API
Display Types are used to display Text and Shapes when using an [On-Screen Display](/docs/api-osd.md) Component. The Display Types are overlaid by adding metadata to a Frame's Display-Meta prior to the On-Screen Display. 

There are two base types used to specify colors and fonts when creating derived types for actual display.

### Base Types
* **RGBA Color** - red, green, blue, and alpha levels between 0.0 and 1.0
* **RGBA Font** - derived from any standard font with size and RGBA Color.

### Derived  Types
* **RGBA Text** - derived from an RGBA Font with an optional RGBA background Color
* **RGBA Line** - with start and end coordinates, linewidth, and RGBA Color
* **RGBA Arrow** - same parameters as RGBA Line with and additional parameter for the arrowhead style.
* **RGBA Rectangle** - with coordinates, dimensions, border-width, and RGBA Colors for the border and background (optional).
* **RGBA Circle** - with coordinates, radius, and RGBA Colors for the border and background (optional)

### Adding Display Types

#### For static display on every frame:
To overlay static Display types over every frame, use an [Overlay Frame ODE Action](/docs/api-ode-action#dsl_ode_action_overlay_frame_new), added to an [Always ODE Trigger](/docs/api-ode-trigger.md). 

Using Python for example
```Python
# new RGBA color Display Type
retval = dsl_display_type_rgba_color_new('full-blue', red=0.0, green=0.0, blue=1.0, alpha=1.0)

# new RGBA font using our new color
retval = dsl_display_type_rgba_font_new('arial-20-blue', font='arial', size=20, color='full-blue')

# new RGBA display text
retval = dsl_display_type_rgba_text_new('display-text', text='My Home Security', x_offset=733, y_offset=5, 
    font='arial-20-blue', has_bg_color=False, bg_color='full-blue')

# Create an Action that will display the text by adding the metadata to the Frame's display meta
retval = dsl_ode_action_overlay_frame_new('overlay-display-text', display_type='display-text')

# new Always triger to overlay our display text on every frame
retval = dsl_ode_trigger_always_new('always-trigger', when=DSL_ODE_PRE_OCCURRENCE_CHECK)

# finally, add the Overlay-Frame Action to our Always Trigger.
retval = dsl_ode_trigger_action_add('always-trigger', action='overlay-display-text')
```
#### For static display on specific frames:
Text or shapes can be used to indicate the occurrence of specific detection events. 

Using Python for example
```Python
# new RGBA color Display Type
retval = dsl_display_type_rgba_color_new('full-yellow', red=1.0, green=1.0, blue=0.0, alpha=1.0)

# new RGBA display rectangle
retval = dsl_display_type_rgba_rectangle_new('warning-rectangle', left=10, top=10, width=50, height=50, 
    border_width=0, color='full-yellow', has_bg_color=True, bg_color='full-yellow')

# Create an Action that will display the warning by adding the metadata to the Frame's display meta
retval = dsl_ode_action_overlay_frame_new('overlay-warning', display_type='warning-rectangle')

# new Maximum Objects triger to invoke our 'overlay-warning' action when to many objects are detected
retval = dsl_ode_trigger_maximum_new('max-trigger', class_id=PGIE_CLASS_ID_PERSON, limit=0, maximum=10)

# finally, add the Overlay-Frame Action to our Max Objects Trigger.
retval = dsl_ode_trigger_action_add('max-trigger', action='overlay-warning')
```
#### Display Types Construction and Destruction
Display Types are created by calling one of the type specific [constructors](#display-type-api) defined below. Each constructor must have a unique name and using a duplicate name will fail with a result of `DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE`. Once created, all Display Types are deleted by calling [dsl_display_type_delete](#dsl_display_type_delete),
[dsl_display_type_delete_many](#dsl_display_type_delete_many), or [dsl_display_type_delete_all](#dsl_display_type_delete_all). Attempting to delete a Display Type in-use by an ODE Action will fail with a result of `DSL_RESULT_ODE_DISPLAY_TYPE_IN_USE`

#### Display Types and the ODE Overlay Frame Action
A Derived Display Type (Text, Line, Arrow, Rectangle, Circle) is added to a Frame Overlay Action when the action is created by calling [dsl_ode_action_overlay_frame_new](docs/api-ode-action#dsl_ode_action_overlay_frame_new). The ODE Action will overlay the Display Type by adding it to the Frame's Metadata when the Action is invoked on ODE occurrence. Use an [ODE Always Trigger](/docs/api-ode-trigger.md#dsl_ode_trigger_always_new) to overlay Display Types on each and every frame.

## Display Type API
**Constructors:**
* [dsl_display_type_rgba_color_new](#dsl_display_type_rgba_color_new)
* [dsl_display_type_rgba_font_new](#dsl_display_type_rgba_font_new)
* [dsl_display_type_rgba_text_new](#dsl_display_type_rgba_text_new)
* [dsl_display_type_rgba_line_new](#dsl_display_type_rgba_line_new)
* [dsl_display_type_rgba_arrow_new](#dsl_display_type_rgba_arrow_new)
* [dsl_display_type_rgba_rectangle_new](#dsl_display_type_rgba_rectangle_new)
* [dsl_display_type_rgba_circle_new](#dsl_display_type_rgba_circle_new)


**Destructors:**
* [dsl_display_type_delete](#dsl_display_type_delete)
* [dsl_display_type_delete_many](#dsl_display_type_delete_many)
* [dsl_display_type_delete_all](#dsl_display_type_delete_all)

**Methods:**
* [dsl_display_type_list_size](#dsl_display_type_list_size)

---
## Return Values
The following return codes are used by the Display Types API
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
The constructor creates a uniquely named **RGBA Color** Display type. This is a Base type and is used as input when creating other Derived Display Types.

**Parameters**
* `name` - [in] unique name for the RGBA Display Color
* `red` - [in] red level for the RGBA color [0..1]
* `green` - [in] green level for the RGBA color [0..1]
* `blue` - [in] blue level for the RGBA color [0..1]
* `alpha` - [in] alpha level for the RGBA color [0..1]

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-while', 1.0, 1.0, 1.0, 1.0)
```

<br>

### *dsl_display_type_rgba_font_new*
```C++
DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, 
  const wchar_t* font, uint size, const wchar_t* color);
```
The constructor creates a uniquely named **RGBA Font** Display type. This is a Base type and is used as input when creating other Derived Display Types.

**Parameters**
* `name` - [in] unique name for the RGBA Display Font
* `font` - [in] standard name of the actual font type (eg. 'arial')
* `size` - [in] size of the Font
* `color` - [in] name of an RGBA Display Color for the Font type 

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-while', 1.0, 1.0, 1.0, 1.0)
retval = dsl_display_type_rgba_font_new('arial-white-16', 'arial', 16, 'full-while')
```

### *dsl_display_type_rgba_text_new*
```C++
DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, const wchar_t* text, uint x_offset, uint y_offset, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);
```    
The constructor creates a uniquely named **RGBA Text** Display type.

**Parameters**
* `name` - [in] unique name for the RGBA Display Text
* `text` - [in] actual text for display
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` - [in] name of an RGB Font Type to use for the Display Text
* `has_bg_color` - [in] set to true if the Text is to Display a background RGBA Color
* `bg_color` - [in] name of an RGBA Display Color for the Text background

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-while', 1.0, 1.0, 1.0, 1.0)
retval = dsl_display_type_rgba_color_new('full-black', 0.0, 0.0, 0.0, 1.0)
retval = dsl_display_type_rgba_font_new('arial-white-16', 'arial', 16, 'full-while')

retval = dsl_display_type_rgba_text_new('my-text', 
   'Hello World!', 100, 25, 'arial-white-16, True, 'full-black')
```

### *dsl_display_type_rgba_line_new*
```C++
DslReturnType dsl_display_type_rgba_line_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, const wchar_t* color);
```
The constructor creates a uniquely named **RGBA Line** Display type.

**Parameters**
* `name` - [in] unique name for the RGBA Display Line
* `x1` - [in] starting x positional offset
* `y1` - [in] starting y positional offset
* `x2` - [in] ending x positional offset
* `y2` - [in] ending y positional offset
* `width` - [in] line width in pixels
* `color` - [in] name of an RGBA Display Color for the Display Line

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-black', 0.0, 0.0, 0.0, 1.0)
retval = dsl_display_type_rgba_text_new('my-line', 100, 25, 250, 25, 2, 'full-black')
```

### *dsl_display_type_rgba_arrow_new*
```C++
DslReturnType dsl_display_type_rgba_arrow_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, const wchar_t* color);
```

The constructor creates a uniquely named **RGBA Arrow** Display type.

**Parameters**
* `name` - [in] unique name for the RGBA Display arrow
* `x1` - [in] starting x positional offset
* `y1` - [in] starting y positional offset
* `x2` - [in] ending x positional offset
* `y2` - [in] ending y positional offset
* `width` - [in] arrow line width in pixels
* `head` - [in] one of DSL_ARROW_START_HEAD, DSL_ARROW_END_HEAD, DSL_ARROW_BOTH_HEAD
* `color` - [in] name of an RGBA Display Color for the Display Arrow

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-black', 0.0, 0.0, 0.0, 1.0)
retval = dsl_display_type_rgba_arrow_new('my-line', 
  100, 25, 250, 25, 2, DSL_ARROW_START_HEAD, 'full-black')
```

### *dsl_display_type_rgba_rectangle_new*
```C++
 DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, uint left, uint top, uint width, uint height, 
    uint border_width, const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);
```
The constructor creates a uniquely named **RGBA Rectangle** Display type.

**Parameters**
* `name` - [in] unique name for the RGBA Rectangle
* `left` - [in] left x-positional offset
* `top` - [in] top y-positional offset
* `width` - [in] width of the rectangle in pixels
* `height` - [in] ending y positional offset
* `border_width` - [in] width of the rectangle border in pixels
* `color` - [in] name of an RGBA Display Color for the Rectangle border
* `has_bg_color` - [in] set to true to to fill the Rectangle's background
* `bg_color` - [in] name of an RGBA Display Color for the Rectangle's background color

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-black', 0.0, 0.0, 0.0, 1.0)
retval = dsl_display_type_rgba_rectangle_new('my-rectangle', 
  100, 25, 250, 350, 2, 'full-black', True, 'full-black)
```

### *dsl_display_type_rgba_circle_new*
```C++
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, uint x_center, uint y_center, uint radius,
    const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);
```
The constructor creates a uniquely named **RGBA Circle** Display type.

**Parameters**
* `name` - [in] unique name for the RGBA Circle
* `x_center` - [in] x-positional offset to center
* `y_center` - [in] y-positional offset to center
* `radius` - [in] radius for the Circle
* `color` - [in] name of an RGBA Display Color for the Cicrcle border
* `has_bg_color` - [in] set to true to to fill the Circle's background
* `bg_color` - [in] name of an RGBA Display Color for the Circle's background color

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_new('full-black', 0.0, 0.0, 0.0, 1.0)
retval = dsl_display_type_rgba_rectangle_new('my-circle', 
  100, 100, 25, 'full-black', True, 'full-black)
```

<br>

---

## Destructors
### *dsl_display_type_delete*
```C++
DslReturnType dsl_display_type_delete(const wchar_t* name);
```
This destructor deletes a single, uniquely named Display Type. The destructor will fail if the Display Type is currently `in-use` by one or more ODE Action

**Parameters**
* `names` - [in] unique name for the Display Types to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_delete('my-rectangle')
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
retval = dsl_display_type_delete_many(['my-line', 'my-rectangle', 'my-circle', None])
```

<br>

### *dsl_display_type_delete_all*
```C++
DslReturnType dsl_display_type_delete_all();
```
This destructor deletes all Display Types currently in memory. The destructor will fail if any of the Display Types are currently `in-use` by one or more ODE Actions. 

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
```C++
uint dsl_display_type_list_size();
```
This service returns the size of the Display Types container, i.e. the number of Types currently in memory. 

**Returns**
* The size of the Display Types container

**Python Example**
```Python
size = dsl_display_type_list_size()
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
* [ODE Handler](/docs/api-ode-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Action](/docs/api-ode-trigger.md)
* [ODE Area](/docs/api-ode-area.md)
* **Display Types**
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
