# Display Type API
Display Types are used to add type specific metadata to a Frame's collection of metadata to be displayed downstream by an [On-Screen Display](/docs/api-osd.md). Display Types, once created, can be added to an [ODE Action](/docs/api-ode-action.md) in turn added to one or more [ODE Triggers](/docs/api-ode-trigger.md).  Each Trigger, on ODE occurrence, invokes the action to add the metadata to the current Frame's metadata that triggerd the event.

Further control of the Display Types can be achieved by enabling/disabling the Action or Trigger in a Client callback function when other events occur.  The start and end of a recording session for example. 

### Construction and Destruction
There are two base types used when creating other complete types for actual display. 
* RGBA Color
* RGBA Font

There are five types for displaying text and shapes. 
* RGBA Line
* RGBA Arrow
* RGBA Rectangle
* RGBA Polygon
* RGBA Circle

And three types for displaying source information specific to each frame. 
* Source Number
* Source Name
* Source Dimensions

Display Types are created by calling their type specific constructor. 

Display Typess are deleted by calling [dsl_display_type_delete](#dsl_display_type_delete), [dsl_display_type_delete_many](#dsl_display_type_delete_many), or [dsl_display_type_delete_all](#dsl_display_type_delete_all).

### Adding to an ODE Action. 
Display Types are added to a Display Action when the action is created by calling [dsl_ode_action_display_meta_add_new](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_new) or [dsl_ode_action_display_meta_add_many_new](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_many_new)

Note: Adding a Base Display Type to an ODE Action will fail. 

### Adding Lines and Polygons to ODE Areas.
RGBA Lines and Polygons are used to define [ODE Areas](/docs/api-ode-area.md) of criteria for one or more [ODE Triggers](/docs/api-ode-trigger.md). The Lines are used when calling [dsl_ode_area_line_new](/docs/api-od-area.md#dsl_ode_area_line_new) with Polygons used when calling [dsl_ode_area_inclusion_new](/docs/api-ode-area#dsl_ode_area_inclusion_new) and calling [dsl_ode_area_exclusion_new](/docs/api-ode-area#dsl_ode_area_exclusion_new)

## Using Display Types

### For static display on every frame:
To add static Display types to every frame, use a Display Meta Action -- [dsl_ode_action_display_meta_add_new](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_new) -- added to an [Always ODE Trigger](/docs/api-ode-trigger.md). 

Using Python for example
```Python
# new RGBA color Display Type
retval = dsl_display_type_rgba_color_new('full-blue', red=0.0, green=0.0, blue=1.0, alpha=1.0)

# new RGBA font using our new color
retval = dsl_display_type_rgba_font_new('arial-20-blue', font='arial', size=20, color='full-blue')

# new RGBA display text
retval = dsl_display_type_rgba_text_new('display-text', text='My Home Security', x_offset=733, y_offset=5, 
    font='arial-20-blue', has_bg_color=False, bg_color=None)

# Create an Action that will add the display text as metadata to the Frame's metadata
retval = dsl_ode_action_display_meta_add_new('add-display-text', display_type='display-text')

# new Always triger to add our display text on every frame, always
retval = dsl_ode_trigger_always_new('always-trigger', when=DSL_ODE_PRE_OCCURRENCE_CHECK)

# finally, add the "Add Display Meta" Action to our Always Trigger.
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

# Create an Action that will add the Rectangle to the Frame's metadata
retval = dsl_ode_action_display_meta_add_new('add-warning', display_type='warning-rectangle')

# new Maximum Objects triger to invoke our 'add-warning' action when to many objects are detected
retval = dsl_ode_trigger_maximum_new('max-trigger', class_id=PGIE_CLASS_ID_PERSON, limit=0, maximum=10)

# finally, add the "Add Display Meta" Action to our Max Objects Trigger.
retval = dsl_ode_trigger_action_add('max-trigger', action='overlay-warning')
```

#### For dynamic display using a Custom ODE Action:

```Python
# Callback function added to a Custom ODE Action which is added to a specific ODE Trigger
def handle_ode_occurrence(event_id, trigger, display_meta, frame_meta, object_meta, client_data):

    # cast the opaque client data back to a python object and dereference
    my_app_data = cast(client_data, POINTER(py_object)).contents.value

    # cast the opaque object_meta to a 
    py_object_meta = cast(object_meta, POINTER(py_object)).contents.value

    # create an Arrow to point to our object that triggered the event
    if (dsl_display_type_rgba_arrow_new('object-pointer`, 
        x1=py_object_meta->rect_params.left - 100,
        y1=py_object_meta->rect_params.top - 10,
        x2=py_object_meta->rect_params.left - 5,
        y2=py_object_meta->rect_params.top - 5,
        width=2,
        head=DSL_ARROW_END_HEAD,
        color='full-red') != DSL_RESULT_SUCCESS):
        
        # failed to create
        return
        
    dsl_display_type_meta_add('object-pointer', display_meta, frame_meta)
    
    dsl_display_type_delete('object-pointer')
```

---

### Display Type API

**Constructors:**
* [dsl_display_type_rgba_color_new](#dsl_display_type_rgba_color_new)
* [dsl_display_type_rgba_font_new](#dsl_display_type_rgba_font_new)
* [dsl_display_type_rgba_line_new](#dsl_display_type_rgba_line_new) 
* [dsl_display_type_rgba_arrow_new](#dsl_display_type_rgba_arrow_new)
* [dsl_display_type_rgba_rectangle_new](#dsl_display_type_rgba_rectangle_new)
* [dsl_display_type_rgba_polygon_new](#dsl_display_type_rgba_polygon_new)
* [dsl_display_type_rgba_circle_new](#dsl_display_type_rgba_circle_new)
* [dsl_display_type_source_number_new](#dsl_display_type_source_number_new)
* [dsl_display_type_source_name_new](#dsl_display_type_source_name_new)
* [dsl_display_type_source_dimensions_new](#dsl_display_type_source_dimensions_new)

**Destructors:**
* [dsl_display_type_delete](#dsl_display_type_delete)
* [dsl_display_type_delete_many](#dsl_display_type_delete_many) 
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
#define DSL_RESULT_DISPLAY_RGBA_POLYGON_NAME_NOT_UNIQUE             0x0010000E
#define DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE              0x0010000F
#define DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE            0x00100010
#define DSL_RESULT_DISPLAY_SOURCE_NAME_NAME_NOT_UNIQUE              0x00100011
#define DSL_RESULT_DISPLAY_SOURCE_DIMENSIONS_NAME_NOT_UNIQUE        0x00100012
#define DSL_RESULT_DISPLAY_SOURCE_FRAMERATE_NAME_NOT_UNIQUE         0x00100013
#define DSL_RESULT_DISPLAY_PARAMETER_INVALID                        0x00100014
```

## Constants
The following symbolic constants are used by the Display Type API
```C++
#define DSL_ARROW_START_HEAD                                        0
#define DSL_ARROW_END_HEAD                                          1
#define DSL_ARROW_BOTH_HEAD                                         2
#define DSL_MIN_POLYGON_COORDINATES                                 3
#define DSL_MAX_POLYGON_COORDINATES                                 8
```

## Types
The following types are used by the Display Type API
```C++
typedef struct _dsl_coordinate
{
    uint x;
    uint y;
} dsl_coordinate;
```
Defines a positional X,Y coordinate.

**Fields**
* `x` - coordinate value for the X dimension.
* `y` - coordinate value for the Y dimension.

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
* `red` - [in]red level for the RGB color [0..1].
* `blue` - [in] blue level for the RGB color [0..1].
* `green` - [in] green level for the RGB color [0..1].
* `alpha` - [in] alpha level for the RGB color [0..1].

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
* `fount` - [in] standard, unique string name of the actual font type (eg. 'arial').
* `size` - [in] size of the font.
* `color` - [in] name of the RGBA Color for the RGBA font.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_font_new('red-arial-18', 'arial', 18, 'full-red')
```

<br>

### *dsl_display_type_rgba_text_new* 
```C++
DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, const wchar_t* text, uint x_offset, 
    uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color);    
```
The constructor creates an RGBA Text Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `text` - [in] text string to display
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` [in] - RGBA font to use for the display dext
* `hasBgColor` - [in] set to true to enable background color, false otherwise
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
The constructor creates an RGBA Line Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x1` - [in] starting x positional offest.
* `y1` - [in] starting y positional offest.
* `x2` - [in] ending x positional offest.
* `y2` - [in] ending y positional offest.
* `width` - [in] width of the line in pixels.
* `color` - [in] RGBA Color for the RGBA Line.
 
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

The constructor creates an RGBA Arrow Display Type.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x1` - [in] starting x positional offest.
* `y1` - [in] starting y positional offest.
* `x2` - [in] ending x positional offest.
* `y2` - [in] ending y positional offest.
* `width` - [in] width of the line in pixels.
* `head` - [in] one of `DSL_ARROW_START_HEAD`, `DSL_ARROW_END_HEAD`, `DSL_ARROW_BOTH_HEAD`.
* `color` - [in] RGBA Color for the RGBA Arrow.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_arrow_new('arrow-pointer', 
    220, 165, 370, 235, 1, DSL_ARROW_END_HEAD, 'full-blue')
```

<br>

### *dsl_display_type_rgba_rectangle_new* 
```C++
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, uint left, uint top, uint width, 
    uint height, uint border_width, const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);
```

The constructor creates an RGBA Rectangle Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `left` - [in] left positional offest.
* `top` - [in] positional offest.
* `width` - [in] width of the rectangle in Pixels.
* `height` - [in] height of the rectangle in Pixels.
* `border_width` - [in] width of the rectangle border in pixels.
* `color` - [in] RGBA Color for this RGBA Rectangle.
* `hasBgColor` - [in] set to true to enable background color, false otherwise.
* `bgColor` - [in] RGBA Color for the Circle background if set.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_rectangle_new('black-rectangle', 
    240, 370, 1200, 940, 2, 'full-black', False, None)
```

<br>

### *dsl_display_type_rgba_polygon_new* 
```C++
DslReturnType dsl_display_type_rgba_polygon_new(const wchar_t* name, 
    const dsl_coordinate* coordinates, uint num_coordinates, uint border_width, const wchar_t* color);
```

The constructor creates an RGBA Polygon Display Type. The Poloygon supports up to DSL_MAX_POLYGON_COORDINATES (currently 8) defined in `DslApi.h`.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `coordinates` - [in] an array of positional coordinates defining the Polygon.
* `num_coordinates` - [in] number of positioanl coordinates that make up the Polygon.
* `border_width` - [in] width of the Polygon border in pixels.
* `color` - [in] RGBA Color for this RGBA Polygon.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
# create a list of X,Y coordinates defining the points of the Polygon.
coordinates = [dsl_coordinate(365,600), dsl_coordinate(580,620), 
    dsl_coordinate(600, 770), dsl_coordinate(180,750)]

retval = dsl_display_type_rgba_polygon_new('polygon1', coordinates, len(coordinates), 4, 'opaque-red')

```

<br>

### *dsl_display_type_rgba_circle_new* 
```C++
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, uint x_center, uint y_center, 
    uint radius, const wchar_t* color, bool has_bg_color, const wchar_t* bg_color);
```

The constructor creates an RGBA Circle Display Type.


**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_center` - [in] X positional offset to center of Circle.
* `y_center` - [in] y positional offset to center of Circle.
* `radius` - [in] radius of the RGBA Circle in pixels.
* `color` - [in] RGBA Color for the RGBA Circle.
* `hasBgColor` - [in] set to true to enable background color, false otherwise.
* `bgColor` - [in] RGBA Color for the Circle background if set.

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

The constructor creates a uniquely name Source Nuumber Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_offset` - [in] starting x positional offset.
* `y_offset` - [in] starting y positional offset.
* `font` - [in] RGBA font to use for the display text.
* `hasBgColor` - [in] set to true to enable background color, false otherwise.
* `bgColor` - [in] RGBA Color for the Text background if set.
 
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

The constructor creates a uniquely name Source Name Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_offset` - [in] starting x positional offset.
* `y_offset` - [in] starting y positional offset.
* `font` - [in] RGBA font to use for the display text.
* `hasBgColor` - [in] set to true to enable background color, false otherwise.
* `bgColor` - [in] RGBA Color for the Text background if set.
 
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

The constructor creates a uniquely name Source Dimensions Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x_offset` - [in] starting x positional offset.
* `y_offset` - [in] starting y positional offset.
* `font` - [in] RGBA font to use for the display text.
* `hasBgColor` - [in] set to true to enable background color, false otherwise.
* `bgColor` - [in] RGBA Color for the Text background if set.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_source_dimensions_new('display-source-dimensions', 
    10, 10, 'arial-blue-14', False, None)
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
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_delete('my-blue-circle')
```

<br>

### *dsl_display_type_delete_many*
```C++
DslReturnType dsl_display_type_delete_many(const wchar_t** names);
```
This destructor deletes multiple uniquely named Display Types. Each name is checked for existence with the function returning on first failure. The destructor will fail if one of the Display Types is currently `in-use` by one or more ODE Actions.

**Parameters**
* `names` - [in] a NULL terminated array of uniquely named Display Types to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

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
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

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
DslReturnType dsl_display_type_meta_add(const wchar_t* name, void* display_meta, void* frame_meta);
```
This service, when called from a custom [Pad Probe Handler](/docs/api-pph.md), adds the named Display Type as display-meta to the frame meta for the associated buffer.

**Parmeters**
* `name` - [in] unique name for the Display Type to add.
* `display_meta` - [in] opaque pointer to the acquired display meta to to add the Display Type to.
* `frame_meta` - [in] opaque pointer to a Frame's meta data to add the Display Type.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_meta_add('blue-circle', buffer, frame_meta)
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
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE-Trigger](/docs/api-ode-trigger.md)
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* **Display Types**
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
