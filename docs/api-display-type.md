# Display Type API
Display Types are used to add display metadata to a video frame's collection of metadata to be displayed downstream by an [On-Screen Display](/docs/api-osd.md). Display Types are be added to [ODE Actions](/docs/api-ode-action.md) which are added to [ODE Triggers](/docs/api-ode-trigger.md).  Each Trigger, on ODE occurrence, invokes the action to add the metadata to the current frame's metadata that triggered the event.

Further control of Display Types can be achieved by enabling/disabling the Action or Trigger in a Client callback function when other events occur.  The start and end of a recording session to enable/disable display of a `REC` symbol for example.

### Construction and Destruction
There are eight (8) base types used when creating other complete types for actual display, seven of which are RGBA Color Types. 
* **RGBA Custom Color** - a static color defined with red, green, blue, and alpha color values.
* **RGBA Predefined Color** - a static color defined from one of twenty (20) predefined colors, plus alpha value.
* **RGBA Random Color** - a dynamic color defined with optional hue and luminosity constraints, plus alpha value.
* **RGBA On-Demand Color** - a dynamic color defined with a client callback function to provide RGBA color values on demand.
* **RGBA Color Palette** - a dynamic palette of colors defined with two or more RGBA colors of any type.
* **RGBA Predefined Color Palette** - a dynamic palette of colors defined from one of five (5) predefined color palettes, plus alpha value.
* **RGBA Random Color Palette** - a dynamic palette of random colors defined with optional hue and luminosity constraints, plus alpha value.
* **RGBA Font** - defined with tty font name, size, and RGBA color.

There are seven types for displaying text and shapes.
* **RGBA Text** - defined with RGBA Font and optional RGBA background color.
* **RGBA Line** - defined with x,y end-point coordinates, width, and RGBA color.
* **RGBA Multi-Line** - defined with a set of x,y coordinates for each connecting point on the multi-line shape, line-width, and RGBA Color.
* **RGBA Arrow** - similar to RGBA Line, but with arrow head(s) defining direction.
* **RGBA Rectangle** - defined with left, top, width, height, border-width, and RGBA colors.
* **RGBA Polygon** - defined with a set of x,y coordinates for each connecting point on the polygon, border-width and RGBA Colors.
* **RGBA Circle** - defined with center point coordinates, radius, border-width, and RGBA Colors.

And three types for displaying source information on each frame.
* **Source Number** - based on the order the sources are added to the Pipeline, defined with RGBA Font and optional RGBA background color.
* **Source Name** - assigned to each source when created, defined with RGBA Font and optional RGBA background color.
* **Source Dimensions** - obtained from the frame dimensions in the frame metadata, defined with RGBA Font and optional RGBA background color.

<br>

The image below provides examples of the Display Types listed above.

![RGBA Display Types](/Images/display-types.png)

<br>

Display Types are created by calling their type specific [constructor](#constructors).

Display Types are deleted by calling [dsl_display_type_delete](#dsl_display_type_delete), [dsl_display_type_delete_many](#dsl_display_type_delete_many), or [dsl_display_type_delete_all](#dsl_display_type_delete_all).

### Adding to an ODE Action
Display Types are added to a Display Action when the action is created by calling [dsl_ode_action_display_meta_add_new](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_new) or [dsl_ode_action_display_meta_add_many_new](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_many_new)

Note: Adding a Base Display Type to an ODE Action will fail.

### Using Lines and Polygons to define ODE Areas
RGBA Lines and Polygons are used to define [ODE Areas](/docs/api-ode-area.md) as event criteria for one or more [ODE Triggers](/docs/api-ode-trigger.md). RGBA Lines are used when calling [dsl_ode_area_line_new](/docs/api-od-area.md#dsl_ode_area_line_new) and [dsl_ode_area_line_multi_new](/docs/api-od-area.md#dsl_ode_area_line_multi_new). RGBA Polygons are used when calling [dsl_ode_area_inclusion_new](/docs/api-ode-area#dsl_ode_area_inclusion_new) and calling [dsl_ode_area_exclusion_new](/docs/api-ode-area#dsl_ode_area_exclusion_new)

**Important:** The line width defined for the RGBA Lines and Polygons is used as hysteresis when tracking objects to determine if they cross over one of the Area's lines when used with an ODE Cross Trigger. A client specified point on the Object's bounding box must fully cross the line to trigger an ODE occurrence. See [dsl_ode_trigger_cross_new](/docs/api-ode-trigger-api.md#dsl_ode_trigger_cross_new) for more information.

### Coloring Tracked Objects
Dynamic RGBA colors can be used to uniquely color the bounding box and object trace of tracked objects as identified by a [Multi-object Tracker](/docs/api-tracker.md) when using an ODE Cross Trigger. See the [ODE Trigger API Reference](/docs/api-ode-trigger.md) and the [dsl_ode_trigger_cross_new](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_new) and [dsl_ode_trigger_cross_view_settings_set](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_view_settings_set) services for more information.

### Display Meta Memory Allocation
Display meta structures, allocated from pool memory, are used to attach the Display Type's metadata to a frame's metadata. Each display meta structure can hold up to 16 display elements for each display type (lines, arrows, rectangles, etc. Note: polygons require a line for each segment). The default allocation size is one structure per frame.  See [dsl_pph_ode_display_meta_alloc_size_set](/docs/api-pph.md#dsl_pph_ode_display_meta_alloc_size_set) if more than one structure per frame is required. Meta data will be discarded if sufficient memory has not allocated.

## Using Display Types
### For display on every frame:
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

# new Always trigger to add our display text on every frame, always
retval = dsl_ode_trigger_always_new('always-trigger', when=DSL_ODE_PRE_OCCURRENCE_CHECK)

# finally, add the "Add Display Meta" Action to our Always Trigger.
retval = dsl_ode_trigger_action_add('always-trigger', action='overlay-display-text')
```

#### For display on specific frames:
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

# new Maximum Objects trigger to invoke our 'add-warning' action when to many objects are detected
retval = dsl_ode_trigger_maximum_new('max-trigger', class_id=PGIE_CLASS_ID_PERSON, limit=0, maximum=10)

# finally, add the "Add Display Meta" Action to our Max Objects Trigger.
retval = dsl_ode_trigger_action_add('max-trigger', action='overlay-warning')
```

---

### Display Type API

**Types:**
* [dsl_coordinate](#dsl_coordinate)

**Client Callback Typedefs:**
* [dsl_display_type_rgba_color_provider_cb](#dsl_display_type_rgba_color_provider_cb)

**Constructors:**
* [dsl_display_type_rgba_color_custom_new](#dsl_display_type_rgba_color_custom_new)
* [dsl_display_type_rgba_color_predefined_new](#dsl_display_type_rgba_color_predefined_new)
* [dsl_display_type_rgba_color_random_new](#dsl_display_type_rgba_color_random_new)
* [dsl_display_type_rgba_color_on_demand_new](#dsl_display_type_rgba_color_on_demand_new)
* [dsl_display_type_rgba_color_palette_new](#dsl_display_type_rgba_color_palette_new)
* [dsl_display_type_rgba_color_palette_predefined_new](#dsl_display_type_rgba_color_palette_predefined_new)
* [dsl_display_type_rgba_color_palette_random_new](#dsl_display_type_rgba_color_palette_random_new)
* [dsl_display_type_rgba_font_new](#dsl_display_type_rgba_font_new)
* [dsl_display_type_rgba_text_new](#dsl_display_type_rgba_text_new)
* [dsl_display_type_rgba_line_new](#dsl_display_type_rgba_line_new)
* [dsl_display_type_rgba_line_multi_new](#dsl_display_type_rgba_line_multi_new)
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
* [dsl_display_type_rgba_color_palette_index_get](#dsl_display_type_rgba_color_palette_index_get)
* [dsl_display_type_rgba_color_palette_index_set](#dsl_display_type_rgba_color_palette_index_set)
* [dsl_display_type_rgba_color_next_set](#dsl_display_type_rgba_color_next_set)
* [dsl_display_type_rgba_text_shadow_add](#dsl_display_type_rgba_text_shadow_add)
* [dsl_display_type_list_size](#dsl_display_type_list_size)
* [dsl_display_type_meta_add](#dsl_display_type_meta_add)

---
## Return Values
The following return codes are used by the Display Type API
```C++
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE                     0x00200001
#define DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND                      0x00200002
#define DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION                     0x00200003
#define DSL_RESULT_DISPLAY_TYPE_IN_USE                              0x00200004
#define DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE                0x00200005
#define DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE                        0x00200006
#define DSL_RESULT_DISPLAY_PARAMETER_INVALID                        0x00200008
```

## Constants
The following symbolic constants are used by the Display Type API

### Predefined Color Ids
Referring to the color palette below, predefined colors are defined from left to right, top row, then bottom.

![](/Images/predefined-colors.png)

```C
#define DSL_COLOR_PREDEFINED_BLACK                                  0
#define DSL_COLOR_PREDEFINED_GRAY_50                                1
#define DSL_COLOR_PREDEFINED_DARK_RED                               2
#define DSL_COLOR_PREDEFINED_RED                                    3
#define DSL_COLOR_PREDEFINED_ORANGE                                 4
#define DSL_COLOR_PREDEFINED_YELLOW                                 5
#define DSL_COLOR_PREDEFINED_GREEN                                  6
#define DSL_COLOR_PREDEFINED_TURQUOISE                              7
#define DSL_COLOR_PREDEFINED_INDIGO                                 8
#define DSL_COLOR_PREDEFINED_PURPLE                                 9

#define DSL_COLOR_PREDEFINED_WHITE                                  10
#define DSL_COLOR_PREDEFINED_GRAY_25                                11
#define DSL_COLOR_PREDEFINED_BROWN                                  12
#define DSL_COLOR_PREDEFINED_ROSE                                   13
#define DSL_COLOR_PREDEFINED_GOLD                                   14
#define DSL_COLOR_PREDEFINED_LIGHT_YELLOW                           15
#define DSL_COLOR_PREDEFINED_LIME                                   16
#define DSL_COLOR_PREDEFINED_LIGHT_TURQUOISE                        17
#define DSL_COLOR_PREDEFINED_BLUE_GRAY                              18
#define DSL_COLOR_PREDEFINED_LAVENDER                               19
```

### Random Color Constraints
Constants used to constrain random colors to a particular hue. Use `DSL_COLOR_HUE_RANDOM` for no constraint.
```C
#define DSL_COLOR_HUE_RED                                           0
#define DSL_COLOR_HUE_RED_ORANGE                                    1
#define DSL_COLOR_HUE_ORANGE                                        2
#define DSL_COLOR_HUE_ORANGE_YELLOW                                 3
#define DSL_COLOR_HUE_YELLOW                                        4
#define DSL_COLOR_HUE_YELLOW_GREEN                                  5
#define DSL_COLOR_HUE_GREEN                                         6
#define DSL_COLOR_HUE_GREEN_CYAN                                    7
#define DSL_COLOR_HUE_CYAN                                          8
#define DSL_COLOR_HUE_CYAN_BLUE                                     9
#define DSL_COLOR_HUE_BLUE                                          10
#define DSL_COLOR_HUE_BLUE_MAGENTA                                  11
#define DSL_COLOR_HUE_MAGENTA                                       12
#define DSL_COLOR_HUE_MAGENTA_PINK                                  13
#define DSL_COLOR_HUE_PINK                                          14
#define DSL_COLOR_HUE_PINK_RED                                      15
#define DSL_COLOR_HUE_RANDOM                                        16
#define DSL_COLOR_HUE_BLACK_AND_WHITE                               17
#define DSL_COLOR_HUE_BROWN                                         18
```
Constants used to constrain random colors to a specific luminosity. Use `DSL_COLOR_LUMINOSITY_RANDOM` for no constraint. Refer to the luminosity scale below for examples.

```C
#define DSL_COLOR_LUMINOSITY_DARK                                   0
#define DSL_COLOR_LUMINOSITY_NORMAL                                 1
#define DSL_COLOR_LUMINOSITY_LIGHT                                  2
#define DSL_COLOR_LUMINOSITY_BRIGHT                                 3
#define DSL_COLOR_LUMINOSITY_RANDOM                                 4
```
![](/Images/luminosity.png)

### Predefined Color Palette Ids
Constants used to identify the available predefined RGBA Color Palettes.
```C
#define DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL                       0
#define DSL_COLOR_PREDEFINED_PALETTE_RED                            1
#define DSL_COLOR_PREDEFINED_PALETTE_GREEN                          2
#define DSL_COLOR_PREDEFINED_PALETTE_BLUE                           3
#define DSL_COLOR_PREDEFINED_PALETTE_GREY                           4
```

### Arrow Head Identifiers
The following constants are used to define the direction of a RGBA Arrow Display Type.
```C
#define DSL_ARROW_START_HEAD                                        0
#define DSL_ARROW_END_HEAD                                          1
#define DSL_ARROW_BOTH_HEAD                                         2
```

### Maximum Coordinates
Constants for the maximum number of coordinates when defining a Polygon or Multi-Line Display Type.

```C
#define DSL_MAX_POLYGON_COORDINATES                                 16
#define DSL_MAX_MULTI_LINE_COORDINATES                              16
```

## Types
The following types are used by the Display Type API
### *dsl_coordinate*
```C
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
## Client Callback Typedefs
### *dsl_display_type_rgba_color_provider_cb*
```C++
typedef void (*dsl_display_type_rgba_color_provider_cb)(double* red,
    double* green, double* blue, double* alpha, void* client_data);
```
Callback typedef for a client to provide RGBA color parameters on call from an [RGBA On-Demand Color Display Type](#dsl_display_type_rgba_color_on_demand_new).

**Parameters**
* `red` - [out] red level for the on-demand RGB color [0..1].
* `green` - [out] green level for the on-demand RGB color [0..1].
* `blue` - [out] blue level for the on-demand RGB color [0..1].
* `alpha` - [out] alpha level for the on-demand RGB color [0..1].
* `client_data` - [in] opaque pointer to client's user data.

<br>

---

## Constructors
### *dsl_display_type_rgba_color_custom_new*
```C
DslReturnType dsl_display_type_rgba_color_custom_new(const wchar_t* name,
    double red, double green, double blue, double alpha);
```
The constructor creates a RGBA Custom Color Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `red` - [in] red level for the RGB color [0..1].
* `blue` - [in] blue level for the RGB color [0..1].
* `green` - [in] green level for the RGB color [0..1].
* `alpha` - [in] alpha level for the RGB color [0..1].

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_custom_new('full-red', 1.0, 0.0, 0.0, 1.0)
```

<br>

### *dsl_display_type_rgba_color_predefined_new*
```C
DslReturnType dsl_display_type_rgba_color_predefined_new(const wchar_t* name,
    uint color_id, double alpha);
```
The constructor creates a RGBA Predefined Color Display Type. 

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `color_id` - [in] one of the [Predefined Color Id](#predefined-color-ids) constants defined above.
* `alpha` - [in] alpha level for the RGB color [0..1].

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_predefined_new('opaque-turquoise',
    DSL_COLOR_PREDEFINED_TURQUOISE, 0.4)
```

<br>

### *dsl_display_type_rgba_color_random_new*
```C
DslReturnType dsl_display_type_rgba_color_random_new(const wchar_t* name,
    uint hue, uint luminosity, double alpha, uint seed);
```
The constructor creates a Dynamic RGBA Random Color Display Type. The random RGB color values are regenerated when [dsl_display_type_rgba_color_next_set](#dsl_display_type_rgba_color_next_set) is called.

**Important:** Random colors can be used to uniquely color tracked objects as identified by a [Multi-object Tracker](/docs/api-tracker.md) when using an ODE Cross Trigger. See the [ODE Trigger API Reference](/docs/api-ode-trigger.md) and the [dsl_ode_trigger_cross_new](/docs/api-triger-api.md#dsl_ode_trigger_cross_new) and [dsl_ode_trigger_cross_view_settings_set](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_view_settings_set) services for more information.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `hue` - [in] one of the [Random Hue Color Constraint](#random-color-constraints) constants defined above. Use `DSL_COLOR_HUE_RANDOM` for no constraint.
* `luminosity` - [in] one of the [Random Luminosity Color Constraint](#random-color-constraints) constants defined above. Use `DSL_COLOR_LUMINOSITY_RANDOM` for no constraint.
* `alpha` - [in] alpha level for the RGB color [0..1].
* `seed` - [in] value to seed the random generator.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_random_new('random-dark-blue',
    DSL_COLOR_HUE_BLUE, DSL_COLOR_LUMINOSITY_DARK, 1.0, datetime.now())
```

<br>

### *dsl_display_type_rgba_color_on_demand_new*
```C
DslReturnType dsl_display_type_rgba_color_on_demand_new(const wchar_t* name,
    dsl_display_type_rgba_color_provider_cb provider, void* client_data);
```
The constructor creates a Dynamic RGBA On-Demand Color Display Type. The client provided callback is called on to provide RGBA color values when the Display Type is created and when [dsl_display_type_rgba_color_next_set](#dsl_display_type_rgba_color_next_set) is called aftwards.

**Important:** On-demand colors can be used to uniquely color tracked objects as identified by a [Multi-object Tracker](/docs/api-tracker.md) when using an ODE Cross Trigger. See the [ODE Trigger API Reference](/docs/api-ode-trigger.md) and the [dsl_ode_trigger_cross_new](/docs/api-triger-api.md#dsl_ode_trigger_cross_new) and [dsl_ode_trigger_cross_view_settings_set](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_view_settings_set) services for more information.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `provider` - [in] callback function of type [dsl_display_type_rgba_color_provider_cb](#dsl_display_type_rgba_color_provider_cb) to provide the next color values on demand.
* `client_data` - [in] opaque pointer to the client's user data, passed back on calls made to `provider`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_on_demand_new('on-demand-color',
    my_color_provider_cb, my_user_data)
```

<br>

### *dsl_display_type_rgba_color_palette_new*
```C
DslReturnType dsl_display_type_rgba_color_palette_new(const wchar_t* name,
    const wchar_t** colors);
```
The constructor creates a Dynamic RGBA Color Palette Display Type consisting of two or more RGBA Color Types. The RGBA Color is a base type used to create other RGBA types. The color's RGBA values are set to the next color in the palette when [dsl_display_type_rgba_color_next_set](#dsl_display_type_rgba_color_next_set) is called.

The color palette index can be queried and updated with calls to [dsl_display_type_rgba_color_palette_index_get](#dsl_display_type_rgba_color_palette_index_get) and [dsl_display_type_rgba_color_palette_index_set](#dsl_display_type_rgba_color_palette_index_set) respectively.

**Important:** a Color Palette can be used to uniquely color object bounding boxes and labels based on `class-id`. See the [ODE Action API Reference](/docs/api-ode-action.md) and the [dsl_ode_action_format_bbox_new](/docs/api-ode-action.md#dsl_ode_action_format_bbox_new) and [dsl_ode_action_format_label_new](/docs/api-ode-action.md#dsl_ode_action_format_label_new) services.

**Important:** a Color Palette can be used to uniquely color tracked objects as identifed by a [Multi-object Tracker](/docs/api-tracker.md) when using an ODE Cross Trigger. See the [ODE Trigger API Reference](/docs/api-ode-trigger.md) and the [dsl_ode_trigger_cross_new](/docs/api-triger-api.md#dsl_ode_trigger_cross_new) [dsl_ode_trigger_cross_view_settings_set](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_view_settings_set) services for more information.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `colors` - [in] a null terminated list of RGBA Color names.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_palette_new('my-color-palette',
    ['my-custom-red', 'my-predefined-blue', 'my-random-color', None])
```

<br>

### *dsl_display_type_rgba_color_palette_predefined_new*
```C
DslReturnType dsl_display_type_rgba_color_palette_predefined_new(const wchar_t* name,
    uint palette_id, double alpha);
```
The constructor creates a predefined Dynamic RGBA Color Palette Display Type . The RGBA Color is a base type used to create other RGBA types. The color's RGBA values are set to the next color in the palette when [dsl_display_type_rgba_color_next_set](#dsl_display_type_rgba_color_next_set) is called.

The color palette index can be queried and updated with calls to [dsl_display_type_rgba_color_palette_index_get](#dsl_display_type_rgba_color_palette_index_get) and [dsl_display_type_rgba_color_palette_index_set](#dsl_display_type_rgba_color_palette_index_set) respectively.

**Important:** a Color Palette can be used to uniquely color object bounding boxes and labels based on `class-id`. See the [ODE Action API Reference](/docs/api-ode-action.md) and the [dsl_ode_action_format_bbox_new](/docs/api-ode-action.md#dsl_ode_action_format_bbox_new) and [dsl_ode_action_format_label_new](/docs/api-ode-action.md#dsl_ode_action_format_label_new) services.

**Important:** a Color Palette can be used to uniquely color tracked objects as identified by a [Multi-object Tracker](/docs/api-tracker.md) when using an ODE Cross Trigger. See the [ODE Trigger API Reference](/docs/api-ode-trigger.md) and the [dsl_ode_trigger_cross_new](/docs/api-triger-api.md#dsl_ode_trigger_cross_new) and [dsl_ode_trigger_cross_view_settings_set](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_view_settings_set) services for more information.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `palette id` - [in] one of the [Predefined Color Palette Id](#predefined-color-palette-ids) constants defined above.
* `alpha` - [in] alpha level for the RGB colors [0..1].

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_palette_predefined_new('my-spectral-color-palette',
    DSL_COLOR_PREDEFINED_PALETTE_SPECTRAL, 0.4)
```

<br>

### *dsl_display_type_rgba_color_palette_random_new*
```C
DslReturnType dsl_display_type_rgba_color_palette_random_new(const wchar_t* name,
    uint size, uint hue, uint luminosity, double alpha, uint seed);
```
The constructor creates a Dynamic RGBA Random Color Palette Display Type. The RGBA Color is a base type used to create other RGBA types. The color's RGBA values are set to the next color in the palette when [dsl_display_type_rgba_color_next_set](#dsl_display_type_rgba_color_next_set) is called.

The color palette index can be queried and updated with calls to [dsl_display_type_rgba_color_palette_index_get](#dsl_display_type_rgba_color_palette_index_get) and [dsl_display_type_rgba_color_palette_index_set](#dsl_display_type_rgba_color_palette_index_set) respectively.

**Important:** a Color Palette can be used to uniquely color object bounding boxes and labels based on `class-id`. See the [ODE Action API Reference](/docs/api-ode-action.md) and the [dsl_ode_action_format_bbox_new](/docs/api-ode-action.md#dsl_ode_action_format_bbox_new) and [dsl_ode_action_format_label_new](/docs/api-ode-action.md#dsl_ode_action_format_label_new) services.

**Important:** a Color Palette can be used to uniquely color tracked objects as identifed by a [Multi-object Tracker](/docs/api-tracker.md) when using an ODE Cross Trigger. See the [ODE Trigger API Reference](/docs/api-ode-trigger.md) and the [dsl_ode_trigger_cross_new](/docs/api-triger-api.md#dsl_ode_trigger_cross_new) and [dsl_ode_trigger_cross_view_settings_set](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_view_settings_set) services for more information.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `size` - [in] size of the Color Palette to create
* `hue` - [in] one of the [Random Hue Color Constraint](#random-color-constraints) constants defined above. Use `DSL_COLOR_HUE_RANDOM` for no constraint.
* `luminosity` - [in] one of the [Random Luminosity Color Constraint](#random-color-constraints) constants defined above. Use `DSL_COLOR_LUMINOSITY_RANDOM` for no constraint.
* `alpha` - [in] alpha level for the RGB colors [0..1].
* `seed` - [in] value to seed the random generator.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_color_palette_random_new('my-spectral-color-palette',
     10, DSL_COLOR_HUE_BLUE, DSL_COLOR_LUMINOSITY_DARK, 1.0, datetime.now())
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
* `fount` - [in] standard, unique string name of the actual font type (e.g. 'arial').
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
The constructor creates a RGBA Text Display Type.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `text` - [in] text string to display
* `x_offset` - [in] starting x positional offset
* `y_offset` - [in] starting y positional offset
* `font` [in] - RGBA font to use for the display text
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
* `x1` - [in] starting x positional offset.
* `y1` - [in] starting y positional offset.
* `x2` - [in] ending x positional offset.
* `y2` - [in] ending y positional offset.
* `width` - [in] width of the line in pixels.
* `color` - [in] RGBA Color for the RGBA Line.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_display_type_rgba_line_new('dividing-line', 400, 10, 400, 700, 2, 'full-red')
```

<br>

### *dsl_display_type_rgba_line_multi_new*
```C++
DslReturnType dsl_display_type_rgba_line_multi_new(const wchar_t* name,
    const dsl_coordinate* coordinates, uint num_coordinates, uint line_width,
    const wchar_t* color);
```
The constructor creates a RGBA Multi Line Display Type. The Polygon supports up to `DSL_MAX_MULTI_LINE_COORDINATES`. See the [Maximum Coordinates](#maximum-coordinates) constants defined above.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `coordinates` - [in] array of [DSL coordinates](#dsl_coordinate) defining the multi-line
* `num_coordinates` - [in] the number of x,y coordinates in the array.
* `line_width` - [in] width of the multi-line in pixels.
* `color` - [in] Unique name of the RGBA Color to use for the multi-line.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
# create a list of X,Y coordinates defining the connecting points of the Multi-Line.
coordinates = [dsl_coordinate(365,600), dsl_coordinate(580,620),
    dsl_coordinate(600, 770), dsl_coordinate(822,750)]
 
retval = dsl_display_type_rgba_line_multi_new('dividing-multi-line',
    coordinates, len(coordinates), 4, 'full-red')
```

<br>

### *dsl_display_type_rgba_arrow_new*
```C++
DslReturnType dsl_display_type_rgba_arrow_new(const wchar_t* name,
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, const wchar_t* color);  
```

The constructor creates a RGBA Arrow Display Type.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `x1` - [in] starting x positional offset.
* `y1` - [in] starting y positional offset.
* `x2` - [in] ending x positional offset.
* `y2` - [in] ending y positional offset.
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

The constructor creates a RGBA Rectangle Display Type.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `left` - [in] left positional offset.
* `top` - [in] positional offset.
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

The constructor creates a RGBA Polygon Display Type. The Polygon supports up to `DSL_MAX_POLYGON_COORDINATES`. See the [Maximum Coordinates](#maximum-coordinates) constants defined above.

**Parameters**
* `name` - [in] unique name for the Display Type to create.
* `coordinates` - [in] an array of positional coordinates defining the Polygon.
* `num_coordinates` - [in] number of positional coordinates that make up the Polygon.
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

The constructor creates a RGBA Circle Display Type.


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

The constructor creates a uniquely name Source Number Display Type.

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
### *dsl_display_type_rgba_color_palette_index_get*
```C++
DslReturnType dsl_display_type_rgba_color_palette_index_get(const wchar_t* name,
    uint* index);
```
This service gets the current index value for the active RGBA color for the named RGBA Color Palette Display Type.

**Parameters**
* `name` - [in] unique name for the RGBA Color Palette to query.
* `index` - [out] the current index into the Color Palette's array of RGBA colors..

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, index = dsl_display_type_rgba_color_palette_index_get('my-color-palette')
```

<br>

### *dsl_display_type_rgba_color_palette_index_set*
```C++
DslReturnType dsl_display_type_rgba_color_palette_index_get(const wchar_t* name,
    uint index);
```
This service sets the index value for the active RGBA color for the named RGBA Color Palette Display Type. All Display Types created with the named RGBA Color Palette will be updated with the new color, visible on the next frame when displayed.

**Parameters**
* `name` - [in] unique name for the RGBA Color Palette to query.
* `index` - [in] the current index into the Color Palette's array of RGBA colors..

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_rgba_color_palette_index_set('my-color-palette', new_index)
```

<br>

### *dsl_display_type_rgba_color_next_set*
```C++
DslReturnType dsl_display_type_rgba_color_next_set(const wchar_t* name);
```
This service sets a dynamic RGBA color type -- [Random](#dsl_display_type_rgba_color_random_new), [On-Demand](#dsl_display_type_rgba_color_on_demand_new), and [Color Palette](#dsl_display_type_rgba_color_palette_new) types -- to their next color. Random display types will generate a new color, On-Demand colors will call their client provided [callback function](dsl_display_type_rgba_color_provider_cb), and Color palettes will increment their color index to the next color.

**Parameters**
* `name` - [in] unique name for the dynamic RGBA Color to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_rgba_color_next_set('my-random-color')
```

<br>

### *dsl_display_type_rgba_text_shadow_add*
```C++
DslReturnType dsl_display_type_rgba_text_shadow_add(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* color);
```
This service adds a shadow to any one of the text-based Display Types - by duplicating and underlaying the text at an x, y offset - creating a raised effect. This service applies to the [RGBA Text](#dsl_display_type_rgba_text_new), [Source Number](#dsl_display_type_source_number_new), [Source Name](#dsl_display_type_source_name_new), and [Source Dimensions](#dsl_display_type_source_dimensions_new) Display Types.

**Parameters**
* `name` - [in] unique name for the RGBA Color Palette to query.
* `x_offset` - [in] shadow offset in the x direction in units of pixels.
* `y_offset` - [in] shadow offset in the y direction in units of pixels.
* `color` - [in] name of the RGBA Color to use for the text shadow.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_display_type_rgba_text_shadow_add('my-rgba-text', 
    10, 10, 'my-opaque-black')
```

<br>

### *dsl_display_type_list_size*
```c++
uint dsl_display_type_list_size();
```
This service returns the size of the display_type container, i.e. the number of Display Types currently in memory.

**Returns**
* The size of the Display Types container.

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
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE-Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* **Display Type**
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
