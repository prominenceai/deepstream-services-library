# ODE Area API Reference
ODE Areas -- derived from [RGBA Lines](/docs/api-display-type.md#dsl_display_type_rgba_line_new), [RGBA Multi-Lines](/docs/api-display-type.md#dsl_display_type_rgba_line_multi_new), and [RGBA Polygons](/docs/api-display-type.md#dsl_display_type_rgba_polygon_new) -- are added to [ODE Triggers](/docs/api-ode-trigger.md) as criteria for Object Detection Events (ODE) to occur.

There are three types of Areas:
* **Inclusion Areas** - when using an ODE Cross Trigger, criteria is met when a specific point of an object's bounding box - south, south-west, west, north-west, north, etc - fully crosses the one of the Polygon's segments. When using other types of Triggers, an ODE Occurrence Trigger for example, criteria is met when a specific point on an object's bounding box - south, south-west, west, north-west, north, etc - is within the Polygon Area.
* **Exclusion Areas** - criteria is met when a specific point on an object's bounding box is NOT within the Polygon Area.
* **Line Areas** - when using an ODE Cross Trigger, criteria is met when a specific point of an object's bounding box - south, south-west, west, north-west, north, etc - fully crosses the Trigger's Single or Multi-Line Area.

The relationship between Triggers and Areas is many-to-many as multiple Areas can be added to one Trigger and one Area can be added to multiple Triggers.  Once added to a Trigger, if an Area's `display` setting is enabled, the Polygon's or Line's metadata will be added to each frame for a downstream On-Screen-Component to display.

If both Areas of Inclusion and Exclusion are added to an ODE Trigger, the order of addition determines the order of precedence.

ODE Actions can be used to update a Trigger's container of ODE Areas on ODE occurrence. See [dsl_ode_action_area_add_new](/docs/api-ode-action.md#dsl_ode_action_area_add_new) and [dsl_ode_action_area_remove_new](/docs/api-ode-action.md#dsl_ode_action_area_remove_new).

#### ODE Area Construction and Destruction
Areas are created by calling one of four (4) type specific constructors: [dsl_ode_area_inclusion_new](#dsl_ode_area_inclusion_new), [dsl_ode_area_exclusion_new](#dsl_ode_area_exclusion_new), [dsl_ode_area_line_new](#dsl_ode_area_line_new), and [dsl_ode_area_line_multi_new](#dsl_ode_area_line_multi_new).

#### Adding/Removing ODE Areas
ODE Areas are added to to ODE Triggers by calling [dsl_ode_trigger_area_add](/docs/api-ode-trigger.md#dsl_ode_trigger_area_add) or [dsl_ode_trigger_area_add_many](/docs/api-ode-trigger.md#dsl_ode_trigger_area_add_many) and removed by [dsl_ode_trigger_area_remove](/docs/api-ode-trigger.md#dsl_ode_trigger_area_remove), [dsl_ode_trigger_area_remove_many](/docs/api-ode-trigger.md#dsl_ode_trigger_area_remove_many), or [dsl_ode_trigger_area_remove_all](/docs/api-ode-trigger.md#dsl_ode_trigger_area_remove_all).

## ODE Area Services API

**Constructors:**
* [dsl_ode_area_inclusion_new](#dsl_ode_area_inclusion_new)
* [dsl_ode_area_exclusion_new](#dsl_ode_area_exclusion_new)
* [dsl_ode_area_line_new](#dsl_ode_area_line_new)
* [dsl_ode_area_line_multi_new](#dsl_ode_area_line_multi_new)

**Destructors:**
* [dsl_ode_area_delete](#dsl_ode_area_delete)
* [dsl_ode_area_delete_many](#dsl_ode_area_delete_many)
* [dsl_ode_area_delete_all](#dsl_ode_area_delete_all)

**Methods:**
* [dsl_ode_area_list_size](#dsl_ode_area_list_size)

---

## Return Values
The following return codes are used by the OSD Area API
```C++
#define DSL_RESULT_ODE_AREA_RESULT                                  0x00100000
#define DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE                         0x00100001
#define DSL_RESULT_ODE_AREA_NAME_NOT_FOUND                          0x00100002
#define DSL_RESULT_ODE_AREA_THREW_EXCEPTION                         0x00100003
#define DSL_RESULT_ODE_AREA_IN_USE                                  0x00100004
#define DSL_RESULT_ODE_AREA_SET_FAILED                              0x00100005
```

## Constants
The following constants are used by the OSD Area API
```C++
#define DSL_BBOX_POINT_CENTER                                       0
#define DSL_BBOX_POINT_NORTH_WEST                                   1
#define DSL_BBOX_POINT_NORTH                                        2
#define DSL_BBOX_POINT_NORTH_EAST                                   3
#define DSL_BBOX_POINT_EAST                                         4
#define DSL_BBOX_POINT_SOUTH_EAST                                   5
#define DSL_BBOX_POINT_SOUTH                                        6
#define DSL_BBOX_POINT_SOUTH_WEST                                   7
#define DSL_BBOX_POINT_WEST                                         8
#define DSL_BBOX_POINT_ANY                                          9

```
<br>

---

## Constructors
### *dsl_ode_area_inclusion_new*
```C++
DslReturnType dsl_ode_area_inclusion_new(const wchar_t* name,
    const wchar_t* polygon, boolean show, uint bbox_test_point);
```
The constructor creates a uniquely named ODE **Area of Inclusion** using a uniquely named RGBA Polygon. Inclusion requires that a specified point on the Object's bounding box be within the Polygon Area to trigger ODE occurrence.

When using an [ODE Cross Trigger](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_new), ODE occurrence is triggered when a specified point on the Object's bounding box fully crosses one of the Polygon's segments. The specified line-width for the RGBA Polygon is used as line-cross hysteresis.

The Polygon can be shown (requires an [On-Screen Display](/docs/api-osd.md)) or left hidden.

**Parameters**
* `name` - [in] unique name for the ODE Inclusion Area to create.
* `polygon` - [in] unique name for the Polygon to use as coordinates and optionally display
* `show` - [in] if true, polygon metadata will be added to each structure of frame metadata.
* `bbox_test_point` - [in] one of the [DSL_BBOX_POINT Constants](#Constants) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_inclusion_new('my-inclusion-area', 'my-polygon', True, DSL_BBOX_POINT_SOUTH)
```

<br>

### *dsl_ode_area_exclusion_new*
```C++
DslReturnType dsl_ode_area_exclusion_new(const wchar_t* name,
    const wchar_t* polygon, boolean show, uint bbox_test_point);
```
The constructor creates a uniquely named ODE **Area of Exclusion** using a uniquely named RGBA Polygon. Exclusion requires that a specified point on the Object's bounding box is **not** within the Polygon Area to trigger ODE occurrence.

The Polygon can be shown (requires an [On-Screen Display](/docs/api-osd.md)) or left hidden.

**Parameters**
* `name` - [in] unique name for the ODE Exclusion Area to create.
* `polygon` - [in] unique name for the Polygon to use for coordinates and optionally display
* `show` - [in] if true, polygon metadata will be added to each structure of frame metadata.
* `bbox_test_point` - [in] one of the [DSL_BBOX_POINT Constants](#Constants) define above

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_exclusion_new('my-exclusion-area', 'my-polygon', True, DSL_BBOX_POINT_SOUTH)
```

<br>

### *dsl_ode_area_line_new*
```C++
DslReturnType dsl_ode_area_line_new(const wchar_t* name,
    const wchar_t* line, boolean show, uint bbox_test_point);
```
The constructor creates a uniquely named ODE **Line Area** using a uniquely named RGBA Line.  When using an [ODE Cross Trigger](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_new), ODE occurrence is triggered when a specified point on the Object's bounding box fully crosses the Line Area. The specified line-width for the RGBA Line is used as line-cross hysteresis.  

The Line can be shown (requires an [On-Screen Display](/docs/api-osd.md)) or left hidden.

**Parameters**
* `name` - [in] unique name for the ODE Line Area to create.
* `line` - [in] unique name for the Line to use for the Area's coordinates and for optional display
* `show` - [in] if true, polygon metadata will be added to each structure of frame metadata.
* `bbox_test_point` - [in] one of the [DSL_BBOX_POINT Constants](#Constants) defining which point of a object's bounding box to use when testing for line crossing
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_line_new('my-line-area',
    'my-line', True, DSL_BBOX_POINT_SOUTH)
```

<br>

### *dsl_ode_area_line_multi_new*
```C++
DslReturnType dsl_ode_area_line_multi_new(const wchar_t* name,
    const wchar_t* multi_line, boolean show, uint bbox_test_point);
```
The constructor creates a uniquely named ODE **Muli-Line Area** using a uniquely named RGBA Multi-Line. When using an [ODE Cross Trigger](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_new), ODE occurrence is triggered when a specified point on the Object's bounding box fully crosses the Line Area. The specified line-width for the RGBA Multi-Line is used as line-cross hysteresis.  

The Multi-Line Area can be shown (requires an [On-Screen Display](/docs/api-osd.md)) or left hidden.

**Parameters**
* `name` - [in] unique name for the ODE Multi-Line Area to create.
* `multi_line` - [in] unique name for the RGBA Multi-Line to use for the Area's coordinates and for optional display
* `show` - [in] if true, the multi-line metadata will be added to each structure of frame metadata.
* `bbox_test_point` - [in] one of the [DSL_BBOX_POINT Constants](#Constants) defining which point of an object's bounding box to use when testing for line crossing
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_line_multi_new('my-line-area',
    'my-multi-line', True, DSL_BBOX_POINT_SOUTH)
```

<br>

---

## Destructors
### *dsl_ode_area_delete*
```C++
DslReturnType dsl_ode_area_delete(const wchar_t* area);
```
This destructor deletes a single, uniquely named ODE Area. The destructor will fail if the Area is currently `in-use` by one or more ODE Triggers

**Parameters**
* `area` - [in] unique name for the ODE Area to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_delete('my-area')
```

<br>

### *dsl_ode_area_delete_many*
```C++
DslReturnType dsl_area_delete_many(const wchar_t** area);
```
This destructor deletes multiple uniquely named ODE Areas. Each name is checked for existence, with the function returning on first failure. The destructor will fail if one of the Areas is currently `in-use` by one or more ODE Triggers

**Parameters**
* `areas` - [in] a NULL terminated array of uniquely named ODE Areas to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_delete_many(['my-area-a', 'my-area-b', 'my-area-c', None])
```

<br>

### *dsl_ode_area_delete_all*
```C++
DslReturnType dsl_ode_area_delete_all();
```
This destructor deletes all ODE Areas currently in memory. The destructor will fail if any one of the Areas is currently `in-use` by one or more ODE Triggers.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_delete_all()
```

<br>

---

## Methods

### *dsl_ode_area_list_size*
```c++
uint dsl_ode_area_list_size();
```
This service returns the size of the ODE Area container, i.e. the number of Areas currently in memory.

**Returns**
* The size of the ODE Area container.

**Python Example**
```Python
size = dsl_ode_area_list_size()
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
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action](/docs/api-ode-action.md)
* **ODE-Area**
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
