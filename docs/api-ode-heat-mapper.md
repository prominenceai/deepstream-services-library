# ODE Heat-Mapper API Reference
An Object Detection Event (ODE) Heat-Mapper -- once added to an ODE Trigger -- accumulates ODE occurrence metrics over subsequent frames. When first constructed, the Heat-Mapper creates a two-dimensional (2D) vector of rows x columns as specified by the client. The width of each column is calculated as frame-width divided by the number of columns, and the height of each row is calculated as frame-height divided by the number of rows. Each entry in the 2D vector maps to a rectangular area within the video frame.  

The ODE Trigger, while post-processing each frame, calls on the ODE Accumulator to add the occurrence metrics as [Display metadata](/docs/api-display-type.md) to the current frame. All entries in the 2D vector with a least one occurrence will be added as an RGBA rectangle derived from a client provided [RGBA Color Palette](/docs/api-display-type.md). The color selected for each rectangle is based on the following simple distribution equation.

```
palette-index = round( vector[i][j] * (palette-size - 1) / most-occurrences )
```

#### Construction and Destruction
An ODE Heat-Mapper is created by calling [dsl_ode_heat_mapper_new](#dsl_ode_heat_mapper_new). Accumulators are deleted by calling [dsl_ode_heat_mapper_delete](#dsl_ode_heat_mapper_delete), [dsl_ode_heat_mapper_delete_many](#dsl_ode_heat_mapper_delete_many), or [dsl_ode_heat_mapper_delete_all](#dsl_ode_accumulator_delete_all).

#### Displaying a Map Legend
The Heat-Mapper can display a map legend derived from the RGBA Color Palette by calling [dsl_ode_heat_mapper_legend_settings_set](#dsl_ode_heat_mapper_legend_settings_set)

#### Adding and Removing Heat-Mappers
The relationship between ODE Triggers and ODE Heat-Mappers is one-to-one. A Trigger can have at most one Heat-Mapper and one Heat-mapper can be added to only on Trigger. An ODE Heat Mapper is added to an ODE Trigger by calling [dsl_ode_trigger_heat_mapper add](/docs/api-ode-trigger.md#dsl_ode_trigger_heat_mapper_add) and removed with [dsl_ode_trigger_heat_mapper_remove](docs/api-ode-trigger.md#dsl_ode_trigger_heat_mapper_remove).

---
## Examples
* [ode_occurrence_trigger_with_heat_mapper.py](/examples/python/ode_occurrence_trigger_with_heat_mapper.py) - a simple example that creates an [ODE Occurrence Trigger](/docs/api-ode-trigger.md#dsl_ode_trigger_occurrence_new) to trigger on each occurrence of an object with a `person` class Id and [minimum inference confidence](/docs/api-ode-trigger.md#dsl_ode_trigger_confidence_min_set).  An ODE Heat-Mapper, created with a [Predefined Spectral RGBA Color Palette](/docs/api-display-type.md#dsl_display_type_rgba_color_palette_predefined_new), is added to the ODE Occurrence Trigger producing a heat-map overlay as shown in the screen shot below. The example creates a set of predefined color palettes which can be cycled through by selecting the `N` key while the Pipeline is playing. The XWindow key handler function calls [dsl_ode_heat_mapper_color_palette_set](#dsl_ode_heat_mapper_color_palette_set) to change the Heat-Mapper's Palette.

![](/Images/spectral-person-heat-map.png)
---

## ODE Accumulator API
**Constructors:**
* [dsl_ode_heat_mapper_new](#dsl_ode_heat_mapper_new)

**Destructors:**
* [dsl_ode_heat_mapper_delete](#dsl_ode_heat_mapper_delete)
* [dsl_ode_heat_mapper_delete_many](#dsl_ode_heat_mapper_delete_many)
* [dsl_ode_heat_mapper_delete_all](#dsl_ode_heat_mapper_delete_all)

**Methods:**
* [dsl_ode_heat_mapper_color_palette_get](#dsl_ode_heat_mapper_color_palette_get)
* [dsl_ode_heat_mapper_color_palette_set](#dsl_ode_heat_mapper_color_palette_set)
* [dsl_ode_heat_mapper_legend_settings_get](#dsl_ode_heat_mapper_legend_settings_get)
* [dsl_ode_heat_mapper_legend_settings_set](#dsl_ode_heat_mapper_legend_settings_set)
* [dsl_ode_heat_mapper_metrics_clear](#dsl_ode_heat_mapper_metrics_clear)
* [dsl_ode_heat_mapper_metrics_get](#dsl_ode_heat_mapper_metrics_get)
* [dsl_ode_heat_mapper_metrics_print](#dsl_ode_heat_mapper_metrics_print)
* [dsl_ode_heat_mapper_metrics_log](#dsl_ode_heat_mapper_metrics_log)
* [dsl_ode_heat_mapper_metrics_file](#dsl_ode_heat_mapper_metrics_file)
* [dsl_ode_heat_mapper_list_size](#dsl_ode_heat_mapper_list_size)

---

## Return Values
The following return codes are used by the ODE Heat-Mapper API
```C++
#define DSL_RESULT_ODE_HEAT_MAPPER_RESULT                           0x00A00000
#define DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_UNIQUE                  0x00A00001
#define DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_FOUND                   0x00A00002
#define DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION                  0x00A00003
#define DSL_RESULT_ODE_HEAT_MAPPER_IN_USE                           0x00A00004
#define DSL_RESULT_ODE_HEAT_MAPPER_SET_FAILED                       0x00A00005
#define DSL_RESULT_ODE_HEAT_MAPPER_IS_NOT_ODE_HEAT_MAPPER           0x00A00006
```

## Constants
The following symbolic constants are used by the ODE Heat-Mapper API

### Bounding Box Test Points
Constants defining the point on the Object's bounding box to use when mapping an Object's location within the video frame.
```C
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

### Heat-Map Legend Locations
Constants defining the on-screen Heat-Map legend locations.
```C
#define DSL_HEAT_MAP_LEGEND_LOCATION_TOP                            0
#define DSL_HEAT_MAP_LEGEND_LOCATION_RIGHT                          1
#define DSL_HEAT_MAP_LEGEND_LOCATION_BOTTOM                         2
#define DSL_HEAT_MAP_LEGEND_LOCATION_LEFT                           3
```

### File Open-Write Modes
Constants defining the file open/write modes
```C
#define DSL_WRITE_MODE_APPEND                                       0
#define DSL_WRITE_MODE_TRUNCATE                                     1
```

### Output File Format Types
Constants defining the output file types supported
```C
#define DSL_EVENT_FILE_FORMAT_TEXT                                  0
#define DSL_EVENT_FILE_FORMAT_CSV                                   1
```

---

## Constructors
### *dsl_ode_heat_mapper_new*
```C++
DslReturnType dsl_ode_heat_mapper_new(const wchar_t* name,
    uint cols, uint rows, uint bbox_test_point, const wchar_t* color_palette);
```

The constructor creates a new ODE Heat-Mapper that when added to an ODE Trigger accumulates the count of ODE occurrence for each mapped-location within the video frame; column by row. The Heat-Mapper calculates the Object's location in the frame using the bounding-box's position and dimensions (left, top, width, height) and the `bbox_test_point` parameter. The ODE Trigger, while post-processing each frame, calls on the ODE Heat-Mapper to add the two-dimensional map as a collection of [RGBA Rectangles](/docs/api-display-type.md) defined with the provided `color_palette`.

**Parameters**
* `name` - [in] unique name for the ODE Heat-Mapper to create.
* `cols` - [in] number of columns for the two-dimensional map. Column width is calculated as frame-width devided by `cols`.
* `rows` - [in] number of rows for the two-dimensional map. Row height is calculated as frame-height devided by `rows`.
* `bbox_test_point` - [in] one of DSL_BBOX_POINT values defining which point of a object's bounding box to use as coordinates for mapping.
* `color_palette` - [in] color_palette a palette of RGBA Colors to assign to the map rectangles based on the number of occurrences for each map location.  

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_new('my-heat_mapper'
    cols = 64,
    rows = 36,
    bbox_test_point = DSL_BBOX_POINT_SOUTH,
    color_palette = 'my-color-palette')
```

<br>

---

## Destructors
### *dsl_ode_heat_mapper_delete*
```C++
DslReturnType dsl_ode_heat_mapper_delete(const wchar_t* name);
```
This destructor deletes a single, uniquely named ODE Heat-Mapper. The destructor will fail if the Heat-Mapper is currently `in-use` by an ODE Trigger.

**Parameters**
* `name` - [in] unique name for the ODE Heat-Mapper to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_delete('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_delete_many*
```C++
DslReturnType dsl_ode_heat_mapper_delete_many(const wchar_t** names);
```
This destructor deletes multiple uniquely named ODE Heat-Mappers. Each name is checked for existence with the service returning on first failure. The destructor will fail if one of the Heat-Mappers is currently `in-use` by an ODE Trigger.

**Parameters**
* `names` - [in] a NULL terminated array of uniquely named ODE Heat-Mappers to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_delete_many(['my-heat-mapper-a', 'my-heat-mapper-b', 'my-heat-mapper-c', None])
```

<br>

### *dsl_ode_heat_mapper_delete_all*
```C++
DslReturnType dsl_ode_heat_mapper_delete_all();
```
This destructor deletes all ODE Heat-Mappers currently in memory. The destructor will fail if any of the Heat-Mappers are currently `in-use` by an ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_heat_mapper_delete_all()
```

<br>

---

## Methods
### *dsl_ode_heat_mapper_color_palette_get*
```c++
DslReturnType dsl_ode_heat_mapper_color_palette_get(const wchar_t* name,
    const wchar_t** color_palette);
```

This service gets the name of the current RGBA Color Palette in use by the named ODE Heat-Mapper

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.
* `color_palette` - [out] unique name of the RGBA Color Palette currently in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, color_palette = dsl_ode_heat_mapper_color_palette_get('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_color_palette_set*
```c++
DslReturnType dsl_ode_heat_mapper_color_palette_set(const wchar_t* name,
    const wchar_t* color_palette);
```

This service sets the name of the GBA Color Palette to use for the named ODE Heat-Mapper.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to update.
* `color_palette` - [in] unique name of the new RGBA Color Palette to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, dsl_ode_heat_mapper_color_palette_set('my-heat-mapper',
  'my-blue-color-palette')
```

<br>

### *dsl_ode_heat_mapper_legend_settings_get*
```c++
DslReturnType dsl_ode_heat_mapper_legend_settings_get(const wchar_t* name,
    boolean* enabled, uint* location, uint* width, uint* height);
```

This service gets the current heat-map legend settings in use by the named ODE Heat-Mapper. The legend will be added one row or column away from the closest frame edge.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.
* `enabled` - [out] true if display is enabled, false otherwise.
* `location` - [out] one of the [Heat-Map Legend Location](heat-map-legend-locations) constants defined above.
* `width` - [out] width of each entry in the legend in units of columns.
* `height` - [out] height of each entry in the legend in units of rows.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled, location, width, height = dsl_ode_heat_mapper_legend_settings_get('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_legend_settings_set*
```c++
DslReturnType dsl_ode_heat_mapper_legend_settings_set(const wchar_t* name,
    boolean enabled, uint location, uint width, uint height);
```

This service gets the current heat-map legend settings in use by the named ODE Heat-Mapper.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.
* `enabled` - [in] set to true to enabled display, false otherwise.
* `location` - [in] one of the [Heat-Map Legend Locations](heat-map-legend-locations) constants defined above.
* `width` - [in] width of each entry in the legend in units of columns.
* `height` - [in] height of each entry in the legend in units of rows.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_legend_settings_set('my-heat-mapper',
  enabled = True,
  location = DSL_HEAT_MAP_LEGEND_LOCATION_TOP,
  width = 2,
  height = 2)
```

<br>

### *dsl_ode_heat_mapper_metrics_clear*
```c++
DslReturnType dsl_ode_heat_mapper_metrics_clear(const wchar_t* name);
```

This service clears the ODE Heat-Mapper's accumulated metrics.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_metrics_clear('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_metrics_get*
```c++
DslReturnType dsl_ode_heat_mapper_metrics_get(const wchar_t* name,
    const uint64_t** buffer, uint* size);
```

This service gets the ODE Heat-Mapper's accumulated metrics.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.
* `buffer` - [out] a linear buffer of metric map data. Each row of the map data is serialized into a single buffer of size columns x rows. Each element in the buffer indicates the total number of occurrences accumulated for the position in the map.
* `size` - [out] size of the linear buffer - columns x rows.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, buffer, size = dsl_ode_heat_mapper_metrics_get('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_metrics_print*
```c++
DslReturnType dsl_ode_heat_mapper_metrics_print(const wchar_t* name);
```

This service prints the ODE Heat-Mapper's accumulated metrics to the console window.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.

**Returns**
* `DSL_RESULT_SUCCESS` on successful print. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_metrics_print('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_metrics_log*
```c++
DslReturnType dsl_ode_heat_mapper_metrics_log(const wchar_t* name);
```

This service logs the ODE Heat-Mapper's accumulated metrics at a log-level of INFO.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.

**Returns**
* `DSL_RESULT_SUCCESS` on successful log. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_metrics_log('my-heat-mapper')
```

<br>

### *dsl_ode_heat_mapper_metrics_file*
```c++
DslReturnType dsl_ode_heat_mapper_metrics_file(const wchar_t* name,
    const wchar_t* file_path, uint mode, uint format);
```

This service writes the ODE Heat-Mapper's accumulated metrics to a text or comma-separated-values (csv) file.

**Parameters**
* `name` - [in] unique name of the ODE Heat-Mapper to query.
* `file_path` - [in] absolute or relative file path to the log file to use or create.
* `mode` - [in] one of the [File Open-Write Modes](#file-open-write-modes) defined above.
* `format` - [in] one of the [Output File Format Types](#output-file-format-types) defined above

**Returns**
* `DSL_RESULT_SUCCESS` on successful write. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_heat_mapper_metrics_file('my-heat-mapper',
    './metrics/person-map.txt', DSL_WRITE_MODE_TRUNCATE, DSL_EVENT_FILE_FORMAT_TEXT)
```

<br>

### *dsl_ode_heat_mapper_list_size*
```c++
uint dsl_ode_heat_mapper_list_size();
```

This service returns the size of the list of ODE Heat-Mappers.

**Returns**
* The current number of ODE Heat-Mappers in memory.

**Python Example**
```Python
size = dsl_ode_heat_mapper_list_size()
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
* [ODE Area](/docs/api-ode-area.md)
* **ODE Heat-Mapper**
* [Display Types](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)  
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
