# ODE Trigger API Reference
ODE Triggers use settable criteria to parse the frame and detected-object metadata looking for occurrences of specific "object detection events" (ODEs): Occurrence, Absence, Intersection, etc. Triggers, on ODE occurrence, iterate through their collection of ordered [ODE Actions](/docs/api-ode-action.md) invoking each.

[ODE Areas](/docs/api-ode-area.md) -- created from [RGBA Lines](/docs/api-display-type.md#dsl_display_type_rgba_line_new), [RGBA Mulit-Lines](/docs/api-display-type.md#dsl_display_type_rgba_line_new), and [RGBA Polygons](/docs/api-display-type.md#dsl_display_type_rgba_polygon_new) -- can be added to one or more Triggers as criteria for ODE occurrence as well.

#### Construction and Destruction
Triggers are created by calling one of the Type specific [constructors](#constructors) defined below. Triggers are deleted by calling [dsl_ode_trigger_delete](#dsl_ode_trigger_delete), [dsl_ode_trigger_delete_many](#dsl_ode_trigger_delete_many), or [dsl_ode_trigger_delete_all](#dsl_ode_trigger_delete_all).

#### Adding and Removing Triggers
The relationship between ODE Pad Prop Handlers and ODE Triggers is one-to-many. A Trigger must be removed from a Handler before it can be used by another. Triggers are added to a handler by calling [dsl_pph_ode_trigger_add](docs/api-pph.md#dsl_pph_ode_trigger_add) and [dsl_pph_ode_trigger_add_many](docs/api-pph.md#dsl_pph_ode_trigger_add_many), and removed with [dsl_pph_ode_trigger_remove](docs/api-pph.md#dsl_pph_ode_trigger_remove), [dsl_pph_ode_trigger_remove_many](docs/api-pph.md#dsl_pph_ode_trigger_remove_many), and [dsl_pph_ode_trigger_remove_all](docs/api-pph.md#dsl_pph_ode_trigger_remove_all).

#### Adding and Removing Actions
Multiple ODE Actions can be added to an ODE Trigger and the same ODE Action can be added to multiple Triggers.  ODE Actions are added to an ODE Trigger by calling [dsl_ode_trigger_action_add](#dsl_ode_trigger_action_add) and [dsl_ode_trigger_action_add_many](#dsl_ode_trigger_action_add_many), and removed with [dsl_ode_trigger_action_remove](#dsl_ode_trigger_action_remove), [dsl_ode_trigger_action_remove_many](#dsl_ode_trigger_action_remove_many), and [dsl_ode_trigger_action_remove_all](#dsl_ode_trigger_action_remove_all).

#### Adding and Removing Areas
As with Actions, multiple ODE areas can be added to an ODE Trigger and the same ODE Areas can be added to multiple Triggers. ODE Areas are added to an ODE Trigger by calling [dsl_ode_trigger_area_add](#dsl_ode_trigger_area_add) and [dsl_ode_trigger_area_add_many](#dsl_ode_trigger_area_add_many) and removed with [dsl_ode_trigger_action_remove](#dsl_ode_trigger_area_remove), [dsl_ode_trigger_area_remove_many](#dsl_ode_trigger_area_remove_many), and [dsl_ode_trigger_area_remove_all](#dsl_ode_trigger_area_remove_all).

#### Adding and Removing an Accumulator
A single ODE Accumulator can be added to an ODE Trigger and the same ODE Accumulator can be added to multiple Triggers. An ODE Accumulator is added to an ODE Trigger by calling [dsl_ode_trigger_accumulator_add](#dsl_ode_trigger_accumulator_add) and removed with [dsl_ode_trigger_accumulator_remove](#dsl_ode_trigger_accumulator_remove). See the [ODE Accumulator API Reference](/docs/api-ode-accumulator.md) for additional information.

#### Adding and Removing a Heat Mapper
A single ODE Heat-Mapper can be added to a single ODE Trigger. An ODE Heat-Mapper is added to an ODE Trigger by calling [dsl_ode_trigger_heat_mapper_add](#dsl_ode_trigger_heat_mapper_add) and removed with [dsl_ode_trigger_heat_mapper_remove](#dsl_ode_trigger_heat_mapper_remove). See the [ODE Heat-Mapper API Reference](/docs/api-ode-heat-mapper.md) for additional information.

**Important** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as these operations can consume all available diskspace.

---

## ODE Trigger API
**Callback Typedefs:**
* [dsl_ode_check_for_occurrence_cb](#dsl_ode_check_for_occurrence_cb)
* [dsl_ode_enabled_state_change_listener_cb](#dsl_ode_enabled_state_change_listener_cb)
* [dsl_ode_trigger_limit_event_listener_cb](#dsl_ode_trigger_limit_event_listener_cb)

**Constructors:**
* [dsl_ode_trigger_always_new](#dsl_ode_trigger_always_new)
* [dsl_ode_trigger_absence_new](#dsl_ode_trigger_absence_new)
* [dsl_ode_trigger_custom_new](#dsl_ode_trigger_custom_new)
* [dsl_ode_trigger_occurrence_new](#dsl_ode_trigger_occurrence_new)
* [dsl_ode_trigger_instance_new](#dsl_ode_trigger_instance_new)
* [dsl_ode_trigger_summation_new](#dsl_ode_trigger_summation_new)
* [dsl_ode_trigger_distance_new](#dsl_ode_trigger_distance_new)
* [dsl_ode_trigger_intersection_new](#dsl_ode_trigger_intersection_new)
* [dsl_ode_trigger_count_new](#dsl_ode_trigger_count_new)
* [dsl_ode_trigger_new_high_new](#dsl_ode_trigger_new_high_new)
* [dsl_ode_trigger_new_low_new](#dsl_ode_trigger_new_low_new)
* [dsl_ode_trigger_smallest_new](#dsl_ode_trigger_smallest_new)
* [dsl_ode_trigger_largest_new](#dsl_ode_trigger_largest_new)
* [dsl_ode_trigger_cross_new](#dsl_ode_trigger_cross_new)
* [dsl_ode_trigger_persistence_new](#dsl_ode_trigger_persistence_new)
* [dsl_ode_trigger_earliest_new](#dsl_ode_trigger_earliest_new)
* [dsl_ode_trigger_latest_new](#dsl_ode_trigger_latest_new)

**Destructors:**
* [dsl_ode_trigger_delete](#dsl_ode_trigger_delete)
* [dsl_ode_trigger_delete_many](#dsl_ode_trigger_delete_many)
* [dsl_ode_trigger_delete_all](#dsl_ode_trigger_delete_all)

**Methods:**
* [dsl_ode_trigger_count_range_get](#dsl_ode_trigger_count_range_get)
* [dsl_ode_trigger_count_range_set](#dsl_ode_trigger_count_range_set)
* [dsl_ode_trigger_distance_range_get](#dsl_ode_trigger_distance_range_get)
* [dsl_ode_trigger_distance_range_set](#dsl_ode_trigger_distance_range_set)
* [dsl_ode_trigger_distance_test_params_get](#dsl_ode_trigger_distance_test_params_get)
* [dsl_ode_trigger_distance_test_params_set](#dsl_ode_trigger_distance_test_params_set)
* [dsl_ode_trigger_cross_test_settings_get](#dsl_ode_trigger_cross_test_settings_get)
* [dsl_ode_trigger_cross_test_settings_set](#dsl_ode_trigger_cross_test_settings_set)
* [dsl_ode_trigger_cross_view_settings_get](#dsl_ode_trigger_cross_view_settings_get)
* [dsl_ode_trigger_cross_view_settings_set](#dsl_ode_trigger_cross_view_settings_set)
* [dsl_ode_trigger_instance_count_settings_get](#dsl_ode_trigger_instance_count_settings_get)
* [dsl_ode_trigger_instance_count_settings_set](#dsl_ode_trigger_instance_count_settings_set)
* [dsl_ode_trigger_persistence_range_get](#dsl_ode_trigger_persistence_range_get)
* [dsl_ode_trigger_persistence_range_set](#dsl_ode_trigger_persistence_range_set)
* [dsl_ode_trigger_reset](#dsl_ode_trigger_reset)
* [dsl_ode_trigger_reset_timeout_get](#dsl_ode_trigger_reset_timeout_get)
* [dsl_ode_trigger_reset_timeout_set](#dsl_ode_trigger_reset_timeout_set)
* [dsl_ode_trigger_enabled_get](#dsl_ode_trigger_enabled_get)
* [dsl_ode_trigger_enabled_set](#dsl_ode_trigger_enabled_set)
* [dsl_ode_trigger_enabled_state_change_listener_add](#dsl_ode_trigger_enabled_state_change_listener_add)
* [dsl_ode_trigger_enabled_state_change_listener_remove](#dsl_ode_trigger_enabled_state_change_listener_remove)
* [dsl_ode_trigger_source_get](#dsl_ode_trigger_source_get)
* [dsl_ode_trigger_source_set](#dsl_ode_trigger_source_set)
* [dsl_ode_trigger_class_id_get](#dsl_ode_trigger_class_id_get)
* [dsl_ode_trigger_class_id_set](#dsl_ode_trigger_class_id_set)
* [dsl_ode_trigger_class_id_ab_get](#dsl_ode_trigger_class_id_ab_get)
* [dsl_ode_trigger_class_id_ab_set](#dsl_ode_trigger_class_id_ab_set)
* [dsl_ode_trigger_limit_event_get](#dsl_ode_trigger_limit_event_get)
* [dsl_ode_trigger_limit_event_set](#dsl_ode_trigger_limit_event_set)
* [dsl_ode_trigger_limit_frame_get](#dsl_ode_trigger_limit_frame_get)
* [dsl_ode_trigger_limit_frame_set](#dsl_ode_trigger_limit_frame_set)
* [dsl_ode_trigger_limit_state_change_listener_add](#dsl_ode_trigger_limit_state_change_listener_add)
* [dsl_ode_trigger_limit_state_change_listener_remove](#dsl_ode_trigger_limit_state_change_listener_remove)
* [dsl_ode_trigger_infer_confidence_min_get](#dsl_ode_trigger_infer_confidence_min_get)
* [dsl_ode_trigger_infer_confidence_min_set](#dsl_ode_trigger_infer_confidence_min_set)
* [dsl_ode_trigger_infer_confidence_max_get](#dsl_ode_trigger_infer_confidence_max_get)
* [dsl_ode_trigger_infer_confidence_max_set](#dsl_ode_trigger_infer_confidence_max_set)
* [dsl_ode_trigger_tracker_confidence_min_get](#dsl_ode_trigger_tracker_confidence_min_get)
* [dsl_ode_trigger_tracker_confidence_min_set](#dsl_ode_trigger_tracker_confidence_min_set)
* [dsl_ode_trigger_tracker_confidence_max_get](#dsl_ode_trigger_tracker_confidence_max_get)
* [dsl_ode_trigger_tracker_confidence_max_set](#dsl_ode_trigger_tracker_confidence_max_set)
* [dsl_ode_trigger_dimensions_min_get](#dsl_ode_trigger_dimensions_min_get)
* [dsl_ode_trigger_dimensions_min_set](#dsl_ode_trigger_dimensions_min_set)
* [dsl_ode_trigger_dimensions_max_get](#dsl_ode_trigger_dimensions_max_get)
* [dsl_ode_trigger_dimensions_max_set](#dsl_ode_trigger_dimensions_max_set)
* [dsl_ode_trigger_infer_done_only_get](#dsl_ode_trigger_infer_done_only_get)
* [dsl_ode_trigger_infer_done_only_set](#dsl_ode_trigger_infer_done_only_set)
* [dsl_ode_trigger_interval_get](#dsl_ode_trigger_interval_get)
* [dsl_ode_trigger_interval_set](#dsl_ode_trigger_interval_set)
* [dsl_ode_trigger_action_add](#dsl_ode_trigger_action_add)
* [dsl_ode_trigger_action_add_many](#dsl_ode_trigger_action_remove_many)
* [dsl_ode_trigger_action_remove](#dsl_ode_trigger_action_add)
* [dsl_ode_trigger_action_remove_many](#dsl_ode_trigger_action_remove_many)
* [dsl_ode_trigger_action_remove_all](#dsl_ode_trigger_action_remove_all)
* [dsl_ode_trigger_area_add](#dsl_ode_trigger_area_add)
* [dsl_ode_trigger_area_add_many](#dsl_ode_trigger_area_remove_many)
* [dsl_ode_trigger_area_remove](#dsl_ode_trigger_area_add)
* [dsl_ode_trigger_area_remove_many](#dsl_ode_trigger_area_remove_many)
* [dsl_ode_trigger_area_remove_all](#dsl_ode_trigger_area_remove_all)
* [dsl_ode_trigger_accumulator_add](#dsl_ode_trigger_accumulator_add)
* [dsl_ode_trigger_accumulator_remove](#dsl_ode_trigger_accumulator_remove)
* [dsl_ode_trigger_heat_mapper_add](#dsl_ode_trigger_heat_mapper_add)
* [dsl_ode_trigger_heat_mapper_remove](#dsl_ode_trigger_heat_mapper_remove)
* [dsl_ode_trigger_list_size](#dsl_ode_trigger_list_size)

---
## Return Values
The following return codes are used by the ODE Trigger API
```C++
#define DSL_RESULT_ODE_TRIGGER_RESULT                               0x000E0000
#define DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE                      0x000E0001
#define DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND                       0x000E0002
#define DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION                      0x000E0003
#define DSL_RESULT_ODE_TRIGGER_IN_USE                               0x000E0004
#define DSL_RESULT_ODE_TRIGGER_SET_FAILED                           0x000E0005
#define DSL_RESULT_ODE_TRIGGER_IS_NOT_ODE_TRIGGER                   0x000E0006
#define DSL_RESULT_ODE_TRIGGER_ACTION_ADD_FAILED                    0x000E0007
#define DSL_RESULT_ODE_TRIGGER_ACTION_REMOVE_FAILED                 0x000E0008
#define DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE                    0x000E0009
#define DSL_RESULT_ODE_TRIGGER_AREA_ADD_FAILED                      0x000E000A
#define DSL_RESULT_ODE_TRIGGER_AREA_REMOVE_FAILED                   0x000E000B
#define DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE                      0x000E000C
#define DSL_RESULT_ODE_TRIGGER_CALLBACK_ADD_FAILED                  0x000F000D
#define DSL_RESULT_ODE_TRIGGER_CALLBACK_REMOVE_FAILED               0x000F000E
#define DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID                    0x000E000F
#define DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE                       0x000E0010
#define DSL_RESULT_ODE_TRIGGER_IS_NOT_TRACK_TRIGGER                 0x000E0011
#define DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_ADD_FAILED               0x000E0012
#define DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_REMOVE_FAILED            0x000E0013
#define DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_ADD_FAILED               0x000E0014
#define DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_REMOVE_FAILED            0x000E0015
```

---

## Constants
The following symbolic constants are used by the ODE Trigger API

#### Source and Class Trigger filter constants for no-filter
```C++
#define DSL_ODE_ANY_SOURCE                                          INT32_MAX
#define DSL_ODE_ANY_CLASS                                           INT32_MAX
#define DSL_ODE_TRIGGER_LIMIT_NONE                                  0
#define DSL_ODE_TRIGGER_LIMIT_ONE                                   1
```

#### ODE Trigger limit state values - for Triggers with limits
```C
#define DSL_ODE_TRIGGER_LIMIT_EVENT_REACHED                         0
#define DSL_ODE_TRIGGER_LIMIT_EVENT_CHANGED                         1
#define DSL_ODE_TRIGGER_LIMIT_FRAME_REACHED                         2
#define DSL_ODE_TRIGGER_LIMIT_FRAME_CHANGED                         3
#define DSL_ODE_TRIGGER_LIMIT_COUNTS_RESET                          4
```

#### Constants that define a Point's location relative to an ODE Area.
```C
#define DSL_AREA_POINT_LOCATION_ON_LINE                             0
#define DSL_AREA_POINT_LOCATION_INSIDE                              1
#define DSL_AREA_POINT_LOCATION_OUTSIDE                             2
```

#### Constants to define the ODE Area Line Cross directions.
```c
#define DSL_AREA_CROSS_DIRECTION_NONE                               0
#define DSL_AREA_CROSS_DIRECTION_IN                                 1
#define DSL_AREA_CROSS_DIRECTION_OUT                                2
```

#### Methods of testing Object-Trace Area Line Crossing
```c
#define DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS                     0
#define DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS                     1
```

#### Constants specifying a set of defined points along a bounding box border.
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

#### Methods of calculating distance between object bounding boxes
```C
#define DSL_DISTANCE_METHOD_FIXED_PIXELS                            0
#define DSL_DISTANCE_METHOD_PERCENT_WIDTH_A                         1
#define DSL_DISTANCE_METHOD_PERCENT_WIDTH_B                         2
#define DSL_DISTANCE_METHOD_PERCENT_HEIGHT_A                        3
#define DSL_DISTANCE_METHOD_PERCENT_HEIGHT_B                        4

```

---

## Callback Typedefs
### *dsl_ode_check_for_occurrence_cb*
```C++
typedef boolean (*dsl_ode_check_for_occurrence_cb)(void* buffer,
    void* frame_meta, void* object_meta, void* client_data);
```
Defines a Callback typedef for a Custom ODE Trigger. Once registered, the function will be called on every object detected that meets the criteria for the Custom Trigger. The client, determining that **custom** criteria have been met, returns true signaling ODE occurrence. The Custom Trigger will then invoke all the client provided Actions.

**Parameters**
* `buffer` - [in] pointer to frame buffer containing the Metadata for the object detected.
* `frame_meta` - [in] opaque pointer to a frame_meta structure that triggered the ODE event.
* `object_meta` - [in] opaque pointer to an object_meta structure that triggered the ODE event.
* `client_data` - [in] opaque point to client user data provided by the client on callback registration

<br>

### *dsl_ode_enabled_state_change_listener_cb*
```C++
 typedef void (*dsl_ode_enabled_state_change_listener_cb)
    (boolean enabled, void* client_data)
```
Defines a Callback typedef for a client listener function. Once added to an ODE Trigger, this function will be called on every change of the Trigger's enabled state.

**Parameters**
* `enabled` - [in] true if the Trigger has been enabled, false if disabled.
* `client_data` - [in] opaque point to client user data provided by the client on callback registration.

<br>

### *dsl_ode_trigger_limit_event_listener_cb*
```C++
 typedef void (*dsl_ode_trigger_limit_event_listener_cb)
    (uint event, uint limit, void* client_data);
```
Defines a Callback typedef for a client listener function. Once added to an ODE Trigger, this function will be called on every trigger limit event; `LIMIT_REACHED`, `LIMIT_CHANGED`, `COUNT_RESET`;

**Parameters**
* `event` - [in] one of the [DSL_ODE_TRIGGER_LIMIT_EVENT](constants) constants.
* `limit` - [in] the current limit value assigned to the ODE Trigger.
* `client_data` - [in] opque point to client user data provided by the client on callback registration.

<br>

---

## Constructors
### *dsl_ode_trigger_always_new*
```C++
DslReturnType dsl_ode_trigger_always_new(const wchar_t* name, const wchar_t* source, uint when);
```

The constructor creates an Always trigger that triggers an ODE occurrence on every new frame. Note, this is a No-Limit trigger, and setting a Class ID filer will have no effect.  Although always triggered, the client selects when to Trigger an ODE occurrence for each frame; before (pre) or after (post) processing of all Object metadata by all other Triggers. As with all Triggers, Always Triggers can be enabled and disabled at any time by calling [dsl_ode_trigger_enabled_set](#dsl_ode_trigger_enabled_set)

Always triggers are helpful for adding [Display Types](/dsoc/api-display-types.md) -- text, lines, rectangles, etc. -- to each frame for one or all sources.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable the filter.
* `when` - [in] either DSL_PRE_CHECK_FOR_OCCURRENCES or DSL_POST_CHECK_FOR_OCCURRENCES.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this operation can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_always_new('my-always-trigger', DSL_ODE_ANY_SOURCE, DSL_PRE_CHECK_FOR_OCCURRENCES)
```

<br>

### *dsl_ode_trigger_absence_new*
```C++
DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);
```

The constructor creates an Absence trigger that checks for the absence of Objects within a frame and generates an ODE occurrence if no objects occur.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this operation can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_absence_new('my-absence-trigger', DSL_ODE_ANY_SOURCE,
    PGIE_PERSON_CLASS_ID, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_custom_new*
```C++
DslReturnType dsl_ode_trigger_custom_new(const wchar_t* name, const wchar_t* source,
    uint class_id, uint limit, dsl_ode_check_for_occurrence_cb client_checker, void* client_data);
```

The constructor creates a Uniquely named Custom Trigger that checks for the occurrence of Objects within a frame that meets the Triggers criteria and calls a Callback function that allows the client to customize the Trigger. The Callback function is called with the buffer


**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_custom_new('my-custom-trigger',
        DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, my_check_for_occurrence_cb, my_client_data)
```

<br>

### *dsl_ode_trigger_occurrence_new*
```C++
DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);
```

The constructor creates an Occurrence trigger that checks for the occurrence of Objects within a frame and generates an ODE occurrence invoking all ODE Action for **each** object detected that passes the triggers (optional) criteria.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_occurrence_new('my-occurrence-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_instance_new*
```C++
DslReturnType dsl_ode_trigger_instance_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);
```

The constructor creates an Instance trigger that checks for new instances of Objects based on the `object_id` provided by the [Object Tracker](/docs/api-tracker.md) over consecutive frames. Each new `object_id` detected generates an ODE occurrence invoking all ODE Actions. 

**Import** the Instance Trigger's default behavior of generating a single ODE occurrence for each object can be modified with the [dsl_ode_trigger_instance_count_settings_set](#dsl_ode_trigger_instance_count_settings_set) service.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_instance_new('my-instance-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_summation_new*
```C++
DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);    
```
This constructor creates a uniquely named Summation trigger that counts the number Objects within a frame that pass the trigger's criteria. The Trigger generates an ODE occurrence invoking all ODE Actions once **per-frame** until the Trigger limit is reached.

Note: Adding Actions to a Summation Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_summation_new('my-summation-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_distance_new*
```C++
DslReturnType dsl_ode_trigger_distance_new(const wchar_t* name, const wchar_t* source,
    uint class_id_a, uint class_id_b, uint limit, uint minimum, uint maximum,
    uint test_point, uint test_method);
```

This constructor creates a uniquely named Distance Trigger that determines if Objects, that meet the Trigger's criteria, are less than a minimum and/or greater than a maximum distance from one another. If detected, the Trigger will generate an ODE occurrence invoking all ODE Actions twice, once for **each object** in the pair. Object detection is based on the specification of two class ids; A and B. All Objects of class A will be tested against all Objects of class B. `class_id_a` and `class_id_b` can both be set to the same class id and `DSL_ODE_ANY_CLASS`.  The points and method of distance measurement (fixed or ratio) are defined as well.

Intersection requires at least one pixel of overlap between a pair of object's rectangles.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id_a` - [in] inference class id filter A. Use `DSL_ODE_ANY_CLASS` to disable the filter.
* `class_id_b` - [in] inference class id filter B. Use `DSL_ODE_ANY_CLASS` to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `minimum` - [in] the minimum distance between objects in either pixels or percentage of BBox edge as specified by the test_method parameter below.
* `maximum` - [in] the maximum distance between objects in either pixels or percentage of BBox edge as specified by the test_method parameter below.
* `test_point` - [in] the point on the bounding box rectangle to use for measurement, one of DSL_BBOX_POINT.
* `test_method` - [in] method of measuring the distance between objects, one of DSL_DISTANCE_METHOD.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_distance_new('my-distance-trigger',
    source = DSL_ODE_ANY_SOURCE,
    class_id_a = PGIE_CLASS_ID_PERSON,
    class_id_b = PGIE_CLASS_ID_VEHICLE,
    minimum = 300,
    maximum = 0,
    test_point = DSL_BBOX_POINT_SOUTH,
    test_method = DSL_DISTANCE_METHOD_PERCENT_WIDTH_A)
```

<br>

### *dsl_ode_trigger_intersection_new*
```C++
DslReturnType dsl_ode_trigger_intersection_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);
```

This constructor creates a uniquely named Intersection Trigger that determines if Objects that meet the Trigger's criteria intersect, and generates an ODE occurrence invoking all ODE Actions twice, once for **each object** in the intersection pair. Object detection is based on the specification of two Class IDs; A and B. All Objects of Class A will be tested against all Objects of Class B. `class_id_a` and `class_id_b` can both be set to the same Class Id and `DSL_ODE_ANY_CLASS`.

For example: Given three objects A, B, and C. If A intersects B and B intersects C, then two unique ODE occurrences are generated. Each Action owned by the Trigger will be called for each object for every overlapping pair, i.e. a total of four times in this example.  If each of the three objects intersect with the other two, then three ODE occurrences will be triggered with each action called a total of 6 times.

Intersection requires at least one pixel of overlap between a pair of object's rectangles.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filer.
* `class_id_a` - [in] inference class id filter A. Use `DSL_ODE_ANY_CLASS` to disable the filter.
* `class_id_b` - [in] inference class id filter B. Use `DSL_ODE_ANY_CLASS` to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_intersection_new('my-intersection-trigger',
    source = DSL_ODE_ANY_SOURCE,
    class_id_a = PGIE_CLASS_ID_PERSON,
    class_id_b = PGIE_CLASS_ID_VEHICLE,
    limit = DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_count_new*
```C++
DslReturnType dsl_ode_trigger_count_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, uint minimum, uint maximum);
```

This constructor creates a uniquely named Count Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's criteria against a range of numbers. The Trigger generates an ODE occurrence invoking all Actions if the object count is within a specified range.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `minimum` - [in] the minimum count for triggering ODE occurrence, 0 = no minimum.
* `maximum` - [in] the maximum count for triggering ODE occurrence, 0 = no maximum.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_count_new('my-count-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, minimum, maximum)
```

<br>

### *dsl_ode_trigger_new_high_new*
```C++
DslReturnType dsl_ode_trigger_new_high_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit, uint preset);
```

This constructor creates a uniquely named New High Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's criteria and determines if the number has reached a new high. The Trigger generates an ODE occurrence invoking all Actions if the object count is above the highest count. The Trigger can be created with a preset high value of 0 or greater.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `preset` - [in] initial high count to start with. High count will be reset to the preset on trigger reset.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_new_high_new('my-new-high-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, preset=0)
```

<br>

### *dsl_ode_trigger_new_low_new*
```C++
DslReturnType dsl_ode_trigger_new_low_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit, uint preset);
```

This constructor creates a uniquely named New Low Trigger that checks for the occurrence of Objects within a frame that meets the Trigger's criteria and determines if the number has reached a new low. The Trigger generates an ODE occurrence invoking all Actions if the object count is above the lowest count. The Trigger can be created with a preset low value of 0 or greater.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filer.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `preset` - [in] initial low count to start with. Low count will be reset to the preset on trigger reset.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_new_low_new('my-new-low-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, preset=4)
```

<br>

### *dsl_ode_trigger_smallest_new*
```C++
DslReturnType dsl_ode_trigger_smallest_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);
```
This constructor creates a uniquely named smallest trigger that checks for the occurrence of Objects within a frame, and if at least one is found, Triggers on the Object with smallest rectangle area.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_smallest_new('my-smallest-trigger',
    DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_largest_new*
```C++
DslReturnType dsl_ode_trigger_largest_new(const wchar_t* name,
    const wchar_t* source, uint class_id, uint limit);
```
This constructor creates a uniquely named Largest trigger that checks for the occurrence of Objects within a frame, and if at least one is found, Triggers on the Object with largest rectangle area.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_largest_new('my-largest-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_cross_new*
```C++
DslReturnType dsl_ode_trigger_cross_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint min_frame_count, 
    uint max_frame_count, uint test_method);
```

This constructor creates a Cross Trigger that tracks Objects through consecutive frames and triggers on the occurrence that an object crosses one of the Trigger's Line, Multi-Line, or Polygon Areas. The Trigger maintains a vector of historical bounding-box coordinates for each object tracked by its unique tracking id. The Trigger, using the bounding box history and the Area's defined Test Point (SOUTH, WEST, etc.), generates an Object Trace - vector of x,y coordinates - to test for line cross with the Area's line(s).

**Important** Frame-to-frame fluxuations in bounding box coordinates, especially for a slow moving objects, can cause the bouding box test-point to jump back-and-forth over a given line. There are two parameters used to define "line cross hysteresis" to minimize false positives. 

1. The `line_width` defined for the Area - the test-point on the object's bounding box must fully cross the width of the line before triggering an event. This is required in both directions.
2. The `min_frame_count` - defines the minimum consecutive number of frames the object must be detected on the initial side of the line, and the minimum (non-consecutive) number of frames the object must be detected on the new side.

**Note** The Object Trace can be added as meta-data for a downstream [On-Screen Display (OSD)](/docs/api-osd.md) to display by calling [dsl_ode_trigger_track_view_settings_set](#dsl_ode_trigger_track_view_settings_set).

**Important!** The default maximum number of elements per display meta type (lines, rectangles, text, etc.) is set to sixteen (16). Setting the test method to `DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS` when displaying the Object Trace can require up to `max_trace_points` of lines per tracked object. See [dsl_pph_ode_display_meta_alloc_size_set](/docs/api-pph.md#dsl_pph_ode_display_meta_alloc_size_set) to allocate additional display meta structures per frame.

**Very Important!** Setting a minimum Inference confidence level can be required to avoid false positives from distorting the Object's historical trace. See [dsl_ode_trigger_confidence_min_set](#dsl_ode_trigger_confidence_min_set). 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `min_frame_count` - [in] setting for the minimum number of past consecutive frames on the initial side and minimum (non-sequential) frames on the new side of a line (line, multi-line or polygon area) to trigger an ODE.
* `max_trace_points` - [in] maximum number of past trace points to maintain for each tracked object.
* `test_method` - [in] either DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS or DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this operation can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_cross_new('my-cross-trigger', 'camera-1',
    PGIE_PERSON_CLASS_ID, DSL_ODE_TRIGGER_LIMIT_NONE, 5, 200,
    DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS)
```

<br>

### *dsl_ode_trigger_persistence_new*
```C++
DslReturnType dsl_ode_trigger_persistence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint minimum, uint maximum);
```

The constructor creates a Persistence trigger that checks for the persistence of Objects tracked -- based on the `object_id` provided by the [Object Tracker](/docs/api-tracker.md) over consecutive frames -- for a specified source and object class_id. Each object tracked for ">= minimum and/or <= maximum time will trigger an ODE occurrence. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `minimum` - [in] the minimum amount of time a unique object must remain detected before triggering an ODE occurrence - in units of seconds. 0 = no minimum.
* `maximum` - [in] the maximum amount of time a unique object can remain detected before triggering an ODE occurrence - in units of seconds. 0 = no maximum.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_persistence_new('my-persistence-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, minimum=15, maximum=0)
```

<br>

### *dsl_ode_trigger_earliest_new*
```C++
DslReturnType dsl_ode_trigger_earliest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);
```
This constructor creates a uniquely named Earliest Trigger that checks for the persistence of Objects tracked and will trigger on the Object with the greatest time of persistence (earliest) if at least one is found. The Trigger maintains a vector of historical bounding-box coordinates for each object tracked by its unique tracking id.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter.
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_earliest_new('my-earliest-trigger',
    DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_latest_new*
```C++
DslReturnType dsl_ode_trigger_latest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit);
```
This constructor creates a uniquely named Latest Trigger that checks for the persistence of Objects tracked and will trigger on the Object with the least time of persistence (latest) if at least one is found. The Trigger maintains a vector of historical bounding-box coordinates for each object tracked by its unique tracking id.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `source` - [in] unique name of the Source to filter on. Use NULL or DSL_ODE_ANY_SOURCE (defined as NULL) to disable filter.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_latest_new('my-latest-trigger', DSL_ODE_ANY_SOURCE,
    DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

---

## Destructors
### *dsl_ode_trigger_delete*
```C++
DslReturnType dsl_ode_trigger_delete(const wchar_t* trigger);
```
This destructor deletes a single, uniquely named ODE Trigger. The destructor will fail if the Trigger is currently `in-use` by an ODE Handler

**Parameters**
* `trigger` - [in] unique name for the ODE Trigger to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_delete('my-trigger')
```

<br>

### *dsl_ode_trigger_delete_many*
```C++
DslReturnType dsl_trigger_delete_many(const wchar_t** triggers);
```
This destructor deletes multiple uniquely named ODE Triggers. Each name is checked for existence, with the function returning on first failure. The destructor will fail if one of the Actions is currently `in-use` by one or more ODE Triggers

**Parameters**
* `trigger` - [in] a NULL terminated array of uniquely named ODE Triggers to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_delete_many(['my-trigger-a', 'my-trigger-b', 'my-trigger-c', None])
```

<br>

### *dsl_ode_trigger_delete_all*
```C++
DslReturnType dsl_ode_trigger_delete_all();
```
This destructor deletes all ODE Triggers currently in memory. The destructor will fail if any one of the Triggers is currently `in-use` by an ODE Handler.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_delete_all()
```

<br>

## Methods
### *dsl_ode_trigger_count_range_get*
```c++
DslReturnType dsl_ode_trigger_count_range_get(const wchar_t* name,
    uint* minimum, uint* maximum);
```

This service gets the current minimum and maximum count settings in use by the named ODE Count Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Count Trigger to query.
* `minimum` - [in] the current minimum count for triggering ODE occurrence, 0 = no minimum.
* `maximum` - [in] the current minimum count for triggering ODE occurrence, 0 = no maximum.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, minimum, maximum = dsl_ode_trigger_count_range_get('my-trigger')
```

<br>

### *dsl_ode_trigger_count_range_set*
```c++
DslReturnType dsl_ode_trigger_count_range_set(const wchar_t* name,
    uint minimum, uint maximum);
```

This service sets the current minimum and maximum count settings to use for the named ODE Count Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Count Trigger to update.
* `minimum` - [in] the new minimum count for triggering ODE occurrence, 0 = no minimum.
* `maximum` - [in] the new maximum count for triggering ODE occurrence, 0 = no maximum.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_count_range_set('my-trigger', 10, 0)
```

<br>

### *dsl_ode_trigger_distance_range_get*
```c++
DslReturnType dsl_ode_trigger_distance_range_get(const wchar_t* name,
    uint* minimum, uint* maximum);
```

This service gets the current minimum and maximum distance settings in use by the named ODE Distance Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `minimum` - [out] the current minimum distance between objects in either pixels or percentage of bounding box point as specified by the test_method parameter.
* `maximum` - [out] the current maximum distance between objects in either pixels or percentage of bounding box point as specified by the test_method parameter.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, minimum, maximum = dsl_ode_trigger_distance_range_get('my-trigger')
```

<br>

### *dsl_ode_trigger_distance_range_set*
```c++
DslReturnType dsl_ode_trigger_distance_range_set(const wchar_t* name,
    uint minimum, uint maximum);
```

This service sets the current minimum and maximum distance settings to use for the named ODE Distance Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `minimum` - [in] the distance between objects in either pixels or percentage of bounding box point as specified by the test_method parameter.
* `maximum` - [in] the maximum distance between objects in either pixels or percentage of bounding box point as specified by the test_method parameter.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_distance_range_set('my-trigger', 100, 300)
```

<br>

### *dsl_ode_trigger_distance_test_params_get*
```c++
DslReturnType dsl_ode_trigger_distance_test_params_get(const wchar_t* name,
    uint* test_point, uint* test_method);
```

This service gets the current Test Point and Test Method parameters for the named ODE Distance Trigger

**Parameters**
* `name` - [in] unique name of the ODE Distance Trigger to update.
* `test_point` - [out] the point on the bounding box rectangle to use for measurement, one of DSL_BBOX_POINT
* `test_method` - [out] method of measuring the distance between objects, one of DSL_DISTANCE_METHOD

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, test_point, test_method = dsl_ode_trigger_distance_test_params_get('my-trigger')
```

<br>

### *dsl_ode_trigger_distance_test_params_set*
```c++
DslReturnType dsl_ode_trigger_distance_test_params_set(const wchar_t* name,
    uint test_point, uint test_method);
```

This service sets the current Test Point and Test Method parameters for the named ODE Distance Trigger to use.

**Parameters**
* `name` - [in] unique name of the ODE Distance Trigger to update.
* `test_point` - [in] the point on the bounding box rectangle to use for measurement, one of DSL_BBOX_POINT
* `test_method` - [in] method of measuring the distance between objects, one of DSL_DISTANCE_METHOD

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_distance_test_params_get('my-trigger',
    test_point = DSL_BBOX_POINT_SOUTH,
    test_method = DSL_DISTANCE_METHOD_PERCENT_WIDTH_A)
```

<br>

<br>

### *dsl_ode_trigger_cross_test_settings_get*
```c++
DslReturnType dsl_ode_trigger_cross_test_settings_get(const wchar_t* name, 
    uint* min_frame_count, uint* max_trace_points, uint* test_method);
```

This service gets the current test settings for the named Cross Trigger

**Parameters**
* `name` - [in] unique name of the ODE Cross Trigger to query.
* `min_frame_count` - [out] current setting for the minimum number of past consecutive frames on the initial side and minimum (non-sequential) frames on the new side of a line (line, multi-line or polygon area) to trigger an ODE.
* `max_trace_points` - [out] current setting for the maximum number of past trace points to maintain for each tracked object.
* `test_method` - [out] either DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS or DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, min_frame_count, max_trace_points, test_method =
    dsl_ode_trigger_cross_test_settings_get('my-trigger')
```

<br>

### *dsl_ode_trigger_cross_test_settings_set*
```c++
DslReturnType dsl_ode_trigger_cross_test_settings_set(const wchar_t* name, 
    uint* min_frame_count, uint* max_trace_points, uint* test_method);
```

This service sets the test settings for the named Cross Trigger to use. 

**Parameters**
* `name` - [in] unique name of the ODE Cross Trigger to update.
* `min_frame_count` - [in] new setting for the minimum number of past consecutive frames on the initial side and minimum (non-sequential) frames on the new side of a line (line, multi-line or polygon area) to trigger an ODE.
* `max_trace_points` - [in] new setting for the maximum number of past trace points to maintain for each tracked object.
* `test_method` - [in] either DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS or DSL_OBJECT_TRACE_TEST_METHOD_ALL_POINTS.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_cross_test_settings_set('my-trigger',
    min_frame_count, max_trace_points, test_method)
```

<br>

### *dsl_ode_trigger_cross_view_settings_get*
```c++
DslReturnType dsl_ode_trigger_cross_view_settings_get(const wchar_t* name, 
    boolean* enabled, const wchar_t** color, uint* line_width);
```

This service gets the current view settings for the named Cross Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Tracking Trigger to query.
* `enabled` - [out] true if object trace display is enabled, false otherwise. Default = disabled.
* `color` - [out] name of the color to use for object trace display, default = no-color.
* `line_width` - [out] width of the object trace if display is enabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled, color, line_width =
    dsl_ode_trigger_cross_view_settings_get('my-trigger')
```

<br>

### *dsl_ode_trigger_cross_view_settings_set*
```c++
DslReturnType dsl_ode_trigger_cross_view_settings_set(const wchar_t* name, 
    boolean enabled, const wchar_t* color, uint line_width);
```

This service Sets the trace view settings for the named Cross Trigger to use.

**Note:** The tracked object's bounding box will be updated with `color` to match the object's trace if `enabled` is set to true.

**Parameters**
* `name` - [in] unique name of the ODE Tracking Cross to update.
* `enabled` - [in] set to true to enable object trace display, false otherwise.
* `color` - [in] name of the color to use for object trace display.
* `line_width` - [in] width of the object trace if display is enabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_cross_view_settings_set('my-trigger',
    True, 'my-color', 6)
```

<br>

### *dsl_ode_trigger_instance_count_settings_get*
```c++
DslReturnType dsl_ode_trigger_instance_count_settings_get(const wchar_t* name,
    uint* instance_count, uint* suppression_count);
```

This service gets the current instance and suppression count settings for the named ODE Instance Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Instance Trigger to query.
* `instance_count` - [out] the number of consecutive instances to trigger an ODE occurrence. Default = 1.
* `suppression_count` - [out] the number of consecutive instances to suppress ODE occurrence once the instance_count has been reached. Default = 0 (suppress indefinitely). If set, the cycle of generated instances followed by suppressed instances will repeat as long as the same object is detected through successive frames.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, instance_count, suppression_count = dsl_ode_trigger_instance_count_settings_get('my-trigger')
```

<br>

### *dsl_ode_trigger_instance_count_settings_set*
```c++
DslReturnType dsl_ode_trigger_instance_count_settings_set(const wchar_t* name,
    uint instance_count, uint suppression_count);
```

This service sets the instance and suppression count settings for the named ODE Instance Trigger to use.

**Parameters**
* `name` - [in] unique name of the ODE Instance Trigger to update.
* `instance_count` - [in] the number of consecutive instances to trigger an ODE occurrence. Default = 1.
* `suppression_count` - [in] the number of consecutive instances to suppress ODE occurrence once the instance_count has been reached. Default = 0 (suppress 
indefinitely).  If set, the cycle of generated instances followed by suppressed instances will repeat as long as the same object is detected through successive frames.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_instance_count_settings_set('my-trigger', 2, 900)
```

### *dsl_ode_trigger_persistence_range_get*
```c++
DslReturnType dsl_ode_trigger_persistence_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum);
```

This service gets the current minimum and maximum time settings in use by the named ODE Persistence Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Persistence Trigger to query.
* `minimum` - [out] the minimum amount of time a unique object must remain detected before triggering an ODE occurrence - in units of seconds. 0 = no minimum
* `maximum` - [out] the maximum amount of time a unique object can remain detected before triggering an ODE occurrence - in units of seconds. 0 = no maximum

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, minimum, maximum = dsl_ode_trigger_persistence_range_get('my-trigger')
```

<br>

### *dsl_ode_trigger_persistence_range_set*
```c++
DslReturnType dsl_ode_trigger_persistence_range_set(const wchar_t* name, 
    uint minimum, uint maximum);
```

This service sets the current minimum and maximum time settings to use for the named ODE Persistence Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Tracking Persistence Trigger to update.
* `minimum` - [in] the minimum amount of time a unique object must remain detected before triggering an ODE occurrence - in units of seconds. 0 = no minimum
* `maximum` - [in] the maximum amount of time a unique object can remain detected before triggering an ODE occurrence - in units of seconds. 0 = no maximum

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_persistence_range_set('my-trigger', 100, 300)
```


### *dsl_ode_trigger_reset*
```c++
DslReturnType dsl_ode_trigger_reset(const wchar_t* name);
```

This service resets a named ODE Trigger, setting its triggered count to 0.  This affects Triggers with fixed limits, whether they have reached their limit or not.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to reset.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_reset('my-trigger')
```

<br>

### *dsl_ode_trigger_reset_timeout_get*
```c++
DslReturnType dsl_ode_trigger_reset_timeout_get(const wchar_t* name, uint *timeout);
```

This service gets the current auto-reset timeout setting for the named ODE Trigger. If set, upon reaching its limit, the Trigger will start a timer to auto-reset on expiration. A timeout of 0 disables auto-reset -- default setting for all triggers.  

**Parameters**
* `name` - [in] unique name of the ODE Trigger to reset.
* `timeout` - [out] current timeout value in units of seconds.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, timeout = dsl_ode_trigger_reset_timeout_get('my-trigger')
```

<br>

### *dsl_ode_trigger_reset_timeout_set*
```c++
DslReturnType dsl_ode_trigger_reset_timeout_set(const wchar_t* name, uint timeout);
```

This service sets the current auto-reset timeout setting for the named ODE Trigger. If set, upon reaching its limit, the Trigger will start a timer to auto-reset on expiration. A timeout of 0 disables auto-reset, the default setting for all triggers.  

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `timeout` - [in] new timeout value in units of seconds.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_reset_timeout_set('my-trigger', 3)
```

<br>

### *dsl_ode_trigger_enabled_get*
```c++
DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled);
```

This service returns the current enabled setting for the named ODE Trigger. Triggers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `enabled` - [out] true if the ODE Trigger is currently enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_ode_trigger_enabled_get('my-trigger')
```

<br>

### *dsl_ode_trigger_enabled_set*
```C++
DslReturnType dsl_ode_trigger_enabled_set(const wchar_t* name, boolean enabled);
```

This service sets the enabled setting for the named ODE Trigger. Triggers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `enabled` - [in] set to true to enable the ODE Trigger, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_enabled_set('my-trigger', False)
```

<br>

### *dsl_ode_trigger_enabled_state_change_listener_add*
```C++
DslReturnType dsl_ode_trigger_enabled_state_change_listener_add(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener, void* client_data);
```
This service adds a callback function of type [dsl_ode_enabled_state_change_listener_cb](#dsl_ode_enabled_state_change_listener_cb) to an ODE Trigger identified by its unique name. The function will be called on every change of the Trigger's enabled state. Multiple callback functions can be registered with one Trigger, and one callback function can be registered with multiple Triggers.

**Parameters**
* `name` - [in] unique name of the Trigger to update.
* `listener` - [in] the enabled-state-change-listener callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the listener is called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful addition. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def enabled_state_change_listener(enabled, client_data):
    print('enabled = ', enabled)
   
retval = dsl_ode_trigger_enabled_state_change_listener_add('my-occurrence-trigger',
    enabled_state_change_listener, None)
```

<br>

### *dsl_ode_trigger_enabled_state_change_listener_remove*
```C++
DslReturnType dsl_ode_trigger_enabled_state_change_listener_remove(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener);
```
This service removes a callback function of type [dsl_ode_enabled_state_change_listener_cb](#dsl_ode_enabled_state_change_listener_cb) from an ODE Trigger identified by its unique name.

**Parameters**
* `name` - [in] unique name of the Trigger to update.
* `listener` - [in] the enabled-state-change-listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_limit_event_listener_remove('my-occurrence-trigger',
    enabled_state_change_listener)
```

<br>

### *dsl_ode_trigger_source_get*
```c++
DslReturnType dsl_ode_trigger_source_get(const wchar_t* name, const wchar_t** source);
```

This service returns the current source_id filter setting for the named ODE Trigger. A value of `DSL_ODE_ANY_SOURCE` (defined as NULL) indicates that the filter is disable and Source name will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `source` - [out] current source name filter for the ODE Trigger to filter on.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, source_name = dsl_ode_trigger_source_get('my-trigger')
```

<br>

### *dsl_ode_trigger_source_set*
```c++
DslReturnType dsl_ode_trigger_source_set(const wchar_t* name, const wchar_t* source);
```

This service sets the current source filter setting for the named ODE Trigger. A value of `DSL_ODE_ANY_SOURCE` (or NULL) disables the filter and the Source name will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `source` - [in] new Source name for the ODE Trigger to filter on, or `DSL_ODE_ANY_SOURCE` to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_source_id_set('my-trigger', 'my-source-1')
```

<br>

### *dsl_ode_trigger_class_id_get*
```c++
DslReturnType dsl_ode_trigger_class_id_get(const wchar_t* name, uint* class_id);
```

This service returns the current class_id filter setting for the named ODE Trigger. A value of `DSL_ODE_ANY_CLASS` indicates that the filter is disable and the GIE class Id will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `class_id` - [out] current class Id filter for the ODE Trigger to filter on.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, class_id = dsl_ode_trigger_class_id_get('my-trigger')
```

<br>

### *dsl_ode_trigger_class_id_set*
```c++
DslReturnType dsl_ode_trigger_class_id_set(const wchar_t* name, uint class_id);
```

This service sets the current class_id filter setting for the named ODE Trigger. A value of `DSL_ODE_ANY_CLASS` disables the filter and the GIE class Id will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `class_id` - [in] new class Id filter for the ODE Trigger to filter on, or `DSL_ODE_ANY_CLASS` to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_class_id_set('my-trigger', DSL_ODE_ANY_CLASS)
```

<br>

### *dsl_ode_trigger_class_id_ab_get*
```c++
DslReturnType dsl_ode_trigger_class_id_ab_get(const wchar_t* name,
    uint* class_id_a, uint* class_id_b);
```

This service returns the current class_id_a and class_id_b filter settings for the named **AB Type** ODE Trigger. A value of `DSL_ODE_ANY_CLASS` indicates that the class filter is disabled.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `class_id_a` - [out] current class Id A filter for the ODE Trigger to filter on.
* `class_id_a` - [out] current class Id B filter for the ODE Trigger to filter on.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, class_id_a, class_id_b = dsl_ode_trigger_class_id_ab_get('my-trigger')
```

<br>

### *dsl_ode_trigger_class_id_ab_set*
```c++
DslReturnType dsl_ode_trigger_class_id_ab_set(const wchar_t* name,
    uint class_id_a, uint class_id_b);;
```

This service sets the currentclass_id_a and class_id_b filter settings for a named **AB Type** ODE Trigger. Both can be set to the same Class Id or `DSL_ODE_ANY_CLASS`

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `class_id_a` - [in] new class Id A filter for the ODE Trigger to filter on.
* `class_id_b` - [in] new class Id B filter for the ODE Trigger to filter on.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_class_id_ab_set('my-trigger',
    class_id_a = DSL_ODE_ANY_CLASS, class_id_b = DSL_ODE_ANY_CLASS)
```

<br>

### *dsl_ode_trigger_limit_event_get*
```c++
DslReturnType dsl_ode_trigger_limit_event_get(const wchar_t* name, uint* limit);
```

This service returns the current Trigger event limit setting for the named ODE Trigger. A value of zero indicates NO Limit

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `limit` - [out] current limit setting for the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure..

**Python Example**
```Python
retval, limit = dsl_ode_trigger_limit_event_get('my-trigger')
```

<br>

### *dsl_ode_trigger_limit_event_set*
```c++
DslReturnType dsl_ode_trigger_limit_event_set(const wchar_t* name, uint limit);
```

This service sets the event limit setting for the named ODE Trigger. Setting the limit to zero disables the limit check, i.e. no limit.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `limit` - [in] new limit for the ODE Trigger to filter on, 0 to indicate no limit.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_limit_event_set('my-trigger', 0)
```

<br>

### *dsl_ode_trigger_limit_frame_get*
```c++
DslReturnType dsl_ode_trigger_limit_frame_get(const wchar_t* name, uint* limit);
```

This service returns the current Trigger frame limit setting for the named ODE Trigger. A value of zero indicates NO Limit.

Each Trigger counts the frames that occur - starting with the frame with the first ODE occurrence - and will stop checking for ODE occurrence once the frame limit is reached. Resetting the Trigger will reset the frame count.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `limit` - [out] current limit setting for the ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure..

**Python Example**
```Python
retval, limit = dsl_ode_trigger_limit_frame_get('my-trigger')
```

<br>

### *dsl_ode_trigger_limit_frame_set*
```c++
DslReturnType dsl_ode_trigger_limit_frame_set(const wchar_t* name, uint limit);
```

This service sets the frame limit setting for the named ODE Trigger. Setting the limit to zero disables the limit check, i.e. no limit.

Each Trigger counts the frames that occur - starting with the frame with the first ODE occurrence - and will stop checking for ODE occurrence once the frame limit is reached. Resetting the Trigger will reset the frame count.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `limit` - [in] new limit for the ODE Trigger to filter on, 0 to indicate no limit.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_limit_frame_set('my-trigger', 0)
```

<br>

### *dsl_ode_trigger_limit_state_change_listener_add*
```C++
DslReturnType dsl_ode_trigger_limit_state_change_listener_add(const wchar_t* name,
    dsl_ode_trigger_limit_event_listener_cb listener, void* client_data);
```
This service adds a callback function of type [dsl_ode_trigger_limit_event_listener_cb](#dsl_ode_trigger_limit_event_listener_cb) to an ODE Trigger identified by its unique name. The function will be called on every limit state-change -- `LIMIT_REACHED`, `LIMIT_CHANGED`, and `COUNTS_RESET` -- that occurs. Listeners will be notified on both [event-limit](#dsl_ode_trigger_limit_event_set) and [frame limit](#dsl_ode_trigger_limit_frame_set) state changes. Multiple callback functions can be registered with one Trigger, and one callback function can be registered with multiple Triggers.

**Parameters**
* `name` - [in] unique name of the Trigger to update.
* `listener` - [in] trigger limit event listener callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the listener is called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful addition. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def limit_event_listener(event, client_data):
    print('event = ', event)
   
retval = dsl_ode_trigger_limit_state_change_listener_add('my-occurrence-trigger',
    limit_event_listener, None)
```

<br>

### *dsl_ode_trigger_limit_state_change_listener_remove*
```C++
DslReturnType dsl_ode_trigger_limit_state_change_listener_remove(const wchar_t* name,
    dsl_ode_trigger_limit_event_listener_cb listener, void* client_data);
```
This service removes a callback function of type [dsl_ode_trigger_limit_event_listener_cb](#dsl_ode_trigger_limit_event_listener_cb) from an ODE Trigger identified by its unique name.

**Parameters**
* `name` - [in] unique name of the Trigger to update.
* `listener` - [in] trigger limit event listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_limit_state_change_listener_remove('my-occurrence-trigger',
    limit_event_listener)
```

<br>

### *dsl_ode_trigger_infer_confidence_min_get*
```c++
DslReturnType dsl_ode_trigger_infer_confidence_min_get(const wchar_t* name, 
    double* min_confidence);
```

This service returns the current minimum Inference confidence criteria for the named ODE Trigger. A value of 0 (default) indicates that the criteria is disable and the detected object's minimum Inference confidence value will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_confidence` - [out] current minimum confidence value between 0.0 and 1.0 for the ODE Trigger to filter on, 0 indicates disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, min_confidence = dsl_ode_trigger_infer_confidence_min_get('my-trigger')
```

<br>

### *dsl_ode_trigger_infer_confidence_min_set*
```c++
DslReturnType dsl_ode_trigger_infer_confidence_min_set(const wchar_t* name, 
    double min_confidence);
```

This service sets the minimum Inference confidence criteria for the named ODE Trigger. A value of 0 disables the filter and the minimum Inference confidence level will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_confidence` - [in] new minimum confidence value as criteria for the ODE Trigger to filter on, or 0 to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_infer_confidence_min_set('my-trigger', min_confidence)
```

<br>

### *dsl_ode_trigger_infer_confidence_max_get*
```c++
DslReturnType dsl_ode_trigger_infer_confidence_max_get(const wchar_t* name, 
    double* max_confidence);
```

This service returns the current maximum Inference confidence criteria for the named ODE Trigger. A value of 0 (default) indicates that the criteria is disable and the detected object's maximum Inference confidence value will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `max_confidence` - [out] current maximum confidence value between 0.0 and 1.0 for the ODE Trigger to filter on, 0 indicates disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, max_confidence = dsl_ode_trigger_infer_confidence_max_get('my-trigger')
```

<br>

### *dsl_ode_trigger_infer_confidence_max_set*
```c++
DslReturnType dsl_ode_trigger_infer_confidence_max_set(const wchar_t* name, 
    double min_confidence);
```

This service sets the maximum Inference confidence criteria for the named ODE Trigger. A value of 0 disables the filter and the maximum Inference confidence level will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `max_confidence` - [in] new maximum confidence value as criteria for the ODE Trigger to filter on, or 0 to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_infer_confidence_max_set('my-trigger', 0.30)
```

<br>

### *dsl_ode_trigger_tracker_confidence_min_get*
```c++
DslReturnType dsl_ode_trigger_tracker_confidence_min_get(const wchar_t* name, double* min_confidence);
```

This service returns the current minimum Tracker confidence criteria for the named ODE Trigger. A value of 0 (default) indicates that the criteria is disable and the detected object's mimimum Tracker confidence value will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_confidence` - [out] current minimum confidence value between 0.0 and 1.0 for the ODE Trigger to filter on, 0 indicates disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, min_confidence = dsl_ode_trigger_tracker_confidence_min_get('my-trigger')
```

<br>

### *dsl_ode_trigger_tracker_confidence_min_set*
```c++
DslReturnType dsl_ode_trigger_tracker_confidence_min_set(const wchar_t* name, double min_confidence);
```

This service sets the minimum Tracker confidence criteria for the named ODE Trigger. A value of 0 disables the filter and the minimum Tracker confidence level will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_confidence` - [in] new minimum confidence value as criteria for the ODE Trigger to filter on, or 0 to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_tracker_confidence_min_set('my-trigger', min_confidence)
```

<br>

### *dsl_ode_trigger_tracker_confidence_max_get*
```c++
DslReturnType dsl_ode_trigger_tracker_confidence_max_get(const wchar_t* name, double* min_confidence);
```

This service returns the current maximum Tracker confidence criteria for the named ODE Trigger. A value of 0 (default) indicates that the criteria is disable and the detected object's maximum Tracker confidence value will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `max_confidence` - [out] current maximum confidence value between 0.0 and 1.0 for the ODE Trigger to filter on, 0 indicates disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, max_confidence = dsl_ode_trigger_tracker_confidence_max_get('my-trigger')
```

<br>

### *dsl_ode_trigger_tracker_confidence_max_set*
```c++
DslReturnType dsl_ode_trigger_tracker_confidence_max_set(const wchar_t* name, double min_confidence);
```

This service sets the maximum Tracker confidence criteria for the named ODE Trigger. A value of 0 disables the filter and the maximum Tracker confidence level will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `max_confidence` - [in] new maximum confidence value as criteria for the ODE Trigger to filter on, or 0 to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_tracker_confidence_max _set('my-trigger', min_confidence)
```

<br>

### *dsl_ode_trigger_dimensions_min_get*
```c++
DslReturnType dsl_ode_trigger_dimensions_min_get(const wchar_t* name, uint* min_width, uint* min_height);
```

This service returns the current minimum dimensions for the named ODE Trigger. A value of 0 (default) indicates that the Object's rectangle width and/or height will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_width` - [out] current minimum width value for the ODE Trigger to filter on, 0 indicates disabled.
* `min_hight` - [out] current minimum height value for the ODE Trigger to filter on, 0 indicates disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, min_width, min_height = dsl_ode_trigger_dimensions_min_get('my-trigger')
```

<br>

### *dsl_ode_trigger_dimensions_min_set*
```c++
DslReturnType dsl_ode_trigger_dimensions_min_set(const wchar_t* name, uint min_width, uint min_height);
```

This service sets the current minimum dimensions for the named ODE Trigger. A value of 0 (default) indicates that the Object's rectangle width and/or height will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_width` - [out] minimum width value for the ODE Trigger to filter on, 0 indicates disabled.
* `min_hight` - [out] minimum height value for the ODE Trigger to filter on, 0 indicates disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_dimensions_min_get('my-trigger', min_width, min_height)
```

<br>

### *dsl_ode_trigger_dimensions_max_get*
```c++
DslReturnType dsl_ode_trigger_dimensions_max_get(const wchar_t* name, uint* max_width, uint* max_height);
```

This service returns the current maximum dimensions for the named ODE Trigger. A value of 0 (default) indicates that the Object's rectangle width and/or height will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `max_width` - [out] current maximum width value for the ODE Trigger to filter on, 0 indicates disabled.
* `max_hight` - [out] current maximum height value for the ODE Trigger to filter on, 0 indicates disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, max_width, max_height = dsl_ode_trigger_dimensions_max_get('my-trigger')
```

<br>

### *dsl_ode_trigger_dimensions_max_set*
```c++
DslReturnType dsl_ode_trigger_dimensions_max_set(const wchar_t* name, uint max_width, uint max_height);
```

This service sets the current maximum dimensions for the named ODE Trigger. A value of 0 (default) indicates that the Object's rectangle width and/or height will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `max_width` - [out] maximum width value for the ODE Trigger to filter on, 0 indicates disabled.
* `max_hight` - [out] maximum height value for the ODE Trigger to filter on, 0 indicates disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_dimensions_max_get('my-trigger', max_width, max_height)
```

<br>

### *dsl_ode_trigger_infer_done_only_get*
```c++
DslReturnType dsl_ode_trigger_infer_done_only_get(const wchar_t* name, boolean* infer_done_only)
```

This service returns the current "infer-done-only" criteria for the named ODE Trigger. A value of False (default) indicates that the Object's "infer-done" meta flag will not be used as criteria for ODE occurrence. If set to True, only those frames with the "infer-done" flag = true can trigger an ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `infer_done_only` - [out] if set to true, then the "inference-done" filer is enabled, false indicates disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, infer_done_only = dsl_ode_trigger_infer_done_only_get('my-trigger')
```

<br>

### *dsl_ode_trigger_infer_done_only_set*
```c++
DslReturnType dsl_ode_trigger_infer_done_only_get(const wchar_t* name, boolean* infer_done_only)
```

This service sets the "inference-done-only" criteria for the named ODE Trigger. A value of False (default) indicates that the Object's "inference-done" flag will not be used as criteria for ODE occurrence. If set to True, only those frames with the "inference-done" flag = true can trigger an ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `infer_done_only` - [in] set to true ot enable the "infer-done-only" criteria, false (default) to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_infer_done_only_get('my-trigger', True)
```

<br>

### *dsl_ode_trigger_interval_get*
```c++
DslReturnType dsl_ode_trigger_interval_get(const wchar_t* name, uint* interval);
```
This service gets the current frame processing interval setting for the named ODE Trigger, If set to `n`, the Trigger will only process every `nth` frame while skipping the others.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `limit` - [out] current interval setting for the ODE Trigger. Default = 0.  

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, interval = dsl_ode_trigger_interval_get('my-trigger')
```

<br>

### *dsl_ode_trigger_interval_set*
```c++
DslReturnType dsl_ode_trigger_interval_set(const wchar_t* name, uint interval);
```

This service sets the current frame processing interval setting for the named ODE Trigger, If set to `n`, the Trigger will only process every `nth` frame while skipping the others.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `interval` - [in] new interval for the ODE Trigger to process frames on. Set to 0 (default) or 1 to process all frames.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_interval_set('my-trigger', 5)
```

<br>

### *dsl_ode_trigger_action_add*
```c++
DslReturnType dsl_ode_trigger_action_add(const wchar_t* name, const wchar_t* action);
```

This service adds a named ODE Action to a named ODE Trigger. The same Action can be added to multiple Triggers.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `action` - [in] unique name of the ODE Action to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_action_add('my-trigger', 'my-action')
```

<br>

### *dsl_ode_trigger_action_add_many*
```c++
DslReturnType dsl_ode_trigger_action_add_many(const wchar_t* name, const wchar_t** actions);
```

This service adds a Null terminated list of named ODE Actions to a named ODE Trigger. The same Actions can be added to multiple Triggers.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `actions` - [in] a Null terminated list of unique names of the ODE Actions to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_action_add_many('my-trigger', ['my-action-a', 'my-action-b', 'my-action-c', None])
```

<br>

### *dsl_ode_trigger_action_remove*
```c++
DslReturnType dsl_ode_trigger_action_remove(const wchar_t* name, const wchar_t* action);
```

This service removes a named ODE Action from a named ODE Trigger. The services will fail if the Action is not currently in-use by the named Trigger

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `action` - [in] unique name of the ODE Action to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_action_remove('my-trigger', 'my-action')
```

<br>

### *dsl_ode_trigger_action_remove_many*
```c++
DslReturnType dsl_ode_trigger_action_remove_many(const wchar_t* name, const wchar_t** actions);
```

This service removes a Null terminated list of named ODE Actions to a named ODE Trigger. The service will fail if any of the named Actions are not currently in-use by the named Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `actions` - [in] a Null terminated list of unique names of the ODE Actions to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_action_remove_many('my-trigger', ['my-action-a', 'my-action-b', 'my-action-c', None])
```

<br>

### *dsl_ode_trigger_action_remove_all*
```c++
DslReturnType dsl_ode_trigger_action_remove_all(const wchar_t* name);
```

This service removes all ODE Actions from a named ODE Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_action_remove_all('my-trigger')
```

<br>

### *dsl_ode_trigger_area_add*
```c++
DslReturnType dsl_ode_trigger_area_add(const wchar_t* name, const wchar_t* action);
```

This service adds a named ODE Area to a named ODE Trigger. The same Area can be added to multiple Triggers.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `area` - [in] unique name of the ODE Area to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_area_add('my-trigger', 'my-area')
```

<br>

### *dsl_ode_trigger_area_add_many*
```c++
DslReturnType dsl_ode_trigger_area_add_many(const wchar_t* name, const wchar_t** areas);
```

This service adds a Null terminated list of named ODE Areas to a named ODE Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `areas` - [in] a Null terminated list of unique names of the ODE Areas to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_area_add_many('my-trigger', ['my-area-a', 'my-area-b', 'my-area-c', None])
```

<br>

### *dsl_ode_trigger_area_remove*
```c++
DslReturnType dsl_ode_trigger_area_remove(const wchar_t* name, const wchar_t* area);
```

This service removes a named ODE Area from a named ODE Trigger. The services will fail if the Area is not currently in-use by the named Trigger

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `area` - [in] unique name of the ODE Area to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_area_remove('my-trigger', 'my-area')
```

<br>

### *dsl_ode_trigger_area_remove_many*
```c++
DslReturnType dsl_ode_trigger_area_remove_many(const wchar_t* name, const wchar_t** areas);
```

This service removes a Null terminated list of named ODE Areas to a named ODE Trigger. The service will fail if one of the named Areas is not currently in-use by the named Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `areas` - [in] a Null terminated list of unique names of the ODE Areas to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_area_remove_many('my-trigger', ['my-area-a', 'my-area-b', 'my-area-c', None])
```

<br>

### *dsl_ode_trigger_area_remove_all*
```c++
DslReturnType dsl_ode_trigger_area_remove_all(const wchar_t* name);
```

This service removes all ODE Areas from a named ODE Trigger.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_area_remove_all('my-trigger')
```

<br>

### *dsl_ode_trigger_accumulator_add*
```c++
DslReturnType dsl_ode_trigger_accumulator_add(const wchar_t* name, 
    const wchar_t* accumulator);
```

This service adds a named ODE Accumulator to a named ODE Trigger. The Trigger can have at most one Accumulator. The same Accumulator can be added to multiple Triggers.  See the [ODE Accumulator API Reference](/docs/api-ode-accumulator.md) for additional information.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `accumulator` - [in] unique name of the ODE Accumulator to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_accumulator_add('my-trigger', 'my-accumulator')
```

<br>

### *dsl_ode_trigger_accumulator_remove*
```c++
DslReturnType dsl_ode_trigger_accumulator_remove(const wchar_t* name);
```

This service removes an ODE Accumulator from a named ODE Trigger. The services will fail if an Accumulator is not currently in-use by the named Trigger. See the [ODE Accumulator API Reference](/docs/api-ode-accumulator.md) for additional information.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_accumulator_remove('my-trigger')
```

<br>

### *dsl_ode_trigger_heat_mapper_add*
```c++
DslReturnType dsl_ode_trigger_heat_mapper_add(const wchar_t* name, 
    const wchar_t* heat_mapper);
```

This service adds a named ODE Heat-Mapper to a named ODE Trigger. The Trigger to Heat-Mapper relationship is one-to-one. See the [ODE Heat-Mapper API Reference](/docs/api-ode-heat-mapper.md) for additional information.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `heat_mapper` - [in] unique name of the ODE Heat-Mapper to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_heat_mapper_add('my-trigger', 'my-heat-mapper')
```

<br>

### *dsl_ode_trigger_heat_mapper_remove*
```c++
DslReturnType dsl_ode_trigger_heat_mapper_remove(const wchar_t* name);
```

This service removes an ODE Heat-Mapper from a named ODE Trigger. The services will fail if a Heat-Mapper is not currently in-use by the named Trigger. See the [ODE Heat-Mapper API Reference](/docs/api-ode-heat-mapper.md) for additional information.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_heat_mapper_remove('my-trigger')
```

<br>

### *dsl_ode_trigger_list_size*
```c++
uint dsl_ode_trigger_list_size();
```
This service returns the size of the ODE Trigger container, i.e. the number of Triggers currently in memory.

**Returns**
* The size of the ODE Trigger container.

**Python Example**
```Python
size = dsl_ode_trigger_list_size()
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
* **ODE-Trigger**
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Types](/docs/api-display-types.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)  
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
