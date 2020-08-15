## ODE Trigger API
ODE Triggers use settable criteria to parse the frame and detected-object metadata looking for occurrences of specific object-detection-events: Absence, Occurrence, Intersection, etc. Triggers, on ODE occurrence, iterate through their collection of ordered [ODE Actions](/docs/api-ode-action.md) invoking each. [ODE Areas](/docs/api-ode-area.md), rectangles with location and dimension, can be added to one or more Triggers as criteria for ODE occurrence as well.

#### Construction and Destruction
Triggers are created by calling one of the Type specific constructors defined below. Triggers are deleted by calling [dsl_ode_trigger_delete](#dsl_ode_trigger_delete), [dsl_ode_trigger_delete_many](#dsl_ode_trigger_delete_many), or [dsl_ode_trigger_delete_all](#dsl_ode_trigger_delete_all).

#### Adding and Removing Triggers
The relationship between ODE Pad Prop Handlers and ODE Triggers is one-to-many. A Trigger must be removed from a Handler before it can be used by another. Triggers are added to a handler by calling [dsl_pph_ode_trigger_add](docs/api-pph.md#dsl_pph_ode_trigger_add) and [dsl_pph_ode_trigger_add_many](docs/api-pph.md#dsl_pph_ode_trigger_add_many), and removed with [dsl_pph_ode_trigger_remove](docs/api-pph.md#dsl_pph_ode_trigger_remove), [dsl_pph_ode_trigger_remove_many](docs/api-pph.md#dsl_pph_ode_trigger_remove_many), and [dsl_pph_ode_trigger_remove_all](docs/api-pph.md#dsl_pph_ode_trigger_remove_all).

#### Adding and Removing Actions
Multiple ODE Actions can be added to an ODE Trigger and the same ODE Action can be added to multiple Triggers.  ODE Actions are added to an ODE Trigger by calling [dsl_ode_trigger_action_add](#dsl_ode_trigger_action_add) and [dsl_ode_trigger_action_add_many](#dsl_ode_trigger_action_add_many), and removed with [dsl_ode_trigger_action_remove](#dsl_ode_trigger_action_remove), [dsl_ode_trigger_action_remove_many](#dsl_ode_trigger_action_remove_many), and [dsl_ode_trigger_action_remove_all](#dsl_ode_trigger_action_remove_all).

#### Adding and Removing Areas
As with Actions, multiple ODE areas can be added to an ODE Trigger and the same ODE Areas can be added to multiple Triggers. ODE Areas are added to an ODE Trigger by calling [dsl_ode_trigger_area_add](#dsl_ode_trigger_area_add) and [dsl_ode_trigger_area_add_many](#dsl_ode_trigger_area_add_many) and removed with [dsl_ode_trigger_action_remove](#dsl_ode_trigger_area_remove), [dsl_ode_trigger_area_remove_many](#dsl_ode_trigger_area_remove_many), and [dsl_ode_trigger_area_remove_all](#dsl_ode_trigger_area_remove_all).


**Important Notes** 
* Be careful when creating No-Limit ODE Triggers with Actions that save data to file as these operations can consume all available diskspace.
* To use GIE Confidence as criteria, see the following NVIDIA [page](https://forums.developer.nvidia.com/t/nvinfer-is-not-populating-confidence-field-in-nvdsobjectmeta-ds-4-0/79319/20) for the required DS 4.02 patch instructions to populate the confidence values in the object's meta data structure.

**Constructors:**
* [dsl_ode_trigger_always_new](#dsl_ode_trigger_always_new)
* [dsl_ode_trigger_absence_new](#dsl_ode_trigger_absence_new)
* [dsl_ode_trigger_occurrence_new](#dsl_ode_trigger_occurrence_new)
* [dsl_ode_trigger_summation_new](#dsl_ode_trigger_summation_new)
* [dsl_ode_trigger_intersection_new](#dsl_ode_trigger_intersection_new)
* [dsl_ode_trigger_minimum_new](#dsl_ode_trigger_minimum_new)
* [dsl_ode_trigger_maximum_new](#dsl_ode_trigger_maximum_new)
* [dsl_ode_trigger_range_new](#dsl_ode_trigger_range_new)
* [dsl_ode_trigger_smallest_new](#dsl_ode_trigger_smallest_new)
* [dsl_ode_trigger_largest_new](#dsl_ode_trigger_largest_new)
* [dsl_ode_trigger_custom_new](#dsl_ode_trigger_custom_new)

**Destructors:**
* [dsl_ode_trigger_delete](#dsl_ode_trigger_delete)
* [dsl_ode_trigger_delete_many](#dsl_ode_trigger_delete_many)
* [dsl_ode_trigger_delete_all](#dsl_ode_trigger_delete_all)

**Methods:**
* [dsl_ode_trigger_reset](#dsl_ode_trigger_reset)
* [dsl_ode_trigger_enabled_get](#dsl_ode_trigger_enabled_get)
* [dsl_ode_trigger_enabled_set](#dsl_ode_trigger_enabled_set)
* [dsl_ode_trigger_class_id_get](#dsl_ode_trigger_class_id_get)
* [dsl_ode_trigger_class_id_set](#dsl_ode_trigger_class_id_set)
* [dsl_ode_trigger_source_id_get](#dsl_ode_trigger_source_id_get)
* [dsl_ode_trigger_source_id_set](#dsl_ode_trigger_source_id_set)
* [dsl_ode_trigger_confidence_min_get](#dsl_ode_trigger_confidence_min_get)
* [dsl_ode_trigger_confidence_min_set](#dsl_ode_trigger_confidence_min_set)
* [dsl_ode_trigger_dimensions_min_get](#dsl_ode_trigger_dimensions_min_get)
* [dsl_ode_trigger_dimensions_min_set](#dsl_ode_trigger_dimensions_min_set)
* [dsl_ode_trigger_dimensions_max_get](#dsl_ode_trigger_dimensions_max_get)
* [dsl_ode_trigger_dimensions_max_set](#dsl_ode_trigger_dimensions_max_set)
* [dsl_ode_trigger_infer_done_only_get](#dsl_ode_trigger_infer_done_only_get)
* [dsl_ode_trigger_infer_done_only_set](#dsl_ode_trigger_infer_done_only_set)
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

---
## Return Values
The following return codes are used by the ODE Trigger API
```C++
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
#define DSL_RESULT_ODE_TRIGGER_CLIENT_CALLBACK_INVALID              0x000E000D
```

---
## Constants
The following symbolic constants are used by the ODE Trigger API
```C++
#define DSL_ODE_ANY_SOURCE                                          INT32_MAX
#define DSL_ODE_ANY_CLASS                                           INT32_MAX
#define DSL_ODE_TRIGGER_LIMIT_NONE                                  0
#define DSL_ODE_TRIGGER_LIMIT_ONE                                   1
```

---

## Callback Typedefs
### *dsl_ode_check_for_occurrence_cb*
```C++
typedef boolean (*dsl_ode_check_for_occurrence_cb)(void* buffer,
    void* frame_meta, void* object_meta, void* client_data);
```
Defines a Callback typedef for a Custom ODE Trigger. Once registered, the function will be called on every object detected that meets the (optional) criteria for the Custom Trigger. The client, determining that **custom** criteria has been met, returns true signaling ODE occurrence. The Custom Trigger will then invoke all the client provided Actions.

**Parameters**
* `buffer` - [in] pointer to frame buffer containing the Metadata for the object detected
* `frame_meta` - [in] opaque pointer to a frame_meta structure that triggered the ODE event
* `object_meta` - [in] opaque pointer to an object_meta structure that triggered the ODE event
* `client_data` - [in] opque point to client user data provided by the client on callback registration

<br>

---

## Constructors
### *dsl_ode_trigger_absence_new* 
```C++
DslReturnType dsl_ode_trigger_always_new(const wchar_t* name, uint when);
```

The constructor creates an Always trigger that triggers and ODE occurrece on every new frame. Note, this is a No-Limit trigger, and setting a Class ID filer will have no effect. The Source ID default == ANY_SOURCE and can be update to specificy a single source id. Although always triggered, the client selects when to Trigger an ODE occurrence for each frame. before (pre) or after (post) processing of all Object metadata by all other Triggers.

Always triggers are helpful for adding [Display Types](/dsoc/api-display-types.md) -- text, lines, rectangles, etc. -- to each frame for one or all sources. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `when` - [in] either DSL_PRE_CHECK_FOR_OCCURRENCES or DSL_POST_CHECK_FOR_OCCURRENCES

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this operation can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_always_new('my-always-trigger', DSL_PRE_CHECK_FOR_OCCURRENCES)
```

<br>

### *dsl_ode_trigger_absence_new* 
```C++
DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, uint class_id, uint limit);
```

The constructor creates an Absence trigger that checks for the absence of Objects within a frame and generates an ODE occurrence if no object occur.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this operation can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_absence_new('my-absence-trigger', PGIE_PERSON_CLASS_ID, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_occurrence_new*
```C++
DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name, uint class_id, uint limit);
```

The constructor creates an Occurrence trigger that checks for the occurrence of Objects within a frame and generates an ODE occurrence invoking all ODE Action for **each** object detected that passes the triggers (option) criteria.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_absence_new('my-absence-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_summation_new*
```C++
DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name, uint class_id, uint limit);    
```
This constructor creates a uniquely named Summation trigger that counts the number Objects within a frame that pass the trigger's (option) criteria. The Trigger generates an ODE occurrence invoking all ODE Actions once for **per-frame** until the Trigger limit is reached. 

Note: Adding Actions to a Summation Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_summation_new('my-summation-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_intersection_new*
```C++
DslReturnType dsl_ode_trigger_intersection_new(const wchar_t* name, uint class_id, uint limit);
```

This constructor creates a uniquely named Intersection Trigger that determines if Objects, that meet the Trigger's (optional) criteria, intersect, and generates an ODE occurrence invoking all ODE Actions twice, once for **each object** in the intersection pair. 

For example: Given three objects A, B, and C. If A intersects B and B intersects C, then two unique ODE occurrences are generated. Each Action owned by the Trigger will be called for each object for every overlapping pair, i.e. a total of four times in this example.  If each of the three objects intersect with the other two, then three ODE occurrences will be triggered with each action called a total of 6 times. 

Intersection requires at least one pixel of overlap between a pair of object's rectangles.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_intersection_new('my-intersection-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_minimum_new*
```C++
DslReturnType dsl_ode_trigger_minimum_new(const wchar_t* name, 
    uint class_id, uint limit, uint minimum);
```

The constructor creates a uniquely named Minimum Occurrence Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's (optional) criteria against a specified minimum number. The Trigger generates an ODE occurrence invoking all Actions if the minimum is not met.

Note: Adding Actions to a Minimum Occurrence Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `minimum` [in] the required minimum number of objects per-frame 

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_minimum_new('my-minimum-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, minimum)
```

<br>

### *dsl_ode_trigger_maximum_new*
```C++
DslReturnType dsl_ode_trigger_maximum_new(const wchar_t* name, 
    uint class_id, uint limit, uint maximum);
```

The constructor creates a uniquely named Maximum Occurrence Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's (optional) criteria against a specified minimum number. The Trigger generates an ODE occurrence invoking all Actions if the maximum is exceeded.

Note: Adding Actions to a Range of Occurrences Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_maximum_new('my-maximum-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, maximum)
```

<br>

### *dsl_ode_trigger_range_new*
```C++
DslReturnType dsl_ode_trigger_range_new(const wchar_t* name, 
    uint class_id, uint limit, uint lower, uint upper);
```

This constructor creates a uniquely named Range of Occurrences Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's (optional) criteria against a range of numbers. The Trigger generates an ODE occurrence invoking all Actions if the object count is within range.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `lower` - [in] defines the lower limit in the range of numbers
* `upper` - [in] defines the upper limit in the range of numbers

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_range_new('my-range-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, maximum)
```

<br>

### *dsl_ode_trigger_smallest_new*
```C++
DslReturnType dsl_ode_trigger_smallest_new(const wchar_t* name, uint class_id, uint limit);
```
This constructor creates a uniquely named smallest trigger that checks for the occurrence of Objects within a frame, and if at least one is found, Triggers on the Object with smallest rectangle area.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_smallest_new('my-range-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_largest_new*
```C++
DslReturnType dsl_ode_trigger_largest_new(const wchar_t* name, uint class_id, uint limit);
```
This constructor creates a uniquely named Largest trigger that checks for the occurrence of Objects within a frame, and if at least one is found, Triggers on the Object with largest rectangle area.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_largest_new('my-range-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE)
```

<br>

### *dsl_ode_trigger_custom_new*
```C++
DslReturnType dsl_ode_trigger_custom_new(const wchar_t* name, 
    uint class_id, uint limit, dsl_ode_check_for_occurrence_cb client_checker, void* client_data);
```

The constructor creates a Uniquely named Custom Trigger that checks for the occurrence of Objects within a frame that meets the Triggers (optional) criteria and calls a Callback function that allows the client to customize the Trigger. The Callback function is called with the buffer 


**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id` - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be careful when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_custom_new('my-custom-trigger', 
        DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, my_check_for_occurrence_cb, my_client_data)
```

---
## Destructors
### *dsl_ode_trigger_delete*
```C++
DslReturnType dsl_ode_trigger_delete(const wchar_t* trigger);
```
This destructor deletes a single, uniquely named ODE Trigger. The destructor will fail if the Trigger is currently `in-use` by an ODE Handler

**Parameters**
* `trigger` - [in] unique name for the ODE Trigger to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_delete('my-trigger')
```

<br>

### *dsl_ode_trigger_delete_many*
```C++
DslReturnType dsl_trigger_delete_many(const wchar_t** triggers);
```
This destructor deletes multiple uniquely named ODE Triggers. Each name is checked for existence, with the function returning `DSL_RESULT_TRIGGER_NAME_NOT_FOUND` on first occurrence of failure. The destructor will fail if one of the Actions is currently `in-use` by one or more ODE Triggers

**Parameters**
* `trigger` - [in] a NULL terminated array of uniquely named ODE Triggers to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

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

### *dsl_ode_trigger_enabled_get*
```c++
DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled);
```

This service returns the current enabled setting for the named ODE Trigger. Note: Triggers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `enabled` - [out] true if the ODE Trigger is currently enabled, false otherwise

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

This service sets the enabled setting for the named ODE Trigger. Note: Triggers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `enabled` - [in] set to true to enable the ODE Trigger, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_enabled_set('my-trigger', False)
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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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
retval, enabled = dsl_ode_trigger_class_id_set('my-trigger', DSL_ODE_ANY_CLASS)
```

<br>

### *dsl_ode_trigger_source_id_get*
```c++
DslReturnType dsl_ode_trigger_source_id_get(const wchar_t* name, uint* source_id);
```

This service returns the current source_id filter setting for the named ODE Trigger. A value of `DSL_ODE_ANY_SOURCE` (default) indicates that the filter is disable and the Unique Source Id will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `source_id` - [out] current source Id filter for the ODE Trigger to filter on, set to `DSL_ODE_ANY_SOURCE` during construction

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, class_id = dsl_ode_trigger_source_id_get('my-trigger')
```

<br>

### *dsl_ode_trigger_source_id_set*
```c++
DslReturnType dsl_ode_trigger_source_id_set(const wchar_t* name, uint source_id);
```

This service sets the current source_id filter setting for the named ODE Trigger. A value of `DSL_ODE_ANY_SOURCE` disables the filter and the Source Id will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `source_id` - [in] new Source Id filter for the ODE Trigger to filter on, or `DSL_ODE_ANY_SOURCE` to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_ode_trigger_source_id_set('my-trigger', 0)
```

<br>

### *dsl_ode_trigger_confidence_min_get*
```c++
DslReturnType dsl_ode_trigger_confidence_min_get(const wchar_t* name, double* min_confidence);
```

This service returns the current minimum confidence criteria for the named ODE Trigger. A value of 0 (default) indicates that the criteria is disable and the detected object's GIE confidence value will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_confidence` - [out] current minimum confidence value between 0.0 and 1.0 for the ODE Trigger to filter on, 0 indicates disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, min_confidence = dsl_ode_trigger_confidence_min_get('my-trigger')
```

<br>

### *dsl_ode_trigger_confidence_min_set*
```c++
DslReturnType dsl_ode_trigger_confidence_min_set(const wchar_t* name, double min_confidence);
```

This service sets the minimum confidence criteria for the named ODE Trigger. A value of 0 disables the filter and the GIE confidence level will not be used as criteria for ODE occurrence.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to query.
* `min_confidence` - [in] new minimum confidence value as criteria for the ODE Trigger to filter on, or 0 to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_confidence_min_set('my-trigger', min_confidence)
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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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
* `infer_done_only` - [out] if set to true, then the "inference-done" filer is enable, false indicates disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_infer_done_only_get('my-trigger', 'infer_done_only')
```

<br>

### *dsl_ode_trigger_action_add*
```c++
DslReturnType dsl_ode_trigger_action_add(const wchar_t* name, const wchar_t* action);
```

This service adds a named ODE Action to a named ODE Trigger. The same Action can be added to multiple Triggers. 

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `action` - [in] unique name of the ODE Action to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

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
* `actions` - [in] a Null terminated list of unique names of the ODE Actions to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_action_add_many('my-trigger', ['my-action-a', 'my-action-b', 'my-action-c', None])
```

<br>

### *dsl_ode_trigger_action_remove*
```c++
DslReturnType dsl_ode_trigger_action_remove(const wchar_t* name, const wchar_t* action);
```

This service removes a named ODE Action from a named ODE Trigger. The services will fail with DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE if the Action is not currently in-use by the named Trigger

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

This service removes a Null terminated list of named ODE Actions to a named ODE Trigger. The service returns DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE if at any point one of the named Actions is not currently in-use by the named Trigger

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `actions` - [in] a Null terminated list of unique names of the ODE Actions to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

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

This service adds a Null terminated list of named ODE Areas to a named ODE Trigger. The same Area can be added to multiple Triggers. 

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `areas` - [in] a Null terminated list of unique names of the ODE Areas to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_area_add_many('my-trigger', ['my-area-a', 'my-area-b', 'my-area-c', None])
```

<br>

### *dsl_ode_trigger_area_remove*
```c++
DslReturnType dsl_ode_trigger_area_remove(const wchar_t* name, const wchar_t* area);
```

This service removes a named ODE Area from a named ODE Trigger. The services will fail with DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE if the Area is not currently in-use by the named Trigger

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

This service removes a Null terminated list of named ODE Areas to a named ODE Trigger. The service returns DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE if at any point one of the named Areas is not currently in-use by the named Trigger

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `areas` - [in] a Null terminated list of unique names of the ODE Areas to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_trigger_area_remove_all('my-trigger')
```

<br>

### *dsl_ode_trigger_list_size*
```c++
uint dsl_ode_trigger_list_size();
```
This service returns the size of the ODE Trigger container, i.e. the number of Triggers currently in memory. 

**Returns**
* The size of the ODE Trigger container

**Python Example**
```Python
size = dsl_ode_trigger_list_size()
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
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* **ODE-Trigger**
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Types](/docs/api-display-types.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)

   
