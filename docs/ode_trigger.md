

## ODE Trigger API

**Important Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as these operations can consume all available diskspace.

**Constructors:**
* [dsl_ode_trigger_absence_new](#dsl_ode_trigger_absence_new)
* [dsl_ode_trigger_occurrence_new](#dsl_ode_trigger_occurrence_new)
* [dsl_ode_trigger_summation_new](#dsl_ode_trigger_summation_new)
* [dsl_ode_trigger_intersection_new](#dsl_ode_trigger_intersection_new)
* [dsl_ode_trigger_minimum_new](#dsl_ode_trigger_minimum_new)
* [dsl_ode_trigger_maximum_new](#dsl_ode_trigger_maximum_new)
* [dsl_ode_trigger_range_new](#dsl_ode_trigger_range_new)
* [dsl_ode_trigger_custom_new](#dsl_ode_trigger_custom_new)

**Destructors:**
* [dsl_ode_trigger_delete](#dsl_ode_trigger_delete)
* [dsl_ode_trigger_delete_many](#dsl_ode_trigger_delete_many)
* [dsl_ode_trigger_delete_all](#dsl_ode_trigger_delete_all)

**Methods:**

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
Defines a Callback typedef for a client ODE Custom Trigger check-for-occurrence function. Once registered, the function will be called on every object detected that meets the minimum criteria for the Custom Trigger. The client, determining that the **custom** criteria is met for ODE occurrence, returns true to invoke all ODE acctions owned by the Custom Trigger

**Parameters**
* `buffer` - [in] pointer to frame buffer containing the Metadata for the object detected
* `frame_meta` - [in] opaque pointer to a frame_meta structure that triggered the ODE event
* `object_meta` - [in] opaque pointer to a object_meta structure that triggered the ODE event
* `client_data` - [in] opque point to client user data provided by the client on callback registration

<br>
---

## Constructors
### *dsl_ode_trigger_absence_new* 
```C++
DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, uint class_id, uint limit);
```

The constructor creates an Absence trigger that checks for the absence of Objects within a frame, and generates an ODE occurence if no object occur.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this operation can consume all available diskspace.

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

The constructor creates an Occurrence trigger that checks for the occurrence of Objects within a frame, and generates an ODE occurence invoking all ODE Action for **each** object detected that passes the triggers (option) critera.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

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
This constructor creates a uniquly named Summation trigger that counts the number Objects within a frame that pass the trigger's (option) critera. The Trigger generates an ODE occurence invoking all ODE Actions once for **per-frame** until the Trigger limit is reached. 

Note: Adding Actions to a Summation Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

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

This constructor creates a uniquely named Intersection Trigger that determines if Objects, that meet the Trigger's (optional) critera, intersect, and generates an ODE occurence invoking all ODE Actions twice, once for **each object** in the intersection pair. 

For example: Given three objects A, B, and C. If A intersects B and B intersects C, then two unique ODE occurrence are generated. Each Action owned by the Trigger will be called for each object for every overlapping pair, i.e. a total of four times in this example.  If each of the three objects intersect with the other two, then three ODE occurrences will be triggered with each action called a total of 6 times. 

Intersection requires at least one pixel of overlap between a pair of object's rectangle.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

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

The constructor creates a uniquely named Minimum Occurence Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's (optional) criteria against a specified minimum number. The Trigger generates an ODE occurence invoking all Actions if the minimum is not met.

Note: Adding Actions to a Minimum Occurence Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `minimum` [in] the required minumum number of objects per-frame 

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_minimum_new('my-minimum-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, minmimum)
```

<br>

### *dsl_ode_trigger_maximum_new*
```C++
DslReturnType dsl_ode_trigger_maximum_new(const wchar_t* name, 
    uint class_id, uint limit, uint maximum);
```

The constructor creates a uniquely named Maximum Occurence Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's (optional) criteria against a specified minimum number. The Trigger generates an ODE occurence invoking all Actions if the maximum is exceeded.

Note: Adding Actions to a Range of Occurences Trigger that require Object metadata during invocation - Object-Capture and Object-Fill as examples - will result in a non-action when invoked. 

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_maximum_new('my-maximum-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, maxmimum)
```

<br>

### *dsl_ode_trigger_range_new*
```C++
DslReturnType dsl_ode_trigger_range_new(const wchar_t* name, 
    uint class_id, uint limit, uint lower, uint upper);
```

This constructor creates a uniquely named Range of Occurences Trigger that checks for the occurrence of Objects within a frame that meet the Trigger's (optional) criteria against a range of numbers. The Trigger generates an ODE occurence invoking all Actions if the object count is within range.

**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.
* `lower` - [in] defines the lower limit in the range of numbers
* `upper` - [in] defines the upper limit in the range of numbers

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_trigger_range_new('my-range-trigger', DSL_ODE_ANY_CLASS, DSL_ODE_TRIGGER_LIMIT_NONE, my_maxmimum)
```

<br>

### *dsl_ode_trigger_custom_new*
```C++
DslReturnType dsl_ode_trigger_custom_new(const wchar_t* name, 
    uint class_id, uint limit, dsl_ode_check_for_occurrence_cb client_checker, void* client_data);
```

The constructor creates a Uniquely named Custom Trigger that checks for the occurrence of Objects within a frame that meets the Triggers (optional) critera, and calls a Callback function that allows the client to customize the Trigger. The Callback function is called with the buffer 


**Parameters**
* `name` - [in] unique name for the ODE Trigger to create.
* `class_id - [in] inference class id filter. Use DSL_ODE_ANY_CLASS to disable the filter
* `limit` - [in] the Trigger limit. Once met, the Trigger will stop triggering new ODE occurrences. Set to DSL_ODE_TRIGGER_LIMIT_NONE (0) for no limit.

**Note** Be carefull when creating No-Limit ODE Triggers with Actions that save data to file as this can consume all available diskspace.

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
This destructor deletes a single, uniquely named ODE Trigger. The destructor will fail if the Trigger is currently `in-use` by a ODE Handler

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

### *dsl_ode_action_trigger_all*
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
### *dsl_ode_trigger_enabled_get*
```c++
DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled);
```
This service returns the current enabled setting for the named ODE Trigger. Note: Trigger are enabled by default at the time of construction.

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
```c++
DslReturnType dsl_ode_trigger_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the enabled setting for the named ODE Trigger. Note: Triggers are enabled by default at the time of construction.

**Parameters**
* `name` - [in] unique name of the ODE Trigger to update.
* `enabled` - [in] set to true to enable the ODE Trigger, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_action_enabled_set('my-action', False)
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
* [ODE Handler](/docs/ode-handler.md)
* **ODE-Trigger**
* [ODE Area](/docs/ode-area.md)
* [ODE Action](/docs/ode-action.md)
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)

   
