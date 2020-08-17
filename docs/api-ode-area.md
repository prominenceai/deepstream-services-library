# ODE Area Services API
Object Detection Event (ODE) Areas use a [Rectangle Display Type](/docs/api-display-type.md) with coordinates  (`left` and `top`) and dimensions (`width` and `height`) to be used as additional [ODE Trigger](/docs/api-ode-trigger.md) criteria for an ODE occurrence. 

There are two types of Areas:
* **Area of Inclusion** - at least one pixel of overlap between object and area is required to trigger ODE occurrence.
* **Area of Exclusion** - not one pixel of overlap can occur for ODE occurrence to be triggered. 

The relationship between Triggers and Areas is many-to-many as multiple Areas can be added to a Trigger and the same Area can be added to multiple Triggers.  If a New Ares's `display` is enabled, Areas owned by Triggers will be added as display metadata for an On-Screen-Component to display.

If both Areas of Inclusion and Exclusion are added to an ODE Trigger, the order of addition determines the order of precedence. 

ODE Actions can be used to update a Trigger's container of ODE Areas on ODE occurrence. See [dsl_ode_action_area_add_new](/docs/api-ode-action.md#dsl_ode_action_area_add_new) and [dsl_ode_action_area_remove_new](/docs/api-ode-action.md#dsl_ode_action_area_remove_new). 

#### ODE Area Construction and Destruction
Areas are created by calling one of two type specific constructors: [dsl_ode_area_inclusion_new](#dsl_ode_area_inclusion_new) and [dsl_ode_area_exclusion_new](#dsl_ode_area_exclusion_new)

#### Adding/Removing ODE Areas
ODE Areas are added to to ODE Triggers by calling [dsl_ode_trigger_area_add](/docs/api-ode-trigger.md#dsl_ode_trigger_area_add), [dsl_ode_trigger_area_add_many](/docs/api-ode-trigger.md#dsl_ode_trigger_area_add_many) and deleted with [dsl_ode_trigger_area_remove](/docs/api-ode-trigger.md#dsl_ode_trigger_area_add)

## ODE Area Services API

**Constructors:**
* [dsl_ode_area_inclusion_new](#dsl_ode_area_inclusion_new)
* [dsl_ode_area_exclusion_new](#dsl_ode_area_exclusion_new)

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

<br>

---

## Constructors
### *dsl_ode_area_inclusion_new*
```C++
DslReturnType dsl_ode_area_inclusion_new(const wchar_t* name, 
    const wchar_t* rectangle, boolean display);
```
The constructor creates a uniquely named ODE **Area of Inclusion** using a uniquely named RGBA Rectangle. Inclusion require at least one pixel of overlap between an Object's rectangle and the Area's rectangle is required to trigger ODE occurrence.

The Rectangle can be displayed (requires an [On-Screen Display](/docs/api-osd.md)) or left hidden.

**Parameters**
* `name` - [in] unique name for the ODE Area of Inclusion to create.
* `rectangle` - [in] unique name for the Rectangle to use for coordinates, dimensions, and optionally display
* `display` - [in] if true, rectangle display-metadata will be added to each structure of frame metadata.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_inclusion_new('my-inclusion-area', 'my-inclusion-rectangle', True)
```

<br>

### *dsl_ode_area_exclusion_new*
```C++
DslReturnType dsl_ode_area_exclusion_new(const wchar_t* name, 
    const wchar_t* rectangle, boolean display);
```
The constructor creates a uniquely named ODE **Area of Exclusion** using a uniquely named RGBA Rectangle. Exclusion requires that not one pixel of overlap can occur for ODE occurrence to be triggered. 

The Rectangle can be displayed (requires an [On-Screen Display](/docs/api-osd.md)) or left hidden.

**Parameters**
* `name` - [in] unique name for the ODE Area of Inclusion to create.
* `rectangle` - [in] unique name for the Rectangle to use for coordinates, dimensions, and optionally display.
* `display` - [in] if true, rectangle display-metadata will be added to each structure of frame metadata.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_area_exclusion_new('my-exclusion-area', 'my-exclusion-rectangle', True)
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
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_area_delete('my-area')
```

<br>

### *dsl_ode_area_delete_many*
```C++
DslReturnType dsl_area_delete_many(const wchar_t** area);
```
This destructor deletes multiple uniquely named ODE Areas. Each name is checked for existence, with the function returning `DSL_RESULT_AREA_NAME_NOT_FOUND` on first occurrence of failure. The destructor will fail if one of the Areas is currently `in-use` by one or more ODE Triggers

**Parameters**
* `areas` - [in] a NULL terminated array of uniquely named ODE Areas to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

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
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

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
* The size of the ODE Area container

**Python Example**
```Python
size = dsl_ode_area_list_size()
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
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Action](/docs/api-ode-action.md)
* **ODE-Area**
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
