# ODE Accumulator API Reference
An Object Detection Event (ODE) Accumulator -- once added to an ODE Trigger -- accumulates ODE occurrence metrics over subsequent frames. The ODE Triger calls on its ODE Accumulator while post-processing each frame with the count of new ODE occurrences that where triggered. The ODE Accumulator updates the running accumulative total(s) and then iterates through its collection of ordered [ODE Actions](/docs/api-ode-action.md) invoking each. 

One example is to add a [Display Action](/docs/api-ode-action.md#dsl_ode_action_display_new) to an Accumulator added to a [Cross Trigger](/docs/api-ode-trigger-md#dsl_ode_trigger_cross_new) with an [ODE Area](/docs/api-ode-area.md). The Cross Trigger will call the Accumlator with the count of `IN` and `OUT` occurrences after each frame. The Accumlator will update the running totals and call the Display Action. The Display Action will format the data into RGBA Display Meta and add it to the current frame for a downstream [On-Screen Display (OSD)](/docs/api-osd.md) to display.

#### Construction and Destruction
An Accumulator is created by calling [dsl_ode_accumulator_new](#dsl_ode_accumulator_new). Accumulators are deleted by calling [dsl_ode_accumulator_delete](#dsl_ode_accumulator_delete), [dsl_ode_accumulator_delete_many](#dsl_ode_accumulator_delete_many), or [dsl_ode_accumulator_delete_all](#dsl_ode_accumulator_delete_all).

#### Adding and Removing Accumulators
The relationship between ODE Triggers and ODE Accumulators is many-to-one. A Trigger can have at most one Accumlator and one Accumulator can be added to multiple Triggers. An ODE Accumulator is added to an ODE Trigger by calling [dsl_ode_trigger_accumulator add](/docs/api-ode-trigger.md#dsl_ode_trigger_accumulator_add) and removed with [dsl_ode_trigger_accumulator_remove](docs/api-ode-trigger.md#dsl_ode_trigger_accumulator_remove).

#### Adding and Removing Actions
Multiple ODE Actions can be added to an ODE Accumulator and the same ODE Action can be added to multiple ODE Accumulators. ODE Actions are added to an ODE Accumulator by calling [dsl_ode_accumulator_action_add](#dsl_ode_accumulator_action_add) and [dsl_ode_accumulator_action_add_many](#dsl_ode_accumulator_action_add_many), and removed with [dsl_ode_accumulator_action_remove](#dsl_ode_accumulator_action_remove), [dsl_ode_accumulator_action_remove_many](#dsl_ode_accumulator_action_remove_many), and [dsl_ode_accumulator_action_remove_all](#dsl_ode_accumulator_action_remove_all).

---

## ODE Accumulator API
**Constructors:**
* [dsl_ode_accumulator_new](#dsl_ode_accumulator_new)

**Destructors:**
* [dsl_ode_accumulator_delete](#dsl_ode_accumulator_delete)
* [dsl_ode_accumulator_delete_many](#dsl_ode_accumulator_delete_many)
* [dsl_ode_accumulator_delete_all](#dsl_ode_accumulator_delete_all)

**Methods:**
* [dsl_ode_accumulator_action_add](#dsl_ode_accumulator_action_add)
* [dsl_ode_accumulator_action_add_many](#dsl_ode_accumulator_action_remove_many)
* [dsl_ode_accumulator_action_remove](#dsl_ode_accumulator_action_remove)
* [dsl_ode_accumulator_action_remove_many](#dsl_ode_accumulator_action_remove_many)
* [dsl_ode_accumulator_action_remove_all](#dsl_ode_accumulator_action_remove_all)
* [dsl_ode_accumulator_list_size](#dsl_ode_accumulator_list_size)

---

## Return Values
The following return codes are used by the ODE Accumulator API
```C++
#define DSL_RESULT_ODE_ACCUMULATOR_RESULT                           0x00900000
#define DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_UNIQUE                  0x00900001
#define DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_FOUND                   0x00900002
#define DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION                  0x00900003
#define DSL_RESULT_ODE_ACCUMULATOR_IN_USE                           0x00900004
#define DSL_RESULT_ODE_ACCUMULATOR_SET_FAILED                       0x00900005
#define DSL_RESULT_ODE_ACCUMULATOR_IS_NOT_ODE_ACCUMULATOR           0x00900006
#define DSL_RESULT_ODE_ACCUMULATOR_ACTION_ADD_FAILED                0x00900007
#define DSL_RESULT_ODE_ACCUMULATOR_ACTION_REMOVE_FAILED             0x00900008
#define DSL_RESULT_ODE_ACCUMULATOR_ACTION_NOT_IN_USE                0x00900009
```

---

## Constructors
### *dsl_ode_accumulator_new*
```C++
DslReturnType dsl_ode_accumulator_new(const wchar_t* name);
```

The constructor creates a new ODE Accumulator that when added to an ODE Trigger, accumulates the count(s) of ODE occurrence and calls on all ODE Actions during the Trigger's post processing of each frame.

**Parameters**
* `name` - [in] unique name for the ODE Accumulator to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_accumulator_new('my-accumulator')
```

<br>

---

## Destructors
### *dsl_ode_accumulator_delete*
```C++
DslReturnType dsl_ode_accumulator_delete(const wchar_t* name);
```
This destructor deletes a single, uniquely named ODE Accumulator. The destructor will fail if the Accumulator is currently `in-use` by an ODE Trigger

**Parameters**
* `name` - [in] unique name for the ODE Accumulator to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_accumulator_delete('my-accumulator')
```

<br>

### *dsl_ode_accumulator_delete_many*
```C++
DslReturnType dsl_ode_accumulator_delete_many(const wchar_t** names);
```
This destructor deletes multiple uniquely named ODE Accumulators. Each name is checked for existence with the function returning on first failure. The destructor will fail if one of the Actions is currently `in-use` by one or more ODE Triggers

**Parameters**
* `names` - [in] a NULL terminated array of uniquely named ODE Accumulators to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure. 

**Python Example**
```Python
retval = dsl_ode_accumulator_delete_many(['my-accumulator-a', 'my-accumulator-b', 'my-accumulator-c', None])
```

<br>

### *dsl_ode_accumulator_delete_all*
```C++
DslReturnType dsl_ode_accumulator_delete_all();
```
This destructor deletes all ODE Accumulators currently in memory. The destructor will fail if any of the Accumulators are currently `in-use` by an ODE Trigger.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_accumulator_delete_all()
```

<br>

---

## Methods
### *dsl_ode_accumulator_action_add*
```c++
DslReturnType dsl_ode_accumulator_action_add(const wchar_t* name, 
    const wchar_t* action);
```

This service adds a named ODE Action to a named ODE Accumulator. The same Action can be added to multiple Accumulators.

**Parameters**
* `name` - [in] unique name of the ODE Accumulator to update.
* `action` - [in] unique name of the ODE Action to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_accumulator_action_add('my-accumulator', 'my-action')
```

<br>

### *dsl_ode_accumulator_action_add_many*
```C
DslReturnType dsl_ode_accumulator_action_add_many(const wchar_t* name, 
    const wchar_t** actions);
```

This service adds a Null terminated array of named ODE Actions to a named ODE Accumulator. The same Actions can be added to multiple Accumulators.

**Parameters**
* `name` - [in] unique name of the ODE Accumulator to update.
* `actions` - [in] a Null terminated list of unique names of the ODE Actions to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_accumulator_action_add_many('my-accumulator', 
    ['my-action-a', 'my-action-b', 'my-action-c', None])
```

<br>

### *dsl_ode_accumulator_action_remove*
```c++
DslReturnType dsl_ode_accumulator_action_remove(const wchar_t* name, 
    const wchar_t* action);
```

This service removes a named ODE Action from a named ODE Accumulator. The services will fail if the Action is not currently in-use by the named Accumulator

**Parameters**
* `name` - [in] unique name of the ODE Accumulator to update.
* `action` - [in] unique name of the ODE Action to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_accumulator_action_remove('my-accumulator', 'my-action')
```

<br>

### *dsl_ode_accumulator_action_remove_many*
```c++
DslReturnType dsl_ode_accumulator_action_remove_many(const wchar_t* name, 
    const wchar_t** actions);
```

This service removes a Null terminated list of named ODE Actions to a named ODE Accumulator. The service will fail if any of the named Actions are not currently in-use by the named Accumulator.

**Parameters**
* `name` - [in] unique name of the ODE Accumulator to update.
* `actions` - [in] a Null terminated list of unique names of the ODE Actions to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_accumulator_action_remove_many('my-accumulator', 
    ['my-action-a', 'my-action-b', 'my-action-c', None])
```

<br>

### *dsl_ode_accumulator_action_remove_all*
```c++
DslReturnType dsl_ode_accumulator_action_remove_all(const wchar_t* name);
```

This service removes all ODE Actions from a named ODE Accumulator.

**Parameters**
* `name` - [in] unique name of the ODE Accumulator to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_accumulator_action_remove_all('my-accumulator')
```

<br>

### *dsl_ode_accumulator_list_size*
```c++
uint dsl_ode_accumulator_list_size();
```

This service returns the size of the list of ODE Accumulators.

**Returns**
* The current number of ODE Accumulators in memory.

**Python Example**
```Python
size = dsl_ode_accumulator_list_size()
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
* [ODE Trigger](/docs/api-ode-trigger.md)
* **ODE Accumulator**
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Types](/docs/api-display-types.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)  
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
