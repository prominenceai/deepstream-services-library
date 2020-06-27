# Object Detection Event (ODE) Handler API Reference

The ODE Handler Component manages an ordered collection of [ODE Triggers](/docs/api-ode-trigger.md), each with their own ordered collections of [ODE Actions](/docs/api-ode-action.md) and (optional) [ODE Areas](/docs/api-ode-area.md). The Handler installs a `batch-meta-handler` callback to process the batched metadata for each GST Buffer flowing over the ODE Handler's Source Pad connected to the Sink Pad of the next component; On-Screen-Display for example. The handler extracts the Frame and Object metadata iterating through its collection of ODE Triggers. Triggers, created with specific purpose and criteria, check for the occurrence of specific Object Detection Events (ODEs). On ODE occurrence, the Trigger iterates through its ordered collection of ODE Actions invoking their `handle-ode-occurrence` service. ODE Areas, rectangle locations and dimensions, can be added to Triggers as additional criteria for ODE occurrence. Both Actions and Areas can be shared, or co-owned, by multiple Triggers

#### ODE Handler Construction and Destruction
ODE Handlers are created by calling [dsl_ode_handler_new](#dsl_ode_handler_new). Handlers are deleted by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)

The Handler's name must be unique from all other components. The relationship between Pipeline/Branch and ODE Handler is one to one and a Handler must be removed from a Pipeline/Branch before it can be used with another.

#### Adding to a Pipeline/Branch
ODE Handlers are added to a Pipeline by calling [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) (when adding with other components) and removed with [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

ODE Handlers are added to a Branch by calling [dsl_branch_component_add](/docs/api-branch.md#dsl_branch_component_add) or [dsl_branch_component_add_many](/docs/api-branch.md#dsl_branch_component_add_many) (when adding with other components) and removed with [dsl_branch_component_remove](/docs/api-branch.md#dsl_branch_component_remove), [dsl_branch_component_remove_many](/docs/api-branch.md#dsl_branch_component_remove_many), or [dsl_branch_component_remove_all](/docs/api-branch.md#dsl_branch_component_remove_all).


#### Adding and Removing Triggers
ODE Triggers are added to an ODE Handler by calling [dsl_ode_handler_trigger_add](#dsl_ode_handler_trigger_add) or [dsl_ode_handler_trigger_add_many](#dsl_ode_handler_trigger_add_many) and removed with [dsl_ode_handler_trigger_remove](#dsl_ode_handler_trigger_remove), [dsl_ode_handler_trigger_remove_many](#dsl_ode_handler_trigger_remove_many), or [dsl_ode_handler_trigger_remove_all](#dsl_ode_handler_trigger_remove_all).

## ODE Handler API
**Constructors:**
* [dsl_ode_handler_new](#dsl_ode_handler_new)

**Methods**
* [dsl_ode_handler_enabled_get](#dsl_ode_handler_enabled_get)
* [dsl_ode_handler_enabled_set](#dsl_ode_handler_enabled_set)
* [dsl_ode_handler_trigger_add](#dsl_ode_handler_trigger_add)
* [dsl_ode_handler_trigger_add_many](#dsl_ode_handler_trigger_add_many)
* [dsl_ode_handler_trigger_remove](#dsl_ode_handler_trigger_remove)
* [dsl_ode_handler_trigger_remove_many](#dsl_ode_handler_trigger_remove_many)
* [dsl_ode_handler_trigger_remove_all](#dsl_ode_handler_trigger_remove_all)

---
## Return Values
The following return codes are used by the ODE Handler API
```C++
#define DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE                      0x000D0001
#define DSL_RESULT_ODE_HANDLER_NAME_NOT_FOUND                       0x000D0002
#define DSL_RESULT_ODE_HANDLER_NAME_BAD_FORMAT                      0x000D0003
#define DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION                      0x000D0004
#define DSL_RESULT_ODE_HANDLER_IS_IN_USE                            0x000D0005
#define DSL_RESULT_ODE_HANDLER_SET_FAILED                           0x000D0006
#define DSL_RESULT_ODE_HANDLER_TRIGGER_ADD_FAILED                   0x000D0007
#define DSL_RESULT_ODE_HANDLER_TRIGGER_REMOVE_FAILED                0x000D0008
#define DSL_RESULT_ODE_HANDLER_TRIGGER_NOT_IN_USE                   0x000D0009
#define DSL_RESULT_ODE_HANDLER_COMPONENT_IS_NOT_ODE_HANDLER         0x000D000A
```

## Constructors
### *dsl_ode_handler_new* 
```C++
DslReturnType dsl_ode_handler_new(const wchar_t* name);;
```

The constructor creates a uniquely named ODE Handler.

**Parameters**
* `name` - [in] unique name for the ODE Handler to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_ode_handler_new('my-ode-handler')
```

<br>

### *dsl_ode_handler_enabled_get*
```c++
DslReturnType dsl_ode_handler_enabled_get(const wchar_t* name, boolean* enabled);
```

This service returns the current enabled setting for the named ODE Handler. Note: Handlers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the ODE Handler to query.
* `enabled` - [out] true if the ODE Handler is currently enabled, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_ode_handler_enabled_get('my-handler')
```

<br>

### *dsl_ode_handler_enabled_set*
```c++
DslReturnType dsl_ode_handler_enabled_set(const wchar_t* name, boolean enabled);
```

This service sets the enabled setting for the named ODE Handler. Note: Handlers are enabled by default during construction.

**Parameters**
* `name` - [in] unique name of the ODE Handler to update.
* `enabled` - [in] set to true to enable the ODE Handler, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_action_enabled_set('my-handler', False)
```

<br>

### *dsl_ode_handler_trigger_add*
```c++
DslReturnType dsl_ode_handler_trigger_add(const wchar_t* name, const wchar_t* action);
```

This service adds a named ODE Trigger to a named ODE Handler. The relationship between Handler and Trigger is one-to-many. 

**Parameters**
* `name` - [in] unique name of the ODE Handler to update.
* `trigger` - [in] unique name of the ODE Trigger to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_handler_trigger_add('my-handler', 'my-trigger')
```

<br>

### *dsl_ode_handler_trigger_add_many*
```c++
DslReturnType dsl_ode_handler_trigger_add_many(const wchar_t* name, const wchar_t** triggers);
```

This service adds a Null terminated list of named ODE Triggers to a named ODE Handler. The relationship between Handler and Trigger is one-to-many. 

**Parameters**
* `name` - [in] unique name of the ODE Handler to update.
* `actions` - [in] a Null terminated list of unique names of the ODE Triggers to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_handler_trigger_add_many('my-handler', ['my-trigger-a', 'my-trigger-b', 'my-trigger-c', None])
```

<br>

### *dsl_ode_handler_trigger_remove*
```c++
DslReturnType dsl_ode_handler_trigger_remove(const wchar_t* name, const wchar_t* trigger);
```

This service removes a named ODE Trigger from a named ODE Handler. The services will fail with DSL_RESULT_ODE_HANDLER_TRIGGER_NOT_IN_USE if the Trigger is not currently in-use by the named Handler

**Parameters**
* `name` - [in] unique name of the ODE Handler to update.
* `Trigger` - [in] unique name of the ODE Trigger to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_handler_trigger_remove('my-handler', 'my-trigger')
```

<br>

### *dsl_ode_handler_trigger_remove_many*
```c++
DslReturnType dsl_ode_handler_trigger_remove_many(const wchar_t* name, const wchar_t** triggers);
```

This service removes a Null terminated list of named ODE Triggers from a named ODE Handler. The service returns DSL_RESULT_ODE_HANDLER_TRIGGER_NOT_IN_USE if at any point one of the named Triggers is not currently in-use by the named Handler

**Parameters**
* `name` - [in] unique name of the ODE Handler to update.
* `triggers` - [in] a Null terminated list of unique names of the ODE Triggers to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_handler_trigger_remove_many('my-trigger', ['my-trigger-a', 'my-trigger-b', 'my-trigger-c', None])
```

<br>

### *dsl_ode_handler_trigger_remove_all*
```c++
DslReturnType dsl_ode_handler_trigger_remove_all(const wchar_t* name);
```

This service removes all ODE Triggers from a named ODE Handler. 

**Parameters**
* `name` - [in] unique name of the ODE Handler to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_ode_handler_trigger_remove_all('my-handler')
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
* **ODE Handler**
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Action](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [On-Screen Display](/docs/api-osd.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)

