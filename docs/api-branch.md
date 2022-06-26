# Branch API Reference
Branch components are similar to Pipelines in the way they're used to manage and link Child components when transitioning to states of `ready`, `paused`, and `playing`. Unlike Pipelines, Branches can not contain Source components, and must be added to a [Demuxer or Splitter Tee](/docs/api-tee.md). 

#### Branch Construction and Destruction
Branches are constructed by calling [dsl_branch_new](#dsl_branch_new),[dsl_branch_new_many](#dsl_branch_new_many),  [dsl_branch_new_component_add_many](#dsl_branch_new_component_add_many), and destructed by calling [dsl_branch_delete](#dsl_branch_delete), [dsl_branch_delete_many](#dsl_branch_delete_many), or [dsl_branch_delete_all](#dsl_branch_delete_all). Deleting a Branch will not delete its child components but will unlink then and return to a state of `not-in-use`. The client application is responsible for deleting all child components by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all), or [dsl_delete_all](/docs/overview.md#dsl-delete-all).
#### Adding and Removing Components
Child components -- Inference Engines, Trackers, Tilers, Demuxers, Splitters, On Screen-Displays, and Sinks -- are added to a Branch by calling [dsl_branch_component_add](#dsl_branch_component_add) and [dsl_branch_component_add_many](#dsl_branch_component_add_many).

Child components can be removed from their Parent Branch by calling [dsl_branch_component_remove](#dsl_branch_componet_remove), [dsl_branch_component_remove_many](#dsl_branch_componet_remove_many), and [dsl_branch_component_remove_all](#dsl_branch_component_remove_all).

#### Adding and Removing Branches from Tees
Once created, Branches are added to a Demuxer or Splitter (Tees) by calling [dsl_tee_branch_add](#dsl_tee_branch_add) and [dsl_tee_branch_add_many](#dsl_tee_branch_add_many). A Tee's current number of child components can be obtained by calling [dsl_tee_branch_list_size](#dsl_tee_branch_list_size).

Branches can be removed from their Parent Tee by calling [dsl_tee_branch_remove](#dsl_tee_branch_remove), [dsl_tee_branch_remove_many](#dsl_tee_branch_remove_many), and [dsl_tee_branch_remove_all](#dsl_tee_branch_remove_all)

--
## Branch API
**Constructors**
* [dsl_branch_new](#dsl_branch_new)
* [dsl_branch_new_many](#dsl_branch_new_many)
* [dsl_branch_new_component_add_many](#dsl_branch_new_component_add_many)

**Methods**
* [dsl_branch_component_add](#dsl_branch_component_add)
* [dsl_branch_component_add_many](#dsl_branch_component_add_many)
* [dsl_branch_component_remove](#dsl_branch_component_remove)
* [dsl_branch_component_remove_many](#dsl_branch_component_remove_many)
* [dsl_branch_component_remove_all](#dsl_branch_component_remove_all)

---
## Return Values
The following return codes are used by the Pipeline API
```C++
#define DSL_RESULT_BRANCH_RESULT                                    0x000B0000
#define DSL_RESULT_BRANCH_NAME_NOT_UNIQUE                           0x000B0001
#define DSL_RESULT_BRANCH_NAME_NOT_FOUND                            0x000B0002
#define DSL_RESULT_BRANCH_NAME_BAD_FORMAT                           0x000B0003
#define DSL_RESULT_BRANCH_THREW_EXCEPTION                           0x000B0004
#define DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED                      0x000B0005
#define DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED                   0x000B0006
#define DSL_RESULT_BRANCH_SOURCE_NOT_ALLOWED                        0x000B0007
#define DSL_RESULT_BRANCH_SINK_MAX_IN_USE_REACHED                   0x000B0008
```

---

## Constructors
### *dsl_branch_new*
```C++
DslReturnType dsl_branch_new(const wchar_t* name);
```
The constructor creates a uniquely named Branch. Construction will fail
if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Branch to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_new('my-branch')
```

<br>

### *dsl_branch_new_many*
```C++
DslReturnType dsl_branch_new_many(const wchar_t** names);
```
The constructor creates multiple uniquely named Branches at once. All names are checked for uniqueness with the call returning on failure.

**Parameters**
* `names` - [in] a NULL terminated array of unique names for the Branches to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation of all Branches. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_new_many(['my-branch-a', 'my-branch-b', 'my-branch-c', None])
```

<br>

### *dsl_branch_new_component_add_many*
```C++
DslReturnType dsl_branch_new_component_add_many(const wchar_t* name, 
    const wchar_t** components);
```
Creates a new Branch with a given list of named Components. The add service will fail if any of the Components are currently `in-use` by another Pipeline or Branch. All the Component's `in-use` states will be set to true on successful add. 



**Parameters**
* `name` - [in] unique name for the Branch to create.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_new_component_add_many('my-branch', ['my-pgie', 'my-tiler', 'my-window-sink', None])
```

<br

---

## Methods

### *dsl_branch_component_add*
```C++
DslReturnType dsl_branch_component_add(const wchar_t* name, 
    const wchar_t* component);
```
Adds a single named Component to a named Branch. The add service will fail if the component is currently `in-use` by another Pipeline or Branch. The add service will also fail if adding a `one-only` type of Component, such as a Tiled-Display, when the Branch already has one. The Component's `in-use` state will be set to `true` on successful add. 

**Parameters**
* `name` - [in] unique name for the Branch to update.
* `component` - [in] unique name of the Component to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_component_add('my-branch', 'my-tiler’)
```

<br>

### *dsl_branch_component_add_many*
```C++
DslReturnType dsl_branch_component_add_many(const wchar_t* name, 
    const wchar_t** components);
```
Adds a list of named Components to a named Branch. The add service will fail if any of components are currently `in-use` by any Branch. The add service will fail if any of the components are of a `one-only` type, a Tiler for example, for which the Branch already has. All the component's `in-use` states will be set to true on successful add. 

**Parameters**
* `name` - [in] unique name for the Branch to update.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_component_add_many('my-branch', ['my-pgie', 'my-tiler', 'my-window-sink', None])
```

<br>

### *dsl_branch_component_remove*
```C++
DslReturnType dsl_branch_component_remove(const wchar_t* name, 
    const wchar_t* component);
```
Removes a single named Component from a named Branch. The remove service will fail if the Component is not currently `in-use` by the Branch. The Component's `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Branch to update.
* `component` - [in] unique name of the Component to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_component_remove('my-branch', 'my-tiler')
```
<br>

### *dsl_branch_component_remove_many*
```C++
DslReturnType dsl_branch_component_remove_many(const wchar_t* name, 
    const wchar_t** components);
```
Removes a list of named components from a named Branch. The remove service will fail if any of components are currently `not-in-use` by the named Branch.  All of the removed component's `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Branch to update.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_component_remove_many('my-branch', ['my-pgie', 'my-tiler', 'my-window-sink', None])
```

<br>

### *dsl_branch_component_remove_all*
```C++
DslReturnType dsl_branch_component_add_(const wchar_t* name);
```
Removes all child components from a named Branch. The add service will fail if any of components are currently `not-in-use` by the named Branch.  All the removed Component's `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Branch to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_component_remove_all('my-branch')
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
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* **Branch**
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
