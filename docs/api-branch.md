# Branch API Reference
Branch components are similar to Piplines in the way they're used to manage and link Child components when transitioning to states of `ready`, `paused`, and `playing`. Unlike Pipelines, Branches can not contain Source components, and must be added as a Child to a Tee, either a Demuxer or Splitter. 

#### Branch Construction and Destruction
Branches are constructed by calling [dsl_branch_new](#dsl_branch_new) or [dsl_branch_new_many](#dsl_branch_new_many), and destructed by calling [dsl_branch_delete](#dsl_branch_delete), [dsl_branch_delete_many](#dsl_branch_delete_many), or [dsl_branch_delete_all](#dsl_branch_delete_all). Deleting a Branch will not delete its child components but will unlink then and return to a state of `not-in-use`. The client application is responsible for deleting all child components by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all).

#### Adding and Removing Components
Child components -- Inference Engines, Trackers, Tilers, Demuxers, Splitters, On Screen-Displays, and Sinks -- are added to a Branch by calling [dsl_branch_component_add](#dsl_branch_component_add) and [dsl_branch_component_add_many](#dsl_branch_component_add_many). A Branch's current number of child components can be obtained by calling [dsl_branch_component_list_size](#dsl_branch_component_list_size).

Child components can be removed from their Parent Branch by calling [dsl_branch_component_remove](#dsl_branch_componet_remove), [dsl_branch_component_remove_many](#dsl_branch_componet_remove_many), and [dsl_branch_component_remove_all](#dsl_branch_component_remove_all)

#### Adding and Removing Branches from Tees
Once created, Branches are added to a Demuxer or Splitter (Tees) by calling [dsl_tee_branch_add](#dsl_tee_branch_add) and [dsl_tee_branch_add_many](#dsl_tee_branch_add_many). A Tee's current number of child components can be obtained by calling [dsl_tee_branch_list_size](#dsl_tee_branch_list_size).

Branches can be removed from their Parent Tee by calling [dsl_tee_branch_remove](#dsl_tee_branch_remove), [dsl_tee_branch_remove_many](#dsl_tee_branch_remove_many), and [dsl_tee_branch_remove_all](#dsl_tee_branch_remove_all)

--
## Branch API
**Constructors**
* [dsl_branch_new](#dsl_branch_new)
* [dsl_branch_new_many](#dsl_branch_new_many)

**Methods**
* [dsl_branch_component_add](#dsl_branch_component_add)
* [dsl_branch_component_add_many](#dsl_branch_component_add_many)
* [dsl_branch_component_count_get](#dsl_branch_component_count_get)
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
DslReturnType dsl_branch_new(const wchar_t* branch);
```
The constructor creates a uniquely named Branch. Construction will fail
if the name is currently in use.

**Parameters**
* `branch` - [in] unique name for the Branch to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_new('my-branch')
```

<br>

### *dsl_branch_new_many*
```C++
DslReturnType dsl_branch_new_many(const wchar_t** branches);
```
The constructor creates multiple uniquely named Branches at once. All names are checked for uniqueness, with the call returning `DSL_RESULT_BRANCH_NAME_NOT_UNIQUE` on first occurrence of a duplicate.

**Parameters**
* `pipelines` - [in] a NULL terminated array of unique names for the Branches to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation of all Branches. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_new_many(['my-branch-a', 'my-branch-b', 'my-branch-c', None])
```

<br>

---
## Destructors
### *dsl_branch_delete*
```C++
DslReturnType dsl_branch_delete(const wchar_t* branch);
```
This destructor deletes a single, uniquely named Branch. 
All components owned by the branch move to a state of `not-in-use`.

**Parameters**
* `branches` - [in] unique name for the Branch to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_delete('my-branch')
```

<br>

### *dsl_branch_delete_many*
```C++
DslReturnType dsl_branch_delete_many(const wchar_t** branches);
```
This destructor deletes multiple uniquely named Branches. Each name is checked for existence, with the function returning `DSL_RESULT_BRANCH_NAME_NOT_FOUND` on first occurrence of failure. 
All components owned by the Branches move to a state of `not-in-use`

**Parameters**
* `branches` - [in] a NULL terminated array of uniquely named Branches to delete.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_delete_many(['my-branch-a', 'my-branch-b', 'my-branch-c', None])
```

<br>

### *dsl_branch_delete_all*
```C++
DslReturnType dsl_branch_delete_all();
```
This destructor deletes all Branches currently in memory  All components owned by the pipelines move to a state of `not-in-use`

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_delete_all()
```

<br>

---
## Methods

### *dsl_branch_component_add*
```C++
DslReturnType dsl_branch_component_add(const wchar_t* branch, const wchar_t* component);
```
Adds a single named Component to a named Branch. The add service will fail if the component is currently `in-use` by any Branch. The add service will also fail if adding a `one-only` type of Component, such as a Tiled-Display, for which the Branch already has. The Component's `in-use` state will be set to `true` on successful add. 

If a Branch is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Branch in the same state.

**Parameters**
* `branch` - [in] unique name for the Branch to update.
* `component` - [in] unique name of the Component to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_csi_new('my-camera-source', 1280, 720, 30, 1)
retval = dsl_branch_new('my-branch')

retval = dsl_branch_component_add('my-branch', 'my-tiler’)
```

<br>

### *dsl_branch_component_add_many*
```C++
DslReturnType dsl_branch_component_add_many(const wchar_t* branch, const wchar_t** components);
```
Adds a list of named Component to a named Branch. The add service will fail if any of components are currently `in-use` by any Branch. The add service will fail if any of the components to add are a `one-only` type of component for which the Branch already has. All the component's `in-use` states will be set to true on successful add. 

If a Branch is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Branch in the same state.

**Parameters**
* `branch` - [in] unique name for the Branch to update.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_branch_component_add_many('my-branch', ['my-pgie', 'my-tiler', 'my-window-sink', None])
```

<br>

### *dsl_branch_componen_count_get*
```C++
uint dsl_branch_list_size(wchar_t* branch);
```
This method returns the size of the current list of Components `in-use` by the named Branch

**Parameters**
* `branch` - [in] unique name for the Branch to query.

**Returns** 
* The size of the list of Components currently in use

<br>

### *dsl_branch_component_remove*
```C++
DslReturnType dsl_branch_component_remove(const wchar_t* branch, const wchar_t* component);
```
Removes a single named Component from a named Branch. The remove service will fail if the Component is not currently `in-use` by the Branch. The Component's `in-use` state will be set to `false` on successful removal. 

If a Branch is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Branch in the same state.

**Parameters**
* `branch` - [in] unique name for the Branch to update.
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
DslReturnType dsl_branch_component_remove_many(const wchar_t* branch, const wchar_t** components);
```
Removes a list of named components from a named Branch. The remove service will fail if any of components are currently `not-in-use` by the named Branch.  All of the removed component's `in-use` state will be set to `false` on successful removal. 

If a Branch is in a `playing` or `paused` state, the service will attempt a dynamic update if possible, returning from the call with the Branch in the same state.

**Parameters**
* `branch` - [in] unique name for the Branch to update.
* `components` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_component_remove_many('my-branch', ['my-pgie', 'my-tiler', 'my-window-sink', None])
```

<br>

### *dsl_branch_component_remove_all*
```C++
DslReturnType dsl_branch_component_add_(const wchar_t* branch);
```
Removes all child components from a named Branch. The add service will fail if any of components are currently `not-in-use` by the named Branch.  All the removed component's `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `branch` - [in] unique name for the Branch to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_branch_component_remove_all('my-branch')
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
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* **Branch**
* [Component](/docs/api-component.md)

