# Demuxer and Splitter - Tee API

#### Tee Construction and Destruction
Demuxers and Splitters are created using a type specific constructor,  [dsl_tee_demuxer_new](#dsl_tee_demuxer_new) and [dsl_tee_splitter_new](#dsl_tee_splitter_new) respectively

Pipelines use Tees to create Branches and Branches can use Tees to create more Branches. Therefore, Tees are added to Pipelines and Branches, while Branches are added to Tees.

The relationship between Pipline/Branch and Tee is one to one with the Tee becoming the end component. The relationship between Tees and Branches is one-to-many. Once added to a Pipeline or Branch, a Tee must be removed before it can used with another. 

Tees and Branches are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

#### Adding and removing Branches from a Tee
Branches are added to a Tee by calling [dsl_tee_branch_add](api-branch.md#dsl_tee_branch_add) or [dsl_tee_branch_add_many](api-branch.md#dsl_tee_branch_add_many) and removed with [dsl_tee_branch_remove](api-branch.md#dsl_tee_branch_remove), [dsl_tee_branch_remove_many](api-branch.md#dsl_tee_branch_remove_many), or [dsl_tee_branch_remove_all](api-branch.md#dsl_tee_branch_remove_all).

## Tee API
**Constructors**
* [dsl_tee_demuxer_new](#dsl_tee_demuxer_new)
* [dsl_tee_demuxer_new_branch_add_many](#dsl_tee_demuxer_new_branch_add_many)
* [dsl_tee_splitter_new](#dsl_tee_splitter_new) 
* [dsl_tee_splitter_new_branch_add_many](#dsl_tee_demuxer_new_branch_add_many)

**Methods**
* [dsl_tee_branch_add](#dsl_tee_branch_add)
* [dsl_tee_branch_add_many](#dsl_tee_branch_add_many)
* [dsl_tee_branch_remove](#dsl_tee_branch_remove)
* [dsl_tee_branch_remove_many](#dsl_tee_branch_remove_many)
* [dsl_tee_branch_remove_all](#dsl_tee_branch_remove_all).
* [dsl_tee_pph_add](#dsl_tee_pph_add).
* [dsl_tee_pph_remove](#dsl_tee_pph_remove).

## Return Values
The following return codes are used by the Tiler API
```C++
#define DSL_RESULT_TEE_RESULT                                       0x000A0000
#define DSL_RESULT_TEE_NAME_NOT_UNIQUE                              0x000A0001
#define DSL_RESULT_TEE_NAME_NOT_FOUND                               0x000A0002
#define DSL_RESULT_TEE_NAME_BAD_FORMAT                              0x000A0003
#define DSL_RESULT_TEE_THREW_EXCEPTION                              0x000A0004
#define DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD                          0x000A0005
#define DSL_RESULT_TEE_BRANCH_ADD_FAILED                            0x000A0006
#define DSL_RESULT_TEE_BRANCH_REMOVE_FAILED                         0x000A0007
#define DSL_RESULT_TEE_HANDLER_ADD_FAILED                           0x000A0008
#define DSL_RESULT_TEE_HANDLER_REMOVE_FAILED                        0x000A0009
#define DSL_RESULT_TEE_COMPONENT_IS_NOT_TEE                         0x000A000A
```

## Constructors

### *dsl_tee_demuxer_new*
```C++
DslReturnType dsl_tee_demuxer_new(const wchar_t* name);
```
The constructor creates a uniquely named Demuxer Tee. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Demuxer to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tee_demuxer_new('my-demuxer')
```

<br>

### *dsl_tee_demuxer_new_branch_add_many*
```C++
DslReturnType dsl_tee_demuxer_new_branch_add_many(const wchar_t* name, const wchar_t** branches);
```
The constructor creates a uniquely named Demuxer Tee and adds a list of branches to it. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Demuxer to create.
* `branches` [in] Null terminated listed of unique branch names to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tee_demuxer_new_branch_add_many('my-demuxer', 
    ['my-branch-1', 'my-branch-2', None])
```

<br>

### *dsl_tee_splitter_new*
```C++
DslReturnType dsl_tee_splitter_new(const wchar_t* name);
```
The constructor creates a uniquely named Splitter Tee. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Splitter to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tee_splitter_new('my-demuxer')
```

<br>

### *dsl_tee_splitter_new_branch_add_many*
```C++
DslReturnType dsl_tee_splitter_new_branch_add_many(const wchar_t* name, const wchar_t** branches)
```
The constructor creates a uniquely named Splitter Tee and adds a list of Branches to it. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Splitter to create.
* `branches` [in] Null terminated listed of unique branch names to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tee_splitter_new_branch_add_many('my-demuxer', 
   ['my-branch-1', 'my-branch-2', None])
```

<br>

## Methods
### *dsl_tee_branch_add*
```C++
DslReturnType dsl_tee_branch_add(const wchar_t* name, const wchar_t* branch);
```
This service adds a single branch to a named Spliter or Demuxer Tee. The add service will fail if the branch is currently `in-use`. The branches `in-use` state will be set to `true` on successful add. 

**Parameters**
* `branch` - [in] unique name for the Branch to update.
* `component` - [in] unique name of the Component to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tee_branch_add('my-splitter', 'my-branch’)
```

<br>

### *dsl_tee_branch_add_many*
```C++
DslReturnType dsl_tee_branch_add_many(const wchar_t* name, const wchar_t** branches);
```
This service adds a list of named Branches to a Null termainted list of branches to a named Splitter or Demuxer Tee.  Each of the branches `in-use` state will be set to true on successful add. 

**Parameters**
* `name` - [in] unique name for the Spliter to update.
* `branches` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_tee_branch_add_many('my-splitter', 
   ['my-branch-1', 'my-branch-2', None])
```

<br>

### *dsl_tee_branch_remove*
```C++
DslReturnType dsl_tee_branch_remove(const wchar_t* name, const wchar_t* branch);
```
This service removes a single named Branch from a Demuxer or Splitter Tee. The remove service will fail if the Branch is not currently `in-use` by the Tee. The branches' `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Demuxer or Spltter Tee to update.
* `branch` - [in] unique name of the Branch to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tee_branch_remove('my-splitter', 'my-branch')
```
<br>

### *dsl_tee_branch_remove_many*
```C++
DslReturnType dsl_tee_branch_remove_many(const wchar_t* name, const wchar_t** branches);
```
This service removes a list of named components from a named Branch. The remove service will fail if any of branches are currently `not-in-use` by the named Branch.  All of the removed branches' `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Demuxer or Splitter Tee to update.
* `components` - [in] a NULL terminated array of uniquely named Brances to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tee_branch_remove_many('my-splitter',
    ['my-branch-1', 'my-branch-2', None])
```

<br>

### *dsl_tee_branch_remove_all*
```C++
DslReturnType dsl_tee_branch_remove_all(const wchar_t* name);
```
This service removes all child branches from a named Demuxer or Splitter Tee. All of the removed branches' `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Branch to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tee_branch_remove_all('my-splitter')
```


### *dsl_tee_pph_add*
```C++
DslReturnType dsl_tee_pph_add(const wchar_t* name, const wchar_t* handler);
```
This services adds a named Pad-Probe-Handler to the sink pad (only) of the Named Tee component. The PPH will be invoked on every buffer-ready event for the sink pad. More than one PPH can be added to a single Tee Component.

**Parameters**
 * `name` [in] unique name of the Tee to update
 * `handler` [in] uninque name of the PPH to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tee_pph_add('my-demuxer-tee', 'my-meter-pph')
```

<br>

### *dsl_tee_pph_remove*
```C++
DslReturnType dsl_tee_pph_remove(const wchar_t* name, const wchar_t* handler);
```
This services removes a named Pad-Probe-Handler from a named Tee. The services will fail if the handler is not a child of the Tee

**Parameters**
 * `name` [in] unique name of the Tee to update.
 * `handler` [in] unique name of the Pad probe handler to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tee_pph_remove('my-demuxer-tee', 'my-meter-pph')
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
* **Demuxer and Splitter**
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
