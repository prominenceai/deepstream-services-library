# Remuxer API
The Remuxer is an aggregate component linking a Demuxer, Tees, Streammuxers, Inference Branches, and a Metamuxer, all to implement [Selective Parallel Inference](/docs/overview.md#selective-parallel-inference). Click the link for a complete overview. 

The following image shows the internal linking of the remuxer for a use case with four Source components, producing streams 0-3, and three parallel inference branches.. 
![DSL Remuxer Component](/Images/remuxer.png)

## Construction and Deletion
The DSL Remuxer Component is created by calling one of the following constructors:
* [`dsl_remuxer_new`](#dsl_remuxer_new) - creates a new Remuxer ready to add Inference Branches. 
* [`dsl_remuxer_new_branch_add_many`](#dsl_remuxer_new_branch_add_many) - creates a new Remuxer and adds a list of Inference Branches to be linked to all streams. See [`dsl_remuxer_branch_add_to`](#dsl_remuxer_branch_add_to) for adding a branch to be linked to a specific set of streams.

As with all Pipeline Components, Remuxers are deleted by calling [`dsl_component_delete`](/docs/api-component.md#dsl_component_delete), [`dsl_component_delete_many`](/docs/api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](/docs/api-component.md#dsl_component_delete_all).

## Inference Branches
An Inference branch can be defined as a single [Primary Inference Component](/docs/api-infer.md) which can be added to the Remuxer directly, or multiple Inference Components in which case an actual [Branch Component](/docs/api-branch.md) must be used. The Inference Components are added to the Branch and the Branch is added to the Remuxer. 

### Adding and Removing Inference Branches
Inference Branches can be added to a Remuxer to process all available streams by calling [`dsl_remuxer_branch_add`](#dsl_remuxer_branch_add) or [`dsl_remuxer_branch_add_many`](#dsl_remuxer_branch_add_many)... or to process a select set of streams by calling [`dsl_remuxer_branch_add_to`](#dsl_remuxer_branch_add_to)

Branches are removed from the Remuxer by calling [`dsl_remuxer_branch_remove`](#dsl_remuxer_branch_remove), [`dsl_remuxer_branch_remove_many`](#dsl_remuxer_branch_remove_many), or [`dsl_remuxer_branch_remove_all`](#dsl_remuxer_branch_remove_all)

### Branch Streammuxers
**IMPORTANT!** Although the DSL Remuxer supports both the [**OLD** NVIDIA Streammux plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html) and the [**NEW** NVIDIA Streammux plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux2.html), there is a critical NVIDIA bug in DS 6.3/6.4 that prevents use of the New Streammuxer. See [Pipelines with new nvstreammux and nvstreamdemux fail to play correctly in DS 6.3](https://forums.developer.nvidia.com/t/pipelines-with-new-nvstreammux-and-nvstreamdemux-fail-to-play-correctly-in-ds-6-3/278396/5)

## Metamuxer Configuration
The NVIDIA Metamuxer plugin requires a configuration file with an entry for each Inference branch that connects to a select set of streams. The Remuxer, given this information by the client, creates the config file under the `/tmp/` folder and provides it to the Metamuxer.  The example in the image above produces the following config file contents. 

```
[property]
[group-0]
src-ids-model-1=0
src-ids-model-2=0,1,2
```
Note that the `src-ids` for `model-3` are not listed since the third Inference Branch connects to all streams. 

## Adding and Removing a Remuxer
The relationship between Pipelines and Remuxers is one-to-one. Once added to a Pipeline, a Remuxer must be removed before it can used with another. 

Remuxer components are added to a Pipeline by calling[`dsl_pipeline_component_add`](/docs/api-pipeline.md#dsl_pipeline_component_add) or [`dsl_pipeline_component_add_many`](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [`dsl_pipeline_component_remove`](/docs/api-pipeline.md#dsl_pipeline_component_remove), [`dsl_pipeline_component_remove_many`](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [`dsl_pipeline_component_remove_all`](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

## Adding/Removing Pad-Probe-handlers
Multiple sink (input) and/or source (output) [Pad-Probe Handlers](/docs/api-pph.md) can be added to a Remuxer by calling [`dsl_remuxer_pph_add`](#dsl_remuxer_pph_add) and removed with [`dsl_remuxer_pph_remove`](#dsl_remuxer_pph_remove).

## Remuxer API
**Constructors**
* [`dsl_remuxer_new`](#dsl_remuxer_new)
* [`dsl_remuxer_new_branch_add_many`](#dsl_remuxer_new_branch_add_many)

**Remuxer Methods (common)**
* [`dsl_remuxer_branch_add`](#dsl_remuxer_branch_add)
* [`dsl_remuxer_branch_add_to`](#dsl_remuxer_branch_add_to)
* [`dsl_remuxer_branch_add_many`](#dsl_remuxer_branch_add_many)
* [`dsl_remuxer_branch_remove`](#dsl_remuxer_branch_remove)
* [`dsl_remuxer_branch_remove_many`](#dsl_remuxer_branch_remove_many)
* [`dsl_remuxer_branch_remove_all`](#dsl_remuxer_branch_remove_all)
* [`dsl_remuxer_pph_add`](#dsl_remuxer_pph_add)
* [`dsl_remuxer_pph_remove`](#dsl_remuxer_pph_remove)

**Remuxer Methods (old Streammuxer)**
* [`dsl_remuxer_batch_properties_get`](#dsl_remuxer_batch_properties_get)
* [`dsl_remuxer_batch_properties_set`](#dsl_remuxer_batch_properties_set)
* [`dsl_remuxer_dimensions_get`](#dsl_remuxer_dimensions_get)
* [`dsl_remuxer_dimensions_set`](#dsl_remuxer_dimensions_set)

**Remuxer Methods (new Streammuxer)**
* [`dsl_remuxer_branch_config_file_get`](#dsl_remuxer_branch_config_file_get)
* [`dsl_remuxer_branch_config_file_set`](#dsl_remuxer_branch_config_file_set)
* [`dsl_remuxer_batch_size_get`](#dsl_remuxer_batch_size_get)
* [`dsl_remuxer_batch_size_set`](#dsl_remuxer_batch_size_set)

## Return Values
The following return codes are used by the Tee API
```C++
#define DSL_RESULT_REMUXER_RESULT                                   0x00C00000
#define DSL_RESULT_REMUXER_NAME_NOT_UNIQUE                          0x00C00001
#define DSL_RESULT_REMUXER_NAME_NOT_FOUND                           0x00C00002
#define DSL_RESULT_REMUXER_NAME_BAD_FORMAT                          0x00C00003
#define DSL_RESULT_REMUXER_THREW_EXCEPTION                          0x00C00004
#define DSL_RESULT_REMUXER_SET_FAILED                               0x00C00005
#define DSL_RESULT_REMUXER_BRANCH_IS_NOT_BRANCH                     0x00C00006
#define DSL_RESULT_REMUXER_BRANCH_IS_NOT_CHILD                      0x00C00007
#define DSL_RESULT_REMUXER_BRANCH_ADD_FAILED                        0x00C00008
#define DSL_RESULT_REMUXER_BRANCH_MOVE_FAILED                       0x00C00009
#define DSL_RESULT_REMUXER_BRANCH_REMOVE_FAILED                     0x00C0000A
#define DSL_RESULT_REMUXER_HANDLER_ADD_FAILED                       0x00C0000B
#define DSL_RESULT_REMUXER_HANDLER_REMOVE_FAILED                    0x00C0000C
#define DSL_RESULT_REMUXER_COMPONENT_IS_NOT_REMUXER                 0x00C0000D
```

## Remuxer Internal Streammuxer Constant Values
```C
#define DSL_STREAMMUX_DEFAULT_WIDTH                                 DSL_1K_HD_WIDTH
#define DSL_STREAMMUX_DEFAULT_HEIGHT                                DSL_1K_HD_HEIGHT
```

## Constructors

### *dsl_remuxer_new*
```C++
DslReturnType dsl_remuxer_new(const wchar_t* name);
```
The constructor creates a uniquely named Remuxer. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Remuxer to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_remuxer_new('my-remuxer')
```

<br>

### *dsl_remuxer_new_branch_add_many*
```C++
DslReturnType dsl_remuxer_new_branch_add_many(const wchar_t* name, const wchar_t** branches)
```
The constructor creates a uniquely named Remuxer and adds a list of Branches to it. Construction will fail if the name is currently in use. 

IMPORTANT! All branches will be linked to all streams. To add a Branch to a select set of streams, see [dsl_remuxer_branch_add_to](#dsl_remuxer_branch_add_to)

**Parameters**
* `name` - [in] unique name for the Remuxer to create.
* `branches` [in] Null terminated list of unique branch names to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_remuxer_new_branch_add_many('my-remuxer', 
   ['my-branch-1', 'my-branch-2', 'my-branch-3', None])
```

<br>

## Methods
### *dsl_remuxer_branch_add*
```C++
DslReturnType dsl_remuxer_branch_add(const wchar_t* name, const wchar_t* branch);
```
This service adds a single branch to a named Remuxer. The add service will fail if the branch is currently `in-use`. The branch's `in-use` state will be set to `true` on successful add. 

**Parameters**
* `name` - [in] unique name for the Remuxer to update.
* `branch` - [in] unique name of the Branch to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_branch_add('my-remuxer', 'my-branch’)
```

<br>

### *dsl_remuxer_branch_add_to*
```C++
DslReturnType dsl_remuxer_branch_add_to(const wchar_t* name, 
    const wchar_t* branch, uint* stream_ids, uint num_stream_ids);
```
This service adds a single [Branch component](/docs/api-branch.md) to a named Remuxer Tee. The Branch will be connected/linked to a specified set of stream-ids.

**IMPORTANT!** This service will fail if called at runtime (i.e. while the Pipeline is playing). Adding Branches dynamically is not supported at this time.

**Parameters**
* `name` - [in] unique name of the Remuxer to update.
* `branch` - [in] unique name of the Branch to add. This may be a [Primary Inference component](/docs/api-infer.md) or [Branch component](/docs/api-branch.md).
* `stream_ids` - [in] array of 0-based unique stream-ids connect this Branch to.
* `num_stream_ids` - [in] - number of stream-ids in the `stream_ids` array.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
stream_ids = [0,1,4,6]
retval = dsl_remuxer_branch_add_to('my-remuxer', 'my-branch’,
    stream_ids, len(stream_ids))
```

<br>

### *dsl_remuxer_branch_add_many*
```C++
DslReturnType dsl_remuxer_branch_add_many(const wchar_t* name, const wchar_t** branches);
```
This service adds a NULL terminated list of named Branches to a named Remuxer.  Each of the branches `in-use` state will be set to true on successful add. 

**Parameters**
* `name` - [in] unique name for the Remuxer to update.
* `branches` - [in] a NULL terminated array of uniquely named Components to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_remuxer_branch_add_many('my-remuxer', 
   ['my-branch-1', 'my-branch-2', None])
```

<br>

### *dsl_remuxer_branch_remove*
```C++
DslReturnType dsl_remuxer_branch_remove(const wchar_t* name, const wchar_t* branch);
```
This service removes a single named Branch from a Remuxer. The remove service will fail if the Branch is not currently `in-use` by the Remuxer. The branch's `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Remuxer to update.
* `branch` - [in] unique name of the Branch to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_branch_remove('my-remuxer', 'my-branch')
```
<br>

### *dsl_remuxer_branch_remove_many*
```C++
DslReturnType dsl_remuxer_branch_remove_many(const wchar_t* name, const wchar_t** branches);
```
This service removes a list of named Branches from a named Remuxer. The remove service will fail if any of the branches are currently `not-in-use` by the named Branch.  All of the removed branches' `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Remuxer to update.
* `components` - [in] a NULL terminated array of uniquely named Branches to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_branch_remove_many('my-remuxer',
    ['my-branch-1', 'my-branch-2', None])
```

<br>

### *dsl_remuxer_branch_remove_all*
```C++
DslReturnType dsl_remuxer_branch_remove_all(const wchar_t* name);
```
This service removes all child branches from a Remuxer. All of the removed branches' `in-use` state will be set to `false` on successful removal. 

**Parameters**
* `name` - [in] unique name for the Remuxer to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_branch_remove_all('my-remuxer')
```

<br>

## Remuxer Methods (new Streammuxer)
### *dsl_remuxer_branch_config_file_get*
```C++
DslReturnType dsl_remuxer_branch_config_file_get(const wchar_t* name, 
    const wchar_t* branch, const wchar_t** config_file);
```
This service returns the current Streammuxer config-file in use by a named Remuxer Branch. To use this service, export USE_NEW_NVSTREAMMUX=yes.

**IMPORTANT!** The named Branch must already be added to the named Remuxer or this service will fail.

**Parameters**
* `name` - [in] unique name of the Remuxer to query.
* `branch` - [in] unique name of the Remuxer Branch to query.
* `config_file` - [out] path to the current Streammux config-file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_remuxer_branch_config_file_get('my-remuxer', 'my-branch-a')
```
<br>

### *dsl_remuxer_branch_config_file_set*
```C++
DslReturnType dsl_remuxer_branch_config_file_set(const wchar_t* name, 
    const wchar_t* branch, const wchar_t* config_file);
```
This service sets the Streammuxer config-file for a named Remuxer Branch to use. To use this service, export USE_NEW_NVSTREAMMUX=yes.

**IMPORTANT!** The named Branch must already be added to the named Remuxer or this service will fail.

**Parameters**
* `name` - [in] unique name of the Remuxer to update.
* `branch` - [in] unique name of the Remuxer Branch to update.
* `config_file` - [in] absolute or relative path to the new Streammux config-file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_branch_config_file_set('my-remuxer',
    'my-branch-a', './streammux_a_config.txt)
```
<br>

### *dsl_remuxer_batch_size_get*
```C++
DslReturnType dsl_remuxer_batch_size_get(const wchar_t* name, 
    uint* batch_size);
```
This service returns the current `batch_size` setting for the named Remuxer. The `batch_size` is used by each internal Streammux plugin for each added Branch that connects to all streams. 
Branches that connect to a select set of stream-ids will set their `batch-size` to the number of streams selected.  To use this service, export USE_NEW_NVSTREAMMUX=yes.

**Note:** Unless explicitly set with a call to [dsl_tee_remuxer_batch_size_set](#dsl_tee_remuxer_batch_size_set), the Remuxer will use the upstream batch-size when the Pipeline is linked and played. 

**IMPORTANT!** If adding/removing Sources dynamically at runtime, you must set the batch-size to the maximum number of upstream Sources that can be added.

**Parameters**
* `name` - [in] unique name for the Remuxer to query.
* `batch_size` - [out] the current batch size set for the Remuxer. Default = 0 until runtime or unless explicitly set.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, batch_size = dsl_remuxer_batch_size_get('my-remuxer')
```

<br>

### *dsl_remuxer_batch_size_set*
```C++
DslReturnType dsl_remuxer_batch_size_set(const wchar_t* name, 
    uint batch_size);
```
This service sets the `batch_size` for the named Remxuer to use.  The `batch_size` is used by each internal Streammux plugin for each added Branch that connects to all streams. 
Branches that connect to a select set of stream-ids will set their `batch-size` to the number of streams selected. To use this service, export USE_NEW_NVSTREAMMUX=yes.

**Note:** Unless explicitly set with this service, the Remuxer will use the upstream batch-size when the Pipeline is linked and played. 

**IMPORTANT!** If adding/removing Sources dynamically at runtime, you must set the batch-size to the maximum number of upstream Sources that can be added.

**Parameters**
* `name` - [in] unique name for the Remuxer to update.
* `batch_size` - [in] the new batch size to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_batch_size_set('my-remuxer', batch_size)
```

<br>

## Remuxer Methods (old Streammuxer)
### *dsl_remuxer_batch_properties_get*
```C++
DslReturnType dsl_remuxer_batch_properties_get(const wchar_t* name, 
    uint* batch_size, int* batch_timeout);
```
This service returns the current `batch_size` and `batch_timeout` for the named Remuxer. The `batch_size` is used by all internal Streammux plugins connecting to Branches that are to connect to all streams. The `batch_timeout` is used by all internal Streammux plugins allocated. 
**Note:** the Remuxer's parent Pipeline or Branch will set the `batch_size` to current number of upstream added Sources at runtime, and the `batch_timeout` to -1 (disabled), if not explicitly set. A Branch connecting to a specific set of stream-ids will set the `batch-size` to the number of streams to connect to.

**Parameters**
* `name` - [in] unique name for the Remuxer to query.
* `batch_size` - [out] the current batch size set for the Remuxer. Default = 0 until runtime or unless explicitly set.
* `batch_timeout` - [out] timeout in milliseconds before a batch meta push is forced. Set to -1 by default.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, batch_size, batch_timeout = dsl_remuxer_batch_properties_get('my-remuxer')
```

<br>

### *dsl_remuxer_batch_properties_set*
```C++
DslReturnType dsl_remuxer_batch_properties_set(const wchar_t* name, 
    uint batch_size, int batch_timeout);
```
This service sets the `batch_size` and `batch_timeout` for the named Remxuer to use.  The `batch_size` is used by all internal Streammux plugins when connecting to Branches that are to connect to all streams. The `batch_timeout` is used by all internal Streammux plugins. 
**Note:** the Remuxer's parent Pipeline or Branch will set the `batch_size` to current number of upstream added Sources at runtime, and the `batch_timeout` to -1 (disabled), if not explicitly set. A Branch connecting to a specific set of stream-ids will set the `batch-size` to the number of streams to connect to.

**Parameters**
* `name` - [in] unique name for the Remuxer to update.
* `batch_size` - [in] the new batch size to use.
* `batch_timeout` - [in] the new timeout in milliseconds before a batch meta push is forced.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_batch_properties_set('my-remuxer',
    batch_size, batch_timeout)
```

<br>

### *dsl_remuxer_dimensions_get*
```C++
DslReturnType dsl_remuxer_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service returns the current output dimensions for all internal Streammuxer plugins for the uniquely named Remuxer. The [default dimensions](#remuxer-internal-streammuxer-constant-values)  are assigned during Remuxer creation. 

**Parameters**
* `name` - [in] unique name of the Remuxer to query.
* `width` - [out] width of all internal Streammuxer's output in pixels.
* `height` - [out] height of all internal Streammuxer's output in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_remuxer_dimensions_get('my-remuxer')
```

<br>

### *dsl_remuxer_dimensions_set*
```C++
DslReturnType dsl_remuxer_dimensions_set(const wchar_t* name, 
    uint width, uint height);
```
This service sets the output dimensions for all internal Streammux plugins for the uniquely named Remuxer. The dimensions cannot be updated while the Pipeline is in a state of `PAUSED` or `PLAYING`.

**Parameters**
* `name` - [in] unique name of the Remuxer to update.
* `width` - [in] new width for all internal Streammuxer's output in pixels.
* `height` - [in] new height for all internal Streammuxer's output in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_remuxer_dimensions_set('my-remuxer', 1280, 720)
```

<br>

### *dsl_remuxer_pph_add*
```C++
DslReturnType dsl_remuxer_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named Remuxer.

**Parameters**
* `name` - [in] unique name of the Remuxer to update.
* `handler` - [in] unique name of Pad Probe Handler to add
* `pad` - [in] to which of the two pads to add the handler: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_remuxer_pph_add('my-remuxer', 'my-pph-handler', `DSL_PAD_SINK`)
```

<br>

### *dsl_remuxer_pph_remove*
```C++
DslReturnType dsl_remuxer_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named Remuxer. The service will fail if the named handler is not owned by the Remuxer

**Parameters**
* `name` - [in] unique name of the Remuxer to update.
* `handler` - [in] unique name of Pad Probe Handler to remove
* `pad` - [in] to which of the two pads to remove the handler from: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_remuxer_pph_remove('my-tiler', 'my-pph-handler', `DSL_PAD_SINK`)
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
* [Demuxer and Splitter Tees](/docs/api-tee.md)
* **Remuxer**
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
