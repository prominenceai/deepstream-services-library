# Remuxer API

Built with a Demuxer, multiple Streammuxers, and a Metamux The Remuxer Tee splits the batched input stream into downstream branches, each with their own unique batched metatdata for parallel inference.  

Remuxing a batched stream is performed as follows:
1. The Demuxer plugin is used to demux the incoming batched stream into individual streams/source-pads.
2. GStreamer tee plugins are connected to the source-pads splitting each single stream into multiple single streams, as required for each downstream Branch.
3. Each added Branch is connected upstream to an NVIDIA [Gst-nvstreammux plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html) 
4. Each Streammuxer is then connected upstream to some or all of the single stream Tees, as specified by the client.

DSL supports both the [**OLD** NVIDIA Streammux pluging](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html) and the [**NEW** NVIDIA Streammux plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux2.html) 

## Remuxer API
**Constructors**
* [`dsl_remuxer_new`](#dsl_remuxer_new)
* [`dsl_remuxer_new_branch_add_many`](#dsl_remuxer_new_branch_add_many)

**Remuxer Methods (common)**
* [`dsl_remuxer_branch_add_to`](#dsl_remuxer_branch_add_to)
* [`dsl_remuxer_branch_add`](#dsl_remuxer_branch_add)
* [`dsl_remuxer_branch_add_many`](#dsl_remuxer_branch_add_many)
* [`dsl_remuxer_branch_remove`](#dsl_remuxer_branch_remove)
* [`dsl_remuxer_branch_remove_many`](#dsl_remuxer_branch_remove_many)
* [`dsl_remuxer_branch_remove_all`](#dsl_remuxer_branch_remove_all)
* [`dsl_remuxer_pph_add`](#dsl_remuxer_pph_add)
* [`dsl_remuxer_pph_remove`](#dsl_remuxer_pph_remove)

**Remuxer Methods (old Streammuxer)**
* [`dsl_remuxer_batch_properties_get`](#dsl__remuxer_batch_properties_get)
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

## Constructors

### *dsl_tee_remuxer_new*
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
* `name` - [in] unique name for the Demuxer or Splitter remuxer to update.
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


