# Preprocessor API Reference
The DeepStream Services Library (DSL) provides services for NVIDIA's Gst-nvdsprerocessor plugin. From the [NVIDIA DeepStream Developer's Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html),

 *"The Gst-nvdspreprocess plugin is a customizable plugin which provides a custom library interface for preprocessing on input streams. Each stream can have its own preprocessing requirements. (e.g., per stream ROIs - Region of Interests processing)."* [Read more...](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspreprocess.html#gst-nvdspreprocess-alpha).

**Note:​** The Gst-nvdsprerocessor plugin released with DeepStream 6.0 is in an "alpha" state and only the Primary GST Inference component can process input-tensor-meta from the Preprocessor.
 
**IMPORTANT!** Raw tensor from the scaled & converted ROIs are passed to the downstream Primary GST Inference Engine (PGIE) via user metadata. You must enable the PGIE's `input-tensor-meta` setting by calling [dsl_infer_gie_tensor_meta_settings_set](/docs/api-infer.md#dsl_infer_gie_tensor_meta_settings_set) when adding a Preprocessor to a Pipeline. Refer to the [Inference API Reference](/docs/api-infer.md)

### Preprocessor Construction and Destruction
The constructor [dsl_preproc_new](#dsl_preproc_new) is used to create a Preprocessor with input parameters for the config file and enabled setting. Once created, the Preprocessor's configuration file and enabled setting can be updated. Preprocessors are deleted by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)

### Preprocessor Configuration
Preprocessor components require a configuration file when constructed. Once created, clients can query the Preprocessor for the Config File in-use by calling  [dsl_preproc_config_file_get](#dsl_preproc_config_file_get) or change the Preprocessor's configuration by calling [dsl_preproc_config_file_set](#dsl_preproc_config_file_get).

### Adding and Removing
A single Preprocessor can be added to Pipeline trunk or individual Branch. A Preprocessor is added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

A similar set of Services are used when adding/removing a Preprocess to/from a branch: [dsl_branch_component_add](api-branch.md#dsl_branch_component_add), [dsl_branch_component_add_many](/docs/api-branch.md#dsl_branch_component_add_many), [dsl_branch_component_remove](/docs/api-branch.md#dsl_branch_component_remove), [dsl_branch_component_remove_many](/docs/api-branch.md#dsl_branch_component_remove_many), and [dsl_branch_component_remove_all](/docs/api-branch.md#dsl_branch_component_remove_all).

Once added to a Pipeline or Branch, a Preprocessor must be removed before it can be used with another.

---
## Preprocessor API
**Constructors:**
* [dsl_preproc_new](#dsl_preproc_new)

**Methods:**
* [dsl_preproc_config_file_get](#dsl_preproc_config_file_get)
* [dsl_preproc_config_file_set](#dsl_preproc_config_file_get)
* [dsl_preproc_enabled_get](#dsl_preproc_enabled_get)
* [dsl_preproc_enabled_set](#dsl_preproc_enabled_set)
* [dsl_preproc_unique_id_get](#dsl_preproc_unique_id_get)

## Return Values
The following return codes are used by the On-Screen Display API
```C++
#define DSL_RESULT_PREPROC_RESULT                                   0x00B00000
#define DSL_RESULT_PREPROC_NAME_NOT_UNIQUE                          0x00B00001
#define DSL_RESULT_PREPROC_NAME_NOT_FOUND                           0x00B00002
#define DSL_RESULT_PREPROC_CONFIG_FILE_NOT_FOUND                    0x00060003
#define DSL_RESULT_PREPROC_THREW_EXCEPTION                          0x00B00004
#define DSL_RESULT_PREPROC_IN_USE                                   0x00B00005
#define DSL_RESULT_PREPROC_SET_FAILED                               0x00B00006
#define DSL_RESULT_PREPROC_IS_NOT_PREPROC                           0x00B00007
```

## Constructors
### *dsl_preproc_new*
```c++
DslReturnType dsl_preproc_new(const wchar_t* name, const wchar_t* config_file);
```
The constructor creates a uniquely named Preprocessor. Construction will fail if the name is currently in use. The Preprocessor is enabled by default. It can be disabled by calling [dsl_preproc_enabled_set](#dsl_preproc_enabled_set).

**Parameters**
* `name` - [in] unique name for the Preprocessor to create.
* `config_file` - [in] absolute or relative path to the Preprocessor configuration file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_preproc_new('my-preprocessor', './my-config-file.txt')
```

<br>

---

## Methods
### *dsl_preproc_config_file_get*
```c++
DslReturnType dsl_preproc_config_file_get(const wchar_t* name,
    const wchar_t** config_file);
```
This service returns the current configuration file in use by a named Preprocessor.

**Parameters**
* `name` - [in] unique name of the Preprocessor to query.
* `config_file` - [out] absolute or relative path to the configuration file in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, config_file = dsl_preproc_config_file_get('my-preprocessor')
```

<br>

### *dsl_preproc_config_file_set*
```c++
DslReturnType dsl_preproc_config_file_set(const wchar_t* name,
    const wchar_t* config_file);
```
This service sets the current configuration file for the named Preprocessor to use.

**Parameters**
* `name` - [in] unique name of the Preprocessor to update.
* `config_file` - [in] absolute or relative path to the new Preprocessor configuration file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_preproc_config_file_set('my-on-screen-display', './my-new-config-file.txt)
```

<br>

### *dsl_preproc_enabled_get*
```c++
DslReturnType dsl_preproc_enabled_get(const wchar_t* name,
    boolean* enabled);
```
This service returns the current enabled setting for the named Preprocessor.

**Parameters**
* `name` - [in] unique name of the Preprocessor to query.
* `enable` - [out] true if preprocessing in enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, enabled = dsl_preproc_enabled_get('my-preprocessor')
```

<br>

### *dsl_preproc_enabled_set*
```c++
DslReturnType dsl_preproc_enabled_set(const wchar_t* name,
    boolean enabled);
```
This service sets the enabled setting for the named Preprocessor.

**Parameters**
* `name` - [in] unique name of the Preprocessor to update.
* `enabled` - [in] set to `True` to enabled preprocessing, `False` to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_preproc_enabled_set('my-preprocessor', False)
```

<br>

### *dsl_preproc_enabled_get*
```c++
DslReturnType dsl_preproc_unique_id_get(const wchar_t* name,
    uint* id);
```

This service the unique Id assigned to the named Preprocessor when created. The Id is used to identify metadata generated by the Preprocessor. Id's start at 0 and are incremented with each new Preprocessor created. Id's will be reused if the Preprocessor is deleted.

**Parameters**
* `name` - [in] unique name of the Preprocessor to query.
* `id` - [out] the unique id assigned to the Preprocessor when created.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, unique_id = dsl_preproc_unique_id_get('my-preprocessor')
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* **Preprocessor**
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
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
