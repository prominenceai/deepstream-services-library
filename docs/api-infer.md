# Primary and Secondary Inference API Reference
The DeepStream Services Library (DSL) provides services for Nvidia's two Inference Plugins; the GST Inference Engine (GIE) and the Triton Inference Server (TIS). See Nvidia's website for complete documentation on each.

Pipelines can have at most one Primary GIE or TIS with any number of corresponding Secondary GIEs or TISs (only limited by hardware). Pipelines cannot be created with a mix of GIEs and TISs. Pipelines that have secondary GIEs/TISs but no Primary GIE/TIS will fail to Link and Play. Secondary GIEs/TISs can `infer-on` both Primary and Secondary GIEs/TISs creating multiple levels of inference.

### Construction and Destruction
Primary GIEs and TISs are constructed by calling [dsl_infer_gie_primary_new](#dsl_infer_gie_primary_new) and [dsl_infer_tis_primary_new](#dsl_infer_tis_primary_new) respectively. Secondary GIEs and TISs are created by calling [dsl_infer_gie_secondary_new](#dsl_gie_secondary_new) and [dsl_infer_tis_secondary_new](#dsl_infer_tis_secondary_new) respectively. As with all components, Primary and Secondary GIEs/TISs must be uniquely named from all other components created. All GIEs and TIEs are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all).

### Inference Configuration
Both GIEs and TIEs require a Primary or Secondary **Inference Configuration File**. Once created, clients can query both Primary and Secondary GIEs/TIEs for their Config File in-use by calling [dsl_infer_config_file_get](#dsl_infer_config_file_get) or change the GIE/TIS's configuration by calling [dsl_infer_config_file_set](#dsl_infer_config_file_set).

### Model Engine Files
GIEs support the specification of a pre-built **Model Engine File**, or one can allow the Plugin to create the model engine based on the configuration. The file in use can be queried by calling [dsl_infer_gie_model_engine_file_get](#dsl_infer_gie_model_engine_file_get) or changed with [dsl_infer_gie_model_engine_file_set](#dsl_infer_gie_model_engine_file_set).

### Unique Id
**IMPORTANT!** DSL explicitly assigns each GIE or TISs a unique component id overriding the (optional) parameter in the inference config file. The unique component id is derived from the first available unused id starting with 1, meaning the first component will be assigned id 1, the second id 2 and so on. The id will be reused if the inference component is deleted and a new one created. The value assigned to the GIE or TIS can be queried by calling [dsl_infer_unique_id_get](#dsl_infer_unique_id_get). All Object metadata structures created by the named GIE/TIE will include a `unique_component_id` field assigned with this id.

### Inference Interval
**IMPORTANT!** DSL sets the inference interval with the input parameter provided on construction overriding the (optional) parameter in the inference config file. The interval for inferencing -- or the number of frames to skip between inferencing -- is set as an unsigned integer with `0 = every frame`, `1 = every other frame`, `2 = every 3rd frame`, etc., when created.  The current interval in-use by any GIE/TIS can be queried by calling [dsl_infer_interval_get](#dsl_infer_interval_get), and changed by calling [dsl_infer_interval_set](#dsl_infer_interval_set).

### Inference Batch Size
**IMPORTANT!** DSL sets the inference batch size overriding the parameter in the inference config file. The batch size for each GIE/TIS can be set explicitly by calling [dsl_infer_batch_size_set](#dsl_infer_batch_size_set). If not set (0-default), the Pipeline will set the batch-size to the same value as the Streammux batch-size which - by default - is derived from the number of sources when the Pipeline is called to play. The Streammux batch-size can be set (overridden) by calling [dsl_pipeline_streammux_batch_properties_set](/docs/api-pipeline.md#dsl_pipeline_streammux_batch_properties_set).

### Adding and Removing
GIEs/TISs are added to a Pipeline by calling [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add) and [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed by calling [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove) and [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many).

A similar set of Services are used when adding/removing a GIE/TIS to/from a branch: [dsl_branch_component_add](api-branch.md#dsl_branch_component_add), [dsl_branch_component_add_many](/docs/api-branch.md#dsl_branch_component_add_many), [dsl_branch_component_remove](/docs/api-branch.md#dsl_branch_component_remove), [dsl_branch_component_remove_many](/docs/api-branch.md#dsl_branch_component_remove_many), and [dsl_branch_component_remove_all](/docs/api-branch.md#dsl_branch_component_remove_all).

Primary and Secondary GIEs/TISs are deleted by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_delete_all](/docs/overview.md#dsl_delete_all).

## Primary and Secondary Inference API
**Constructors**
* [dsl_infer_gie_primary_new](#dsl_infer_gie_primary_new)
* [dsl_infer_gie_secondary_new](#dsl_infer_gie_secondary_new)
* [dsl_infer_tis_primary_new](#dsl_infer_tis_primary_new)
* [dsl_infer_tis_secondary_new](#dsl_infer_tis_secondary_new)

**Methods**
* [dsl_infer_batch_size_get](#dsl_infer_batch_size_get)
* [dsl_infer_batch_size_set](#dsl_infer_batch_size_set)
* [dsl_infer_unique_id_get](#dsl_infer_unique_id_get)
* [dsl_infer_gie_model_engine_file_get](#dsl_infer_gie_model_engine_file_get)
* [dsl_infer_gie_model_engine_file_set](#dsl_infer_gie_model_engine_file_set)
* [dsl_infer_gie_tensor_meta_settings_get](#dsl_infer_gie_tensor_meta_settings_get)
* [dsl_infer_gie_tensor_meta_settings_set](#dsl_infer_gie_tensor_meta_settings_set)
* [dsl_infer_config_file_get](#dsl_infer_config_file_get)
* [dsl_infer_config_file_set](#dsl_infer_config_file_set)
* [dsl_infer_interval_get](#dsl_infer_interval_get)
* [dsl_infer_interval_set](#dsl_infer_interval_set)
* [dsl_infer_primary_pph_add](#dsl_infer_primary_pph_add)
* [dsl_infer_primary_pph_remove](#dsl_infer_primary_pph_remove)

---
## Return Values
The following return codes are used by the Inference API
<br>

```C
#define DSL_RESULT_INFER_RESULT                                     0x00060000
#define DSL_RESULT_INFER_NAME_NOT_UNIQUE                            0x00060001
#define DSL_RESULT_INFER_NAME_NOT_FOUND                             0x00060002
#define DSL_RESULT_INFER_NAME_BAD_FORMAT                            0x00060003
#define DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND                      0x00060004
#define DSL_RESULT_INFER_MODEL_FILE_NOT_FOUND                       0x00060005
#define DSL_RESULT_INFER_THREW_EXCEPTION                            0x00060006
#define DSL_RESULT_INFER_IS_IN_USE                                  0x00060007
#define DSL_RESULT_INFER_SET_FAILED                                 0x00060008
#define DSL_RESULT_INFER_HANDLER_ADD_FAILED                         0x00060009
#define DSL_RESULT_INFER_HANDLER_REMOVE_FAILED                      0x0006000A
#define DSL_RESULT_INFER_PAD_TYPE_INVALID                           0x0006000B
#define DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER                     0x0006000C
#define DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST                  0x0006000D
```

## Constructors
**Python Example**
```Python
# Filespecs for the Primary GIE
pgie_config_file = './configs/config_infer_primary_nano.txt'
pgie_model_file = './models/Primary_Detector_Nano/resnet10.caffemodel.engine'

# Filespecs for the Secondary GIE
sgie_config_file = './configs/config_infer_secondary_carcolor.txt'
sgie_model_file = './models/Secondary_CarColor/resnet18.caffemodel.engine'

# New Primary GIE using the filespecs above, with interval set to 0
retval = dsl_infer_gie_primary_new('pgie', pgie_config_file, pgie_model_file, 0)
if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # handle error condition

# New Secondary GIE set to Infer on the Primary GIE defined above
retval = dsl_infer_gie_seondary_new('sgie', sgie_config_file, sgie_model_file, 0, 'pgie')
if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # handle error condition

# Add both Primary and Secondary GIEs to an existing Pipeline
retval = dsl_pipeline_component_add_many('pipeline', ['pgie', 'sgie', None])
if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # handle error condition
```

<br>

### *dsl_infer_gie_primary_new*
```C++
DslReturnType dsl_infer_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval);
```
This constructor creates a uniquely named Primary GST Inference Engine (GIE). Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Primary GIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load. Set to NULL and the GIE Plugin will attempt to create a model-engine file based on the configuration file.
* `interval` - [in] frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_gie_primary_new('my-pgie', pgie_config_file, pgie_model_file, 0)
```

<br>

### *dsl_infer_gie_secondary_new*
```C++
DslReturnType dsl_infer_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie, uint interval);
```

This constructor creates a uniquely named Secondary GST Inference Engine (GIE). Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Secondary GIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load. Set to NULL and the GIE Plugin will attempt to create a model-engine file based on the configuration file.
* `infer_on_gie` - [in] unique name of the Primary or Secondary GIE to infer on
* `interval` - [in] frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_gie_seondary_new('my-sgie', sgie_config_file, sgie_model_file, 'my-pgie', 0)
```

<br>

### *dsl_infer_tis_primary_new*
```C++
DslReturnType dsl_infer_tis_primary_new(const wchar_t* name,
    const wchar_t* infer_config_file, uint interval);
```
This constructor creates a uniquely named Primary Triton Inference Server (TIS). Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Primary TIS to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `interval` - [in] frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_tis_primary_new('my-ptis', ptis_config_file, 0)
```

<br>

### *dsl_infer_tis_secondary_new*
```C++
DslReturnType dsl_infer_tis_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* infer_on_tis, uint interval);
```

This constructor creates a uniquely named Secondary Triton Inference Server (TIS). Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Secondary TIS to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `infer_on_tis` - [in] unique name of the Primary or Secondary TIS to infer on
* `interval` - [in] frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_tis_seondary_new('my-stis', stis_config_file, 0, 'my-ptis')
```

<br>

---

## Methods
### *dsl_infer_batch_size_get*
```C++
DslReturnType dsl_infer_batch_size_get(const wchar_t* name, uint* size);
```
This service gets the client defined batch-size setting for the named GIE or TIS. If not set (0-default), the Pipeline will set the batch-size to the same as the Streammux batch-size which - by default - is derived from the number of sources when the Pipeline is called to play. The Streammux batch-size can be set (overridden) by calling [dsl_pipeline_streammux_batch_properties_set](/docs/api-pipeline.md#dsl_pipeline_streammux_batch_properties_set).

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to query.
* `size` - [out] returns the client defined batch size for the named GIE or TIS if set. ). 0 otherwise.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, batch_size = dsl_infer_batch_size_get('my-pgie')
```

<br>

### *dsl_infer_batch_size_set*
```C++
DslReturnType dsl_infer_batch_size_set(const wchar_t* name, uint size);
```
This service sets the client defined batch-size setting for the named GIE or TIS. If not set (0-default), the Pipeline will set the batch-size to the same as the Streammux batch-size which - by default - is derived from the number of sources when the Pipeline is called to play. The Streammux batch-size can be set (overridden) by calling [dsl_pipeline_streammux_batch_properties_set](/docs/api-pipeline.md#dsl_pipeline_streammux_batch_properties_set).

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to query.
* `size` - [in] the new client defined batch size for the named GIE or TIS to use. Set to 0 to unset.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_batch_size_get('my-pgie', 4)
```

<br>

### *dsl_infer_unique_id_get*
```C++
DslReturnType dsl_infer_unique_id_get(const wchar_t* name, uint* id);
```
This service queries the named Primary or Secondary GIE or TIS for its unique id derived from its unique name.

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to query.
* `id` - [out] returns the unique id for the named GIE or TIS

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, id = dsl_infer_unique_id_get('my-pgie')
```

<br>

### *dsl_infer_config_file_get*
```C++
DslReturnType dsl_infer_config_file_get(const wchar_t* name,
    const wchar_t** infer_config_file);
```

This service returns the current Inference Config file in use by the named Primary or Secondary GIE or TIS.

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to query.
* `infer_config_file` - [out] returns the absolute file path/name for the infer config file in use

**Returns**
`DSL_RESULT_SUCCESS` if successful. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, infer_config_file = dsl_infer_config_file_get('my-sgie)
```

<br>

### *dsl_infer_config_file_set*
```C++
DslReturnType dsl_infer_config_file_set(const wchar_t* name,
    const wchar_t* infer_config_file);
```

This service set the Inference Config file to use by the named Primary or Secondary GIE or TIS.

**Parameters**
* `name` - unique name of the Primary or Secondary GIE of TIS to update.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load

**Returns**
`DSL_RESULT_SUCCESS` if successful. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, dsl_infer_config_file_set('my-pgie',  './configs/config_infer_primary_nano.txt')
```

<br>

### *dsl_infer_gie_model_engine_file_get*
```C++
DslReturnType dsl_infer_gie_model_engine_file_get(const wchar_t* name,
    const wchar_t** model_engine_file);
```
The service returns the current Model Engine file in use by the named Primary or Secondary GIE.
This serice is not applicable for Primary or Secondary TISs

**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `model_engine_file` - [out] returns the absolute file path/name for the model engine file in use

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval,  model_engine_file = dsl_infer_gie_model_engine_file_get('my-sgie')
```

<br>

### *dsl_infer_gie_model_engine_file_set*
```C++
DslReturnType dsl_infer_gie_model_engine_file_set(const wchar_t* name,
    const wchar_t* model_engine_file);
```
The service sets the Model Engine file to use for the named Primary or Secondary GIE.
This service is not applicable for Primary or Secondary TISs

**Parameters**
* `name` - unique name of the Primary or Secondary GIE to update.
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists, and the model_engine_file was found, one of the
[Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_gie_model_engine_file_set('my-sgie',  
    './test/models/Secondary_CarColor/resnet18.caffemodel_b16_fp16.engine"')
```

<br>

### *dsl_infer_gie_tensor_meta_settings_get*
```C++
DslReturnType dsl_infer_gie_tensor_meta_settings_get(const wchar_t* name,
    boolean* input_enabled, boolean* output_enabled);
```
The service gets the current input and output tensor-meta settings in use by the named Primary or Secondary GIE.

**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `input_enabled` - [out] if true, the GIE will preprocess input tensors attached as metadata instead of preprocessing inside the plugin, false otherwise.
* `output_enable` - [out] if true, the GIE will attach tensor outputs as metadata on the GstBuffer.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, input_enabled, output_enabled = dsl_infer_gie_tensor_meta_settings_get('my-pgie')
```

<br>

### *dsl_infer_gie_tensor_meta_settings_set*
```C++
DslReturnType dsl_infer_gie_tensor_meta_settings_set(const wchar_t* name,
    boolean input_enabled, boolean output_enabled);
```
The service sets the input amd output tensor-meta settings for the named Primary or Secondary GIE.

**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `input_enabled` - [in] set to true to have the GIE preprocess input tensors attached as metadata instead of preprocessing inside the plugin, false otherwise.
* `output_enable` - [in] set to true to have the GIE attach tensor outputs as metadata on the GstBuffer.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_gie_tensor_meta_settings_get('my-pgie', True, False)
```

<br>

### *dsl_infer_interval_get*
```C++
DslReturnType dsl_infer_interval_get(const wchar_t* name, uint* interval);
```
This service queries the named Primary or Secondary GIE or TIS for its current inference interval setting.

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to query.
* `interval` - [out] returns the current inference interval in use by the named GIE or TIS

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, interval = dsl_gie_interval_get('my-pgie')
```

<br>

### *dsl_infer_interval_set*
```C++
DslReturnType dsl_infer_interval_set(const wchar_t* name, uint interval);
```
This service updates the inference interval to use by the named Primary or Secondary GIE or TIS

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to update.
* `interval` - [in] inference interval to use for the named GIE or TIS

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists one of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gie_interval_set('my-pgie', 2)
```

<br>

### *dsl_infer_primary_pph_add*
```C++
DslReturnType dsl_infer_primary_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named Primary GIE or TIS.

**Parameters**
* `name` - [in] unique name of the Primary GIE or TIS to update.
* `handler` - [in] unique name of Pad Probe Handler to add
* `pad` - [in] to which of the two pads to add the handler: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
* Example using Nvidia's pyds lib to handle batch-meta data

```Python
retval = dsl_infer_primary_pph_add('my-primary-gie', 'my-pph-handler', `DSL_PAD_SINK`)
```

<br>

### *dsl_infer_primary_pph_remove*
```C++
DslReturnType dsl_infer_primary_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named Primary GIE or TIS. The service will fail if the named handler is not owned by the Primary GIE

**Parameters**
* `name` - [in] unique name of the Primary GIE to TIS to update.
* `handler` - [in] unique name of Pad Probe Handler to remove
* `pad` - [in] to which of the two pads to remove the handler from: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_infer_primary_pph_remove('my-primary-gie', 'my-pph-handler', `DSL_PAD_SINK`)
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
* **Primary and Secondary Inference**
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
* [Display Types](/docs/api-display-types.md)
* [branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
