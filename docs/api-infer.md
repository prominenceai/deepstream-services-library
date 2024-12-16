# Inference Engine and Server API Reference
The DeepStream Services Library (DSL) provides services for Nvidia's three Inference Plugins; 

1. [Audio Inference Engine (AIE)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferaudio.html#gst-nvinferaudio) - Primary only - referred to as a PAIE.
2. [GPU Inference Engine (GIE)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer) - Primary and Secondary - referred to as PGIEs and SGIES.
3. [Triton Inference Server (TIS)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#gst-nvinferserver) - Primary and Secondary - reffered to as PTISs and STISs.

Pipelines can have multiple Primary Engines or Servers (PAIEs, PGIEs, or PTISs) linked in succession to operate on the full frame, with any number of corresponding Secondary GIEs or TISs. Pipelines that have secondary GIEs/TISs but no Primary GIE/TIS will fail to Link and Play. Secondary GIEs/TISs can `infer-on` both Primary and Secondary GIEs/TISs creating multiple levels of inference. 

**IMPORTANT**: the current release supports up to two levels of secondary inference.

## Construction and Destruction
Primary AIEs, GIEs and TISs are constructed by calling [`dsl_infer_aie_primary_new`](#dsl_infer_aie_primary_new), [`dsl_infer_gie_primary_new`](#dsl_infer_gie_primary_new) and [`dsl_infer_tis_primary_new`](#dsl_infer_tis_primary_new) respectively. Secondary GIEs and TISs are created by calling [`dsl_infer_gie_secondary_new`](#dsl_infer_gie_secondary_new) and [`dsl_infer_tis_secondary_new`](#dsl_infer_tis_secondary_new) respectively. As with all components, Primary and Secondary Inference Engines and Servers must be uniquely named from all other components created. All AIEs, GIEs and TIEs are deleted by calling [`dsl_component_delete`](api-component.md#dsl_component_delete), [`dsl_component_delete_many`](api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](api-component.md#dsl_component_delete_all).

## Inference Configuration
All Inference Engines and Servers require an **Inference Configuration File**. Once created, clients can query the Engine or Server for their Config File in-use by calling [`dsl_infer_config_file_get`](#dsl_infer_config_file_get) or change the configuration by calling [`dsl_infer_config_file_set`](#dsl_infer_config_file_set).

## Model Engine Files
With Triton Inference Servers (PTISs and STISs), the model-engine-file must be specified in the inference-configuration-file. 

With Inference Engines (PAIEs, PGIEs, and SGIEs), the model-engine-file, can be specified in the inference-configuration-file or by using the constructor's `model_engine_file` parameter. The file in use can be queried by calling [`dsl_infer_engine_model_engine_file_get`](#dsl_infer_engine_model_engine_file_get) or changed with [`dsl_infer_engine_model_engine_file_set`](#dsl_infer_engine_model_engine_file_set).

With Inference Engines (PAIEs, PGIEs, and SGIEs), the model-engine-file can be updated at runtime. Refer to [Dynamic Model Updates](#dynamic-model-updates) below.


## Dynamic Model Updates.
All Inference Engines (PAIEs, PGIEs, and SGIEs) support dynamic model updates. See NVIDIA's [on-the-fly model updates](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_the_fly_model.html) for information on restrictions. Clients can register a [model-update-listener](#dsl_infer_engine_model_update_listener_cb) callback function to be notified when a new model-engine is successfully loaded. See [`dsl_infer_engine_model_update_listener_add`](#dsl_infer_engine_model_update_listener_add), not applicable to TISs. 

## Unique Id
**IMPORTANT!** DSL explicitly assigns each Inference Component a unique component-id overriding the (optional) parameter in the inference config file. The unique component-id is derived from the first available unused id starting with 1, meaning the first component will be assigned id 1, the second id 2 and so on. The id will be reused if the inference component is deleted and a new one created. The value assigned to the Inference component can be queried by calling [`dsl_infer_unique_id_get`](#dsl_infer_unique_id_get). All Object metadata structures created by the named Inference Component will include a `unique_component_id` field assigned with this id.

## Video Inference Interval
**IMPORTANT!** With Video Inference Engines and Servers (PGIE, SGIE, PTIS, and STIS) DSL sets the inference interval with the input parameter provided on construction overriding the (optional) parameter in the inference config file. The interval for inferencing -- or the number of frames to skip between inferencing -- is set as an unsigned integer with `0 = every frame`, `1 = every other frame`, `2 = every 3rd frame`, etc., when created.  The current interval in-use can be queried by calling [`dsl_infer_video_interval_get`](#dsl_infer_video_interval_get), and changed by calling [`dsl_infer_video_interval_set`](#dsl_infer_video_interval_set).

## Inference Batch Size
**IMPORTANT!** DSL sets the inference batch size overriding the parameter in the inference config file for all Engines and Servers. The batch size for each AIE/GIE/TIS can be set explicitly by calling [`dsl_infer_batch_size_set`](#dsl_infer_batch_size_set). If not set (0-default), the Pipeline will set the batch-size to the same value as the Streammux batch-size which - by default - is derived from the number of sources when the Pipeline is called to play. The Streammux batch-size can be set (overridden) by calling [`dsl_pipeline_streammux_batch_properties_set`](/docs/api-pipeline.md#dsl_pipeline_streammux_batch_properties_set).

## Adding and Removing
AIEs,GIEs and TISs are added to a Pipeline by calling [`dsl_pipeline_component_add`](/docs/api-pipeline.md#dsl_pipeline_component_add) and [`dsl_pipeline_component_add_many`](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed by calling [`dsl_pipeline_component_remove`](/docs/api-pipeline.md#dsl_pipeline_component_remove) and [`dsl_pipeline_component_remove_many`](/docs/api-pipeline.md#dsl_pipeline_component_remove_many).

A similar set of Services are used when adding/removing a GIE/TIS to/from a branch: [`dsl_branch_component_add`](api-branch.md#dsl_branch_component_add), [`dsl_branch_component_add_many`](/docs/api-branch.md#dsl_branch_component_add_many), [`dsl_branch_component_remove`](/docs/api-branch.md#dsl_branch_component_remove), [`dsl_branch_component_remove_many`](/docs/api-branch.md#dsl_branch_component_remove_many), and [`dsl_branch_component_remove_all`](/docs/api-branch.md#dsl_branch_component_remove_all).

## Adding/Removing Pad-Probe-handlers
Multiple sink (input) and/or source (output) [Pad-Probe Handlers](/docs/api-pph.md) can be added to any AIE, GIE, or TIS by calling [`dsl_infer_pph_add`](#dsl_infer_pph_add) and removed with [`dsl_infer_pph_remove`](#dsl_infer_pph_remove).

---

## Primary and Secondary Inference API
**Client Callback Typedefs**
* [`dsl_infer_engine_model_update_listener_cb`](#dsl_infer_engine_model_update_listener_cb)

**Constructors**
* [`dsl_infer_aie_primary_new`](#dsl_infer_aie_primary_new)
* [`dsl_infer_gie_primary_new`](#dsl_infer_gie_primary_new)
* [`dsl_infer_gie_secondary_new`](#dsl_infer_gie_secondary_new)
* [`dsl_infer_tis_primary_new`](#dsl_infer_tis_primary_new)
* [`dsl_infer_tis_secondary_new`](#dsl_infer_tis_secondary_new)

**Audio Inference Engine (AIE) Methods**
* [`dsl_infer_aie_frame_size_get`](#dsl_infer_aie_frame_size_get)
* [`dsl_infer_aie_frame_size_set`](#dsl_infer_aie_frame_size_set)
* [`dsl_infer_aie_hop_size_get`](#dsl_infer_aie_hop_size_get)
* [`dsl_infer_aie_hop_size_set`](#dsl_infer_aie_hop_size_set)
* [`dsl_infer_aie_transform_get`](#dsl_infer_aie_transform_get)
* [`dsl_infer_aie_transform_set`](#dsl_infer_aie_transform_set)

**Inference Engine (PAIE, PGIE & SGIE) Methods**
* [`dsl_infer_engine_model_engine_file_get`](#dsl_infer_engine_model_engine_file_get)
* [`dsl_infer_engine_model_engine_file_set`](#dsl_infer_engine_model_engine_file_set)
* [`dsl_infer_engine_model_update_listener_add`](#dsl_infer_engine_model_update_listener_add)
* [`dsl_infer_engine_model_update_listener_remove`](#dsl_infer_engine_model_update_listener_remove)
* [`dsl_infer_engine_tensor_meta_settings_get`](#dsl_infer_engine_tensor_meta_settings_get)
* [`dsl_infer_engine_tensor_meta_settings_set`](#dsl_infer_engine_tensor_meta_settings_set)

**Video Inference Engine and Server (PGIE, SGIE, PTIS, STIS) Methods**
* [`dsl_infer_video_interval_get`](#dsl_infer_video_interval_get)
* [`dsl_infer_video_interval_set`](#dsl_infer_video_interval_set)

**Common Methods**
* [`dsl_infer_batch_size_get`](#dsl_infer_batch_size_get)
* [`dsl_infer_batch_size_set`](#dsl_infer_batch_size_set)
* [`dsl_infer_unique_id_get`](#dsl_infer_unique_id_get)
* [`dsl_infer_config_file_get`](#dsl_infer_config_file_get)
* [`dsl_infer_config_file_set`](#dsl_infer_config_file_set)
* [`dsl_infer_pph_add`](#dsl_infer_pph_add)
* [`dsl_infer_pph_remove`](#dsl_infer_pph_remove)

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
#define DSL_RESULT_INFER_ID_NOT_FOUND                               0x0006000E
#define DSL_RESULT_INFER_CALLBACK_ADD_FAILED                        0x0006000F
#define DSL_RESULT_INFER_CALLBACK_REMOVE_FAILED                     0x00060010
```

---

## Client Callback Typedefs
### *dsl_infer_engine_model_update_listener_cb*
```C++
typedef void (*dsl_infer_engine_model_update_listener_cb)(const wchar_t* name,
   const wchar_t* model_engine_file, void* client_data);
```
Callback typedef for a client model-update listener. Functions of this type are added to a Primary or Secondary Inference Engine by calling [dsl_infer_engine_model_update_listener_add](#dsl_infer_engine_model_update_listener_add). Once added, the function will be called each time a new model-engine has been successfully loaded while the Pipeline is in a state of playing.

**Parameters**
* `name` - [in] name of the Primary or Secondary Inference Component that loaded the model-engine.
* `model_engine_file` - [in] one of [DSL_PIPELINE_STATE](#DSL_PIPELINE_STATE) constants for the new pipeline state
* `client_data` - [in] opaque pointer to the client's user data provided to the Inference Component when this function is added.

<br>

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

### *dsl_infer_aie_primary_new*
```C++
DslReturnType dsl_infer_aie_primary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, const wchar_t* model_engine_file, 
    uint frame_size, uint hop_size, const wchar_t* transform);
```
This constructor creates a uniquely named Primary Audio Inference Engine (PAIE). Construction will fail if the name is currently in use. 

**IMPORTANT!** The PAIE's media-type is set to `DSL_MEDIA_TYPE_AUDIO_ONLY` (read-only).

**IMPORTANT!** The audio-transform parameter is a string representation of a GstStructure as defined under the [NVIDIA DeepStream Audio Inference Plugin Reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferaudio.html#id5).

**Parameters**
* `name` - [in] unique name for the Primary AIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load. Set to NULL to use the engine file specified in the config file.
* `frame_size` - [in] audio frame-size in units of samples/frame.
* `hop_size` - [in] audio hop-size in units of samples.
* `transform` - [in] transform type and parameters to use.

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
tranform = \
   'melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,sample_rate=44100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,p2db_top_db=(float)80.0'

retval = dsl_infer_aie_primary_new('my-pgie', paie_config_file, paie_model_file, 
   441000, 110250, transform)
```

<br>

### *dsl_infer_gie_primary_new*
```C++
DslReturnType dsl_infer_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
   const wchar_t* model_engine_file, uint interval);
```
This constructor creates a uniquely named Primary GPU Inference Engine (PGIE). Construction will fail if the name is currently in use.

**IMPORTANT!** The PGIE's media-type is set to `DSL_MEDIA_TYPE_VIDEO_ONLY` (read-only).

**Parameters**
* `name` - [in] unique name for the Primary GIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load. Set to NULL to use the engine file specified in the config file.
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

This constructor creates a uniquely named Secondary GPU Inference Engine (SGIE). Construction will fail if the name is currently in use.

**IMPORTANT!** The SGIE's media-type is set to `DSL_MEDIA_TYPE_VIDEO_ONLY` (read-only).

**Parameters**
* `name` - [in] unique name for the Secondary GIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load. Set to NULL to use the engine file specified in the config file.
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
This constructor creates a uniquely named Primary Triton Inference Server (PTIS). Construction will fail if the name is currently in use.

**IMPORTANT!** The PTIS's media-type is set to `DSL_MEDIA_TYPE_VIDEO_ONLY` (read-only).

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

This constructor creates a uniquely named Secondary Triton Inference Server (STIS). Construction will fail if the name is currently in use.

**IMPORTANT!** The STIS's media-type is set to `DSL_MEDIA_TYPE_VIDEO_ONLY` (read-only).

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

## Audio Inference Engine (AIE) Methods

### *dsl_infer_aie_frame_size_get*
```C++
DslReturnType dsl_infer_aie_frame_size_get(const wchar_t* name, uint* frame_size);
```
This service gets the current audio frame-size setting for the named Audio Inference Engine (AIE).

**Parameters**
* `name` - [in] unique name of the AIE to query.
* `frame_size` - [out] current frame-size setting for the named AIE in units of samples per frame.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval, frame_size = dsl_infer_aie_frame_size_get('my-paie')
```

<br>

### *dsl_infer_aie_frame_size_set*
```C++
DslReturnType dsl_infer_aie_frame_size_set(const wchar_t* name, uint frame_size);
```
This service sets the audio frame-size setting for the named Audio Inference Engine (AIE) to use.

**Parameters**
* `name` - [in] unique name of the AIE to update.
* `frame_size` - [in] the new frame-size setting for the named AIE to use in units of samples per frame.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_aie_frame_size_set('my-paie', 441000)
```

<br>

### *dsl_infer_aie_hop_size_get*
```C++
DslReturnType dsl_infer_aie_hop_size_get(const wchar_t* name, uint* hop_size);
```
This service gets the current audio hop-size setting for the named Audio Inference Engine (AIE).

**Parameters**
* `name` - [in] unique name of the AIE to query.
* `hop_size` - [out] current hop-size setting for the named AIE in units of samples.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval, hop_size = dsl_infer_aie_hop_size_get('my-paie')
```

<br>

### *dsl_infer_aie_hop_size_set*
```C++
DslReturnType dsl_infer_aie_hop_size_set(const wchar_t* name, uint hop_size);
```
This service sets the audio hop-size setting for the named Audio Inference Engine (AIE) to use.

**Parameters**
* `name` - [in] unique name of the AIE to update.
* `hop_size` - [in] the new hop-size setting for the named AIE to use in units of samples.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_aie_hop_size_set('my-paie', 110250)
```

<br>

### *dsl_infer_aie_transform_get*
```C++
DslReturnType dsl_infer_aie_transform_get(const wchar_t* name, 
    const wchar_t** transform);
```
This service gets the current audio transform and parameters in use by the named Audio Inference Engine (AIE).  

**IMPORTANT!** The audio-transform parameter is a string representation of a GstStructure as defined under the [NVIDIA DeepStream Audio Inference Plugin Reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferaudio.html#id5).

**Parameters**
* `name` - [in] unique name of the AIE to query.
* `transform` - [out] current transform and parameters for the named AIE.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure


**Python Example**
```Python
retval, transform = dsl_infer_aie_transform_get('my-paie')
```

<br>

### *dsl_infer_aie_transform_set*
```C++
DslReturnType dsl_infer_aie_transform_set(const wchar_t* name, 
   const wchar_t* transform);
```
This service sets the audio transform setting for the named Audio Inference Engine (AIE) to use.

**IMPORTANT!** The audio-transform parameter is a string representation of a GstStructure as defined under the [NVIDIA DeepStream Audio Inference Plugin Reference](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferaudio.html#id5).

**Parameters**
* `name` - [in] unique name of the AIE to update.
* `transform` - [in] the new transform and parameters for the named AIE. 

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
tranform = \
   'melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,sample_rate=4100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,p2db_top_db=(float)80.0'

retval = dsl_infer_aie_transform_set('my-paie', transform)
```

<br>

## Inference Engine (PAIE, PGIE, & SGIE) Methods

### *dsl_infer_engine_model_engine_file_get*
```C++
DslReturnType dsl_infer_engine_model_engine_file_get(const wchar_t* name,
   const wchar_t** model_engine_file);
```
The service returns the current Model Engine file in use by the named PAIE, PGIE, or SGIE.
This service is not applicable for Primary or Secondary TISs

**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `model_engine_file` - [out] returns the absolute file path/name for the model engine file in use

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval,  model_engine_file = dsl_infer_engine_model_engine_file_get('my-sgie')
```

<br>

### *dsl_infer_engine_model_engine_file_set*
```C++
DslReturnType dsl_infer_engine_model_engine_file_set(const wchar_t* name,
   const wchar_t* model_engine_file);
```
The service sets the model-engine-file for the named PAIE, PGIE, or SGIE.

This service is not applicable for Primary or Secondary TISs.

**IMPORTANT!** This service can be called in any Pipeline state. [On-the-fly](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_the_fly_model.html#) model updates are performed asynchronously. Clients can register a [model-update-listener](#dsl_infer_engine_model_update_listener_cb) callback function to notified when the new model-engine is successfully loaded. See [`dsl_infer_engine_model_update_listener_add`](#dsl_infer_engine_model_update_listener_add).

**Parameters**
* `name` - unique name of the PAIE, PGIE, or SGIE to update.
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load

**Returns**

`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_engine_model_engine_file_set('my-sgie', 
   './test/models/Secondary_CarColor/resnet18.caffemodel_b16_fp16.engine"')
```

<br>

### *dsl_infer_engine_model_update_listener_add*
```C++
DslReturnType dsl_infer_engine_model_update_listener_add(const wchar_t* name,
   dsl_infer_engine_model_update_listener_cb listener, void* client_data);
```
The service adds a [model-update-listener](#dsl_infer_engine_model_update_listener_cb) callback function to a named PAIE, PGIE, or SGIE. The callback will be called after a new model-engine-file has been successfully loaded while the Pipeline is in a state of playing.

This service is not applicable for Primary or Secondary TISs.

**Parameters**
* `name` - unique name of the PAIE, PGIE, or SGIE to update.
* `listener` - [in] client callback function to add.
* `client_data` - [in] opaque pointer to client data returned to the listener callback function.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
##
# Function to be called when a model update has been completed
# 
def model_update_listener(name, model_engine_file, client_data):
   print(name, "completed loading model", model_engine_file)

retval = dsl_infer_engine_model_update_listener_add('my-pgie',
   model_update_listener, None)
```

<br>

### *dsl_infer_engine_model_update_listener_remove*
```C++
DslReturnType dsl_infer_engine_model_update_listener_remove(const wchar_t* name,
   dsl_infer_engine_model_update_listener_cb listener);
```
This service removes a [model-update-listener](#dsl_infer_engine_model_update_listener_cb) callback function from a named PAIE, PGIE, or SGIE.

This service is not applicable for Primary or Secondary TISs

**Parameters**
* `name` - unique name of the PAIE, PGIE, or SGIE to update.
* `listener` - [in] client callback function to remove.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_infer_engine_model_update_listener_remove('my-pgie',
   model_update_listener)
```

<br>

### *dsl_infer_engine_tensor_meta_settings_get*
```C++
DslReturnType dsl_infer_engine_tensor_meta_settings_get(const wchar_t* name,
   boolean* input_enabled, boolean* output_enabled);
```
The service gets the current input and output tensor-meta settings in use by the named PAIE, PGIE, or SGIE.

**Parameters**
* `name` - unique name of the PAIE, PGIE, or SGIE to query.
* `input_enabled` - [out] if true, the Inference Engine will preprocess input tensors attached as metadata instead of preprocessing inside the plugin, false otherwise.
* `output_enable` - [out] if true, the Inference Engine will attach tensor outputs as metadata on the GstBuffer.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, input_enabled, output_enabled = dsl_infer_engine_tensor_meta_settings_get('my-pgie')
```
<br>

### *dsl_infer_engine_tensor_meta_settings_set*
```C++
DslReturnType dsl_infer_engine_tensor_meta_settings_set(const wchar_t* name,
   boolean input_enabled, boolean output_enabled);
```
The service sets the input amd output tensor-meta settings for the named PAIE, PGIE, or SGIE.

**Parameters**
* `name` - unique name of the PAIE, PGIE, or SGIE to update.
* `input_enabled` - [in] set to true to have the Inference Engine preprocess input tensors attached as metadata instead of preprocessing inside the plugin, false otherwise.
* `output_enable` - [in] set to true to have the Inference Engine attach tensor outputs as metadata on the GstBuffer.

**Returns**
`DSL_RESULT_SUCCESS` on success. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_infer_engine_tensor_meta_settings_get('my-pgie', True, False)
```

<br>

---

## Common Methods

### *dsl_infer_batch_size_get*
```C++
DslReturnType dsl_infer_batch_size_get(const wchar_t* name, uint* size);
```
This service gets the client defined batch-size setting for the named GIE or TIS. If not set (0-default), the Pipeline will set the batch-size to the same as the Streammux batch-size which - by default - is derived from the number of sources when the Pipeline is called to play. The Streammux batch-size can be set (overridden) by calling [`dsl_pipeline_streammux_batch_properties_set`](/docs/api-pipeline.md#dsl_pipeline_streammux_batch_properties_set).

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
This service sets the client defined batch-size setting for the named GIE or TIS. If not set (0-default), the Pipeline will set the batch-size to the same as the Streammux batch-size which - by default - is derived from the number of sources when the Pipeline is called to play. The Streammux batch-size can be set (overridden) by calling [`dsl_pipeline_streammux_batch_properties_set`](/docs/api-pipeline.md#dsl_pipeline_streammux_batch_properties_set).

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE or TIS to update.
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

This service sets the Inference Config file for named Primary or Secondary GIE or TIS to use.

**IMPORTANT!** This service can be called in any Pipeline state. [On-the-fly](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_the_fly_model.html#) model updates are performed asynchronously. With Primary and Secondary GIEs, clients can register a [model-update-listener](#dsl_infer_engine_model_update_listener_cb) callback function to notified when a new model-engine is successfully loaded. See [`dsl_infer_engine_model_update_listener_add`](#dsl_infer_engine_model_update_listener_add), not applicable to TISs. 

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

### *dsl_infer_video_interval_get*
```C++
DslReturnType dsl_infer_video_interval_get(const wchar_t* name, uint* interval);
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

### *dsl_infer_video_interval_set*
```C++
DslReturnType dsl_infer_video_interval_set(const wchar_t* name, uint interval);
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

### *dsl_infer_pph_add*
```C++
DslReturnType dsl_infer_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named Primary or Secondary GIE or TIS.

**Parameters**
* `name` - [in] unique name of the Inference Component to update.
* `handler` - [in] unique name of Pad Probe Handler to add.
* `pad` - [in] to which of the two pads to add the handler: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_infer_pph_add('my-primary-gie', 'my-pph-handler', DSL_PAD_SINK)
```

<br>

### *dsl_infer_pph_remove*
```C++
DslReturnType dsl_infer_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named Primary or Secondary GIE or TIS. The service will fail if the named handler is not owned by the Inference Component

**Parameters**
* `name` - [in] unique name of the Inference Component to update.
* `handler` - [in] unique name of Pad Probe Handler to remove
* `pad` - [in] to which of the two pads to remove the handler from: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_infer_pph_remove('my-primary-gie', 'my-pph-handler', DSL_PAD_SINK)
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
* [Demuxer and Splitter Tees](/docs/api-tee.md)
* [Remuxer](/docs/api-remuxer.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [GST Element](/docs/api-gst.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Types](/docs/api-display-types.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
