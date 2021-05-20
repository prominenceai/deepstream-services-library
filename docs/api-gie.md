# GST Inference Engine (GIE) API Reference
Pipelines can have at most one Primary GIE with any number of Secondary GIEs (only limited by hardware). Pipelines that have secondary GIEs but no primary GIE will fail to Link and Play. Secondary GIEs operate as detectors on Primary bounding boxs. Secondary GIEs can `infer-on` both Primary and Secondary GIE's creating multiple levels of inference. 

Primary and Secondary GIEs are constructed by calling [dsl_gie_primary_new](#dsl_gie_primary_new) and 
[dsl_gie_secondary_new](#dsl_gie_secondary_new) respectively. As with all components, Primary and Secondary GIEs must be uniquely named from all other components created. 

The interval for inferencing (number of batched buffers skipped)  Secondary GIEs inherit the single Primary GIE's interval setting when the Pipeline is played. The current interval can obtained from any GIE be changed by calling [dsl_gie_primary_interval_set](#dsl_gie_prmary_interval_set).

Once created, clients can query both Primary and Secondary GIEs for their Infer Config File and Model engine file in use by calling [dsl_gie_infer_config_file_get](#dsl_gie_infer_config_file_get) and [dsl_gie_model_engine_file_get](#dsl_gie_model_engine_file_get). Clients can update the File settings, while a GIE is not `in-use`, by calling [dsl_gie_infer_config_file_set](#dsl_gie_infer_config_file_set) and [dsl_gie_model_engine_file_set](#dsl_gie_model_engine_file_set).

The raw output for any GIE can be streamed to file by calling [dsl_gie_enable_raw_ouput](#dsl_gie_enable_raw_ouput). Raw output is disabled by calling [dsl_gie_disable_raw_output](#dsl_gie_enable_raw_ouput). 

GIEs are added to a Pipeline by calling [dsl_pipeline_component_add](#dsl_pipeline_component_add) and [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many), and removed by calling [dsl_pipeline_component_remove](#dsl_pipeline_component_remove) and [dsl_pipeline_component_remove_many](#dsl_pipeline_component_remove_many).

Primary and Secondary GIEs are deleted by calling [dsl_component_delete](#dsl_component_delete) or [dsl_component_delete_many](#dsl_component_delete_many)

## Primary and Secondary GIE API
**Constructors**
* [dsl_gie_primary_new](#dsl_gie_primary_new)
* [dsl_gie_secondary_new](#dsl_gie_secondary_new)


**Methods**
* [dsl_gie_infer_config_file_get](#dsl_gie_infer_config_file_get)
* [dsl_gie_infer_config_file_set](#dsl_gie_infer_config_file_set)
* [dsl_gie_model_engine_file_get](#dsl_gie_model_engine_file_get)
* [dsl_gie_model_engine_file_set](#dsl_gie_model_engine_file_set)
* [dsl_gie_enable_raw_ouput](#dsl_gie_enable_raw_ouput)
* [dsl_gie_disable_raw_output](#dsl_gie_enable_raw_ouput)
* [dsl_gie_interval_get](#dsl_gie_interval_get)
* [dsl_gie_interval_set](#dsl_gie_interval_set)
* [dsl_gie_primary_pph_add](#dsl_gie_primary_pph_add)
* [dsl_gie_primary_pph_remove](#dsl_gie_primary_pph_remove)
* [dsl_gie_secondary_infer_on_get](#dsl_gie_secondary_infer_on_get)
* [dsl_gie_secondary_infer_on_set](#dsl_gie_secondary_infer_on_set)
* [dsl_gie_num_in_use_get](#dsl_gie_num_in_use_get)
* [dsl_gie_num_in_use_max_get](#dsl_gie_num_in_use_max_get)
* [dsl_gie_num_in_use_max_set](#dsl_gie_num_in_use_max_set)

<br>

## Constructors
**Python Example**
```Python
# Filespecs for the Primary GIE
pgie_config_file = './configs/config_infer_primary_nano.txt'
pgie_model_file = './models/Primary_Detector_Nano/resnet10.caffemodel.engine'

# Filespecs for the Secondary GIE
sgie_config_file = './configs/config_infer_secondary_carcolor
sgie_model_file = './models/Secondary_CarColor/resnet18.caffemodel'

# New Primary GIE using the filespecs above, with interval set to  0
retval = dsl_gie_primary_new('pgie', pgie_config_file, pgie_model_file, 0)

if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # handle error condition

# New Secondary GIE set to Infer on the Primary GIE defined above
retval = dsl_gie_seondary_new('sgie', sgie_config_file, sgie_model_file, 0, 'pgie')

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

### *dsl_gie_primary_new*
```C++
DslReturnType dsl_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval);
```
The constructor creates a uniquely named Primary GIE. Construction will fail
if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Primary GIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load
* `interval` - [in] frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
# Filespecs for the Primary GIE
pgie_config_file = './configs/config_infer_primary_nano.txt'
pgie_model_file = './models/Primary_Detector_Nano/resnet10.caffemodel.engine'

# New Primary GIE using the filespecs above, with interval set to  0
retval = dsl_gie_primary_new('my-pgie', pgie_config_file, pgie_model_file, 0)
```

<br>

### *dsl_gie_secondary_new*
```C++
DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie_name, uint interval);
```

This constructor creates a uniquely named Secondary GIE. Construction will fail
if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Secondary GIE to create.
* `infer_config_file` - [in] relative or absolute file path/name for the infer config file to load
* `model_engine_file` - [in] relative or absolute file path/name for the model engine file to load
* `infer_on_gie_name` - [in] unique name of the Primary or Secondary GIE to infer on
* `interval` - [in] frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
# Filespecs for the Secondary GIE
sgie_config_file = './test/configs/config_infer_secondary_carcolor_nano.txt'
sgie_model_file = './test/models/Secondary_CarColor/resnet18.caffemodel_b16_fp16.engine'

# New Secondary GIE set to Infer on the Primary GIE defined above
retval = dsl_gie_seondary_new('my-sgie', sgie_config_file, sgie_model_file, 0, 'my-pgie')
```

<br>

## Methods
### *dsl_gie_infer_config_file_get*
```C++
DslReturnType dsl_gie_infer_config_file_get(const wchar_t* name, const wchar_t** infer_config_file);
```

This service return the current Infer Engine Config file in use by the named Primary or Secondary GIE.

**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE to query.
* `infer_config_file` - [out] returns the absolute file path/name for the infer config file in use

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists. `DSL_RESULT_GIE_NAME_NOT_FOUND` otherwise

**Python Example**
```Python
retval, infer_config_file = dsl_gie_infer_config_file_get('my-sgie)
```

<br>

### *dsl_gie_infer_config_file_set*
```C++
DslReturnType dsl_gie_infer_config_file_set(const wchar_t* name, const wchar_t* infer_config_file);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to update.
* `infer_config_file` - relative or absolute file path/name for the infer config file to load

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists, and the infer_config_file was found, one of the 
[Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, dsl_gie_infer_config_file_get('my-sgie',  './configs/config_infer_primary_nano.txt')
```

<br>

### *dsl_gie_model_engine_file_get*
```C++
DslReturnType dsl_gie_model_engine_file_get(const wchar_t* name, const wchar_t** model_engine_file);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `model_engine_file` - returns the absolute file path/name for the model engine file in use

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists. `DSL_RESULT_GIE_NAME_NOT_FOUND` otherwise

**Python Example**
```Python
retval,  model_engine_file = dsl_gie_model_engine_file_get('my-sgie')
```

<br>

### *dsl_gie_model_engine_file_set*
```C++
DslReturnType dsl_gie_model_engine_file_set(const wchar_t* name, const wchar_t* model_engine_file);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to update.
* `model_engine_file` - relative or absolute file path/name for the model engine file to load

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists, and the model_engine_file was found, one of the 
[Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gie_infer_config_file_get('my-sgie',  
    './test/models/Secondary_CarColor/resnet18.caffemodel_b16_fp16.engine"')
```

<br>

### *dsl_gie_interval_get*
```C++
DslReturnType dsl_gie_interval_get(const wchar_t* name, uint* interval);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `interval[out]` - returns the current frame interval in use by the named GIE

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists. `DSL_RESULT_GIE_NAME_NOT_FOUND` otherwise

**Python Example**
```Python
retval, interval = dsl_gie_interval_get('my-pgie')
```

<br>

### *dsl_gie_interval_set*
```C++
DslReturnType dsl_gie_interval_set(const wchar_t* name, uint interval);
```
**Parameters**
* `name` - [in] unique name of the Primary or Secondary GIE to update.
* `interval` - [in] frame interval in use by the named GIE

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists one of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_gie_interval_set('my-pgie', 2)
```

<br>

### *dsl_gie_primary_pph_add*
```C++
DslReturnType dsl_gie_primary_pph_add(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service adds a [Pad Probe Handler](/docs/api-pph.md) to either the Sink or Source pad of the named Primary GIE.

**Parameters**
* `name` - [in] unique name of the Primary GIE to update.
* `handler` - [in] unique name of Pad Probe Handler to add
* `pad` - [in] to which of the two pads to add the handler: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
* Example using Nvidia's pyds lib to handle batch-meta data

```Python
retval = dsl_gie_primary_pph_add('my-primary-gie', 'my-pph-handler', `DSL_PAD_SINK`)
```

<br>

### *dsl_gie_primary_pph_remove*
```C++
DslReturnType dsl_gie_primary_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from either the Sink or Source pad of the named Primary GIE. The service will fail if the named handler is not owned by the Primary GIE

**Parameters**
* `name` - [in] unique name of the Primary GIE to update.
* `handler` - [in] unique name of Pad Probe Handler to remove
* `pad` - [in] to which of the two pads to remove the handler from: `DSL_PAD_SIK` or `DSL_PAD SRC`

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_gie_primary_pph_remove('my-primary-gie', 'my-pph-handler', `DSL_PAD_SINK`)
```

<br>


### *dsl_gie_num_in_use_get*
```C++
uint dsl_gie_num_in_use_get();
```
This service returns the total number of all Primary and Secondary GIEs currently `in-use` by all Pipelines.

**Returns**
* The current number of GIEs `in-use`

**Python Example**
```Python
gies_in_use = dsl_gie_num_in_use_get()
```

<br>

### *dsl_gie_num_in_use_max_get*
```C++
uint dsl_gie_num_in_use_max_get();
```
This service returns the "maximum number of Prmary and Secondary GIEs" that can be `in-use` at any one time, defined as `DSL_DEFAULT_GIE_NUM_IN_USE_MAX` on service initilization, and can be updated by calling [dsl_gie_num_in_use_max_set](#dsl_gie_num_in_use_max_set). The actual maximum is impossed by the Jetson model in use. It's the responsibility of the client application to set the value correctly.

**Returns**
* The current max number of Primary and Secondary GIEs that can be `in-use` by all Pipelines at any one time. 

**Python Example**
```Python
max_gie_in_use = dsl_gie_num_in_use_max_get()
```

<br>

### *dsl_gie_num_in_use_max_set*
```C++
boolean dsl_gie_num_in_use_max_set(uint max);
```
This service sets the "maximum number of Primary and Secondary GIEs" that can be `in-use` at any one time. The value is defined as `DSL_DEFAULT_GIE_NUM_IN_USE_MAX` on service initilization. The actual maximum is impossed by the Jetson model in use. It's the responsibility of the client application to set the value correctly.

**Returns**
* `false` if the new value is less than the actual current number of GIEs in use, `true` otherwise

**Python Example**
```Python
retval = dsl_gie_num_in_use_max_set(36)
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
* **Primary and Seconday GIE**
* [Tracker](/docs/api-tracker.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Display Types](/docs/api-display-types.md)
* [branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
