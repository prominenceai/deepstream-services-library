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
* [dsl_gie_primary_new](#dsl_gie_primary_new)
* [dsl_gie_secondary_new](#dsl_gie_secondary_new)
* [dsl_gie_infer_config_file_get](#dsl_gie_infer_config_file_get)
* [dsl_gie_infer_config_file_set](#dsl_gie_infer_config_file_set)
* [dsl_gie_model_engine_file_get](#dsl_gie_model_engine_file_get)
* [dsl_gie_model_engine_file_set](#dsl_gie_model_engine_file_set)
* [dsl_gie_enable_raw_ouput](#dsl_gie_enable_raw_ouput)
* [dsl_gie_disable_raw_output](#dsl_gie_enable_raw_ouput)
* [dsl_gie_interval_get](#dsl_gie_interval_get)
* [dsl_gie_primary_interval_set](#dsl_gie_prmary_interval_set)
* [dsl_gie_secondary_infer_on_get](#dsl_gie_secondary_infer_on_get)
* [dsl_gie_secondary_infer_on_set](#dsl_gie_secondary_infer_on_set)

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
* `name` - unique name for the Primary GIE to create.
* `infer_config_file` - relative or absolute file path/name for the infer config file to load
* `model_engine_file` - relative or absolute file path/name for the model engine file to load
* `interval` - frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_gie_secondary_new*
```C++
DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval, const wchar_t* infer_on_gie_name);
```

**Parameters**
* `name` - unique name for the Primary GIE to create.
* `infer_config_file` - relative or absolute file path/name for the infer config file to load
* `model_engine_file` - relative or absolute file path/name for the model engine file to load
* `interval` - frame interval to infer on

**Returns**
`DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

## Methods
### *dsl_gie_infer_config_file_get*
```C++
DslReturnType dsl_gie_infer_config_file_get(const wchar_t* name, const wchar_t** infer_config_file);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `infer_config_file` - returns the absolute file path/name for the infer config file in use

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists. `DSL_RESULT_GIE_NAME_NOT_FOUND` otherwise

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

<br>

### *dsl_gie_interval_get*
```C++
DslReturnType dsl_gie_interval_get(const wchar_t* name, uint* interval);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to query.
* `interval` - returns the current frame interval in use by the named GIE

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists. `DSL_RESULT_GIE_NAME_NOT_FOUND` otherwise

<br>

### *dsl_gie_interval_set*
```C++
DslReturnType dsl_gie_interval_set(const wchar_t* name, uint interval);
```
**Parameters**
* `name` - unique name of the Primary or Secondary GIE to update.
* `interval` - frame interval in use by the named GIE

**Returns**
`DSL_RESULT_SUCCESS` if the GIE exists, and the interval was within range. one of the 
[Return Values](#return-values) defined above on failure

---

## API Reference
* [Source](source-api.md)
* [Dewarper](api-dewarper.md)
* **Primary and Seconday GIE**
* [Tracker](api-tracker.md)
* [On-Screen Display](api-osd.md)
* [Tiler](api-tiler.md)
* [Sink](api-sink.md)
* [Component](api-component.md)
* [Pipeline](api-pipeline.md)
