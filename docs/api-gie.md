# GST Infer Engine (GIE) API Refernce
Primary and Secondary GIE's are constructed by calling [dsl_gie_primary_new](#dsl_gie_primary_new) and 
[dsl_gie_secondary_new](#dsl_gie_secondary_new) respectively.

## Primary and Secondary GIE API
* [dsl_gie_primary_new](#dsl_gie_primary_new)
* [dsl_gie_secondary_new](#dsl_gie_secondary_new)
* [dsl_gie_infer_config_file_get](#dsl_gie_infer_config_file_get)
* [dsl_gie_infer_config_file_set](#dsl_gie_infer_config_file_set)
* [dsl_gie_model_engine_file_get](#dsl_gie_model_engine_file_get)
* [dsl_gie_model_engine_file_set](#dsl_gie_model_engine_file_set)
* [dsl_gie_interval_get](#dsl_gie_interval_get)
* [dsl_gie_interval_set](#dsl_gie_interval_set)
* [dsl_gie_secondary_infer_on_get](#dsl_gie_secondary_infer_on_get)
* [dsl_gie_secondary_infer_on_set](#dsl_gie_secondary_infer_on_set)

<br>

## Constructors
**Python Example**
```Python
# Filespecs for the Primary GIE
pgie_config_file = './configs/config_infer_primary_nano.txt'
primaryModelEngineFile = './models/Primary_Detector_Nano/resnet10.caffemodel.engine'

# Filespecs for the Secondary GIE
secondaryInferConfigFile = './configs/config_infer_secondary_carcolor
secondaryModelEngineFile = './models/Secondary_CarColor/resnet18.caffemodel'

# New Primary GIE using the filespecs above, with interval set to  0
retval = dsl_gie_primary_new('resnet10-caffemodel-pgie', inferConfigFile, modelEngineFile, 0)

if retval != DSL_RETURN_SUCCESS:
    print(retval)
    # handle error condition
```

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
