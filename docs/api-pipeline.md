# Pipeline API Refernce

## Pipeline API
* [dsl_pipeline_new](#dsl_pipeline_new)
* [dsl_pipeline_delete](#dsl_pipeline_delete)
* [dsl_pipeline_delete_many](#dsl_pipeline_delete_many)
* [dsl_pipeline_delete_all](#dsl_pipeline_delete_all)
* [dsl_pipeline_list_size](#dsl_pipeline_list_size)
* [dsl_pipeline_list_all](#dsl_pipeline_list_all)

## Return Values
The following return codes are used by the Pipeline API
```C++
#define DSL_RESULT_PIPELINE_RESULT                                  0x11000000
#define DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE                         0x11000001
#define DSL_RESULT_PIPELINE_NAME_NOT_FOUND                          0x11000010
#define DSL_RESULT_PIPELINE_NAME_BAD_FORMAT                         0x11000011
#define DSL_RESULT_PIPELINE_STATE_PAUSED                            0x11000100
#define DSL_RESULT_PIPELINE_STATE_RUNNING                           0x11000101
#define DSL_RESULT_PIPELINE_NEW_EXCEPTION                           0x11000110
#define DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED                    0x11000111
#define DSL_RESULT_PIPELINE_STREAMMUX_SETUP_FAILED                  0x11001000
#define DSL_RESULT_PIPELINE_FAILED_TO_PLAY                          0x11001001
#define DSL_RESULT_PIPELINE_FAILED_TO_PAUSE                         0x11001010
```

## Constructors
### *dsl_pipeline_new*
```C++
DslReturnType dsl_pipeline_new(const char* pipeline);
```
The constructor creates a uniquely named Pipeline. Construction will fail
if the name is currently in use.

**Parameters**
`* pipeline` - unique name for the Pipeline to create.

**Returns**
`DSL_RESULT_SUCCESS` on successfull creation. One of the [Return Values](#return-values) defined above on failure

## Destructors
### *dsl_pipeline_delete*
```C++
DslReturnType dsl_pipeline_delete(const char* pipeline);
```
This destructor deletes a single uniquely named Pipeline. 
All components owned by the pipeline move to a state of `not-in-use`

**Parameters**
* `pipelines` - unique name for the Pipeline to delete

**Returns**
`DSL_RESULT_SUCCESS` on successfull deleteion. One of the [Return Values](#return-values) defined above on failure

### *dsl_pipeline_delete_many*
```C++
DslReturnType dsl_pipeline_delete_many(const char** pipelines);
```
This destructor deletes multiple uniquely named Pipelines. All names are first checked for existance. 
The function returns DSL_RESULT_PIPELINE_NAME_NOT_FOUND on first occurance of not found, before making any deletions. 
All components owned by the Pipelines move to a state of `not-in-use`

**Parameters**
* `pipelines` - a NULL terminated array of uniquely named Pipelines to delete.

**Returns**
`DSL_RESULT_SUCCESS` on successfull deleteion. One of the [Return Values](#return-values) defined above on failure

### *dsl_pipeline_delete_all*
```C++
DslReturnType dsl_pipeline_delete_all();
```
This destructor deletes all Pipelines currently in memory  All components owned by the pipelines move to a state of `not-in-use`

**Returns**
`DSL_RESULT_SUCCESS` on successfull deleteion. One of the [Return Values](#return-values) defined above on failure

## Methods
### *dsl_pipeline_list_size*
```C++
uint dsl_pipeline_list_size();
```
This method returns the Pipeline list size

**Returns** the number of Pipelines currently in memeory

### *dsl_pipeline_list_all*
```C++
const char** dsl_pipeline_list_all();
```
This method returns the list of Pipelines currently in  memory

**Returns** a NULL terminated array of Pipeline (char*) names
