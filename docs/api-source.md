# Sources
Sources are the head components for all DSL GStreamer Pipeline. Pipeline must have at least one source in use, among other components, to reach a state of Ready. DSL supports four types of Streaming Sources:
1. Camera Serial Interface ( CSI ) - 
2. Universal Serial Bus ( USB )
2. Uniform Resource Identifier ( URI )
4. Real-time Messaging Protocol ( RTMP )

Sources are created using one of four type-specific constructors. As with all components, Streaming Sources must be uniquely named from all other Pipeline components created. 

Sources are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all]((api-pipeline.md#dsl_pipeline_component_remove_all).

The relationship between Pipelines and Sinks is one-to-many. Once added to a Pipeline, a Sink must be removed before it can used with another. Sinks are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

There is no (practical) limit to the number of Sources that can be created, just to the number of Sources that can be `in use` - a child of a Pipeline - at one time. The in-use limit is imposed by the Jetson Model in use. 

The maximum number of in-use Sinks is set to `DSL_DEFAULT_SINK_IN_USE_MAX` on DSL initialization. The value can be read by calling [dsl_source_num_in_use_max_get](#dsl_source_num_in_use_max_get) and updated with [dsl_source_num_in_use_max_set](#dsl_source_num_in_use_max_set). The number of Sources in use by all Pipelines can obtained by calling [dsl_source_get_num_in_use](#dsl_source_get_num_in_use). 


## Source API
* [dsl_source_csi_new](#dsl_source_csi_new)
* [dsl_source_usb_new](#dsl_source_usb_new)
* [dsl_source_uri_new](#dsl_source_uri_new)
* [dsl_source_rtmp_new](#dsl_source_rtmp_new)
* [dsl_source_pause](#dsl_source_pause)
* [dsl_source_play](#dsl_source_play)
* [dsl_source_state_is](#dsl_source_state_is)
* [dsl_source_is_live](#dsl_source_is_live)
* [dsl_source_get_num_in_use](#dsl_source_get_num_in_use)
* [dsl_source_get_num_in_use_max](#dsl_source_get_num_in_use_max)
* [dsl_source_set_num_in_use_max](#dsl_source_set_num_in_use_max)

## Return Values
Streaming Source Methods use the following return codes
```C++
#define DSL_RESULT_SUCCESS                                          0x00000000

#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00020001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00020002
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00020003
#define DSL_RESULT_SOURCE_THREW_EXCEPTION                           0x00020004
#define DSL_RESULT_SOURCE_FILE_NOT_FOUND                            0x00020005
#define DSL_RESULT_SOURCE_NOT_IN_USE                                0x00020006
#define DSL_RESULT_SOURCE_NOT_IN_PLAY                               0x00020007
#define DSL_RESULT_SOURCE_NOT_IN_PAUSE                              0x00020008
#define DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE                    0x00020009
#define DSL_RESULT_SOURCE_CODEC_PARSER_INVALID                      0x0002000A
#define DSL_RESULT_SOURCE_SINK_ADD_FAILED                           0x0002000B
#define DSL_RESULT_SOURCE_SINK_REMOVE_FAILED                        0x0002000C
```

<br>

## Constructors

### *dsl_source_csi_new*
```C++
DslReturnType dsl_source_csi_new(const char* source,
    guint width, guint height, guint fps_n, guint fps_d);
```
Creates a new, uniquely named CSI Camera Source object

**Parameters**
* `source` - unique name for the new Source
* `width` - width of the source in pixels
* `height` - height of the source in pixels
* `fps-n` - frames per second fraction numerator
* `fps-d` - frames per second fraction denominator

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_source_usb_new*
TBI

<br>

### *dsl_source_uri_new*
```C++
DslReturnType dsl_source_uri_new(const char* name, 
    const char* uri, guint cudadec_mem_type, guint intra_decode);
```
**Parameters**
* `name` - unique name for the new Source
* `uri` - fully qualified live URI, or path/file specification`
* `cudadec_mem_type` - one of the MEMTYPE constants defined below.
* `intra_decode` - ?

Cuda decode memory types
```C++
#define DSL_CUDADEC_MEMTYPE_DEVICE   0
#define DSL_CUDADEC_MEMTYPE_PINNED   1
#define DSL_CUDADEC_MEMTYPE_UNIFIED  2
```
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_source_rtmp_new*
TBI

## Destructors
As with all Pipeline components, Sources are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

## Methods

### *dsl_source_play*
```C++
DslReturnType dsl_source_play(const char* source);
```
Sets the state of the Source component to Playing. This method tries to change the state of an `in-use` Source component to `DSL_STATE_PLAYING`. The current state of the Source component can be obtained by calling [dsl_source_state_is](#dsl_source_state_is). The Pipeline, when transitioning to a state of `DSL_STATE_PLAYING`, will set each of its Sources' 
state to `DSL_STATE_PLAYING`. An individual Source, once playing, can be paused by calling [dsl_source_pause](#dsl_source_pause).

<br>

**Parameters**
* `source` - unique name of the Source to play

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_source_pause*
```C++
DslReturnType dsl_source_pause(const char* source);
```
Sets the state of the Source component to Paused. This method tries to change the state of an *in-use* Source component to `GST_STATE_PAUSED`. The current state of the Source component can be obtained by calling [dsl_source_state_is](#dsl_source_state_is).

**Parameters**
* `source` - unique name of the Source to pause

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_source_state_is*
```C++
DslReturnType dsl_source_state_is(const wchar_t* source, uint* state);
```
Returns a Source component's current state as defined by the [DSL_STATE](#DSL_STATE) values.

**Parameters**
* `source` - [in] unique name of the Source to query
* `state` - [out] one of the [DSL_STATE](#DSL_STATE) values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_source_is_live*
```C++
DslReturnType dsl_source_is_live(const wchar_t* source, boolean* is_live);
```
Returns `true` if the Source component's stream is live. CSI and V4L2 Camera sources will always return `True`.

**Parameters**
* `source` - [in] unique name of the Source to query
* `is_live` - [out] `true` if the source is live, false otherwise

* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

<br>

### *dsl_sink_num_in_use_get*
```C++
uint dsl_sink_num_in_use_get();
```
This service returns the total number of all Sinks currently `in-use` by all Pipelines.

**Returns**
* The current number of Sinks `in-use`

**Python Example**
```Python
sources_in_use = dsl_source_num_in_use_get()
```

<br>

### *dsl_source_num_in_use_max_get*
```C++
uint dsl_source_num_in_use_max_get();
```
This service returns the "maximum number of Sources" that can be `in-use` at any one time, defined as `DSL_DEFAULT_SOURCE_NUM_IN_USE_MAX` on service initilization, and can be updated by calling [dsl_source_num_in_use_max_set](#dsl_source_num_in_use_max_set). The actual maximum is impossed by the Jetson model in use. It's the responsibility of the client application to set the value correctly.

**Returns**
* The current max number of Sources that can be `in-use` by all Pipelines at any one time. 

**Python Example**
```Python
max_source_in_use = dsl_source_num_in_use_max_get()
```

<br>

### *dsl_source_num_in_use_max_set*
```C++
boolean dsl_source_num_in_use_max_set(uint max);
```
This service sets the "maximum number of Source" that can be `in-use` at any one time. The value is defined as `DSL_DEFAULT_SOURCE_NUM_IN_USE_MAX` on service initilization. The actual maximum is impossed by the Jetson model in use. It's the responsibility of the client application to set the value correctly.

**Returns**
* `false` if the new value is less than the actual current number of Sources in use, `true` otherwise

**Python Example**
```Python
retval = dsl_source_num_in_use_max_set(24)
```
