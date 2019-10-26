# Sources
Streaming Sources are the head component for any DSL GStreamer Pipeline. At minimum, a Pipeline must have one source in use, among other components, to reach a state of Ready. DSL supports four types of Streaming Sources:
1. Camera Serial Interface ( CSI )
2. Video For Linux ( V4L2 )
2. Uniform Resource Identifier ( URI )
4. Real-time Messaging Protocol ( RTMP )

Sources are created using one of four type-specific constructors. As with all components, Streaming Sources must be uniquely named from all other Pipeline components created. 

Streaming Sources are added to a Pipeline by calling [dsl_pipeline_component_add](#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](#dsl_pipeline_component_remove) or [dsl_pipeline_component_remove_many](dsl_pipeline_component_remove_many). 

The relationship between Pipelines and Sources is one-to-many. Once added to a Pipeline, a Source must be removed before it can used with another. Sources are deleted by calling [dsl_component_delete](#dsl_component_delete) or [dsl_component_delete_many](#dsl_component_delete_many)

There is no (practical) limit to the number of Sources that can be created, just to the number of Sources that can be **in use** - a child of a Pipeline - at one time. The in-use limit is imposed by the Jetson Model in use. 

The maximum number of in-use sources is set to `DSL_DEFAULT_SOURCE_IN_USE_MAX` on DSL initialization. The value can be read by calling [dsl_source_get_num_in_use_max](#dsl_source_get_num_in_use_max) and updated with [dsl_source_set_num_in_use_max](#dsl_source_set_num_in_use_max). The number of Sources in use by all Pipelines can obtained by calling [dsl_source_get_num_in_use](#dsl_source_get_num_in_use). 


## Source API
* [dsl_source_csi_new](#dsl_source_csi_new)
* [dsl_source_v4l2_new](#dsl_source_v4l2_new)
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

#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00100001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00100010
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00100011
#define DSL_RESULT_SOURCE_NEW_EXCEPTION                             0x00100100
#define DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND                     0x00100101
#define DSL_RESULT_SOURCE_STATE_READY                               0x00100110
#define DSL_RESULT_SOURCE_STATE_PAUSED                              0x00100111
#define DSL_RESULT_SOURCE_STATE_PLAYING                             0x00101000
```
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

**Returns**  - ```DSL_RESULT_SUCCESS``` on success. One of ```DSL_RESULT_SOURCE_RESULT``` values on failure.

### *dsl_source_v4l2_new*
TBI

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
**Returns**  - ```DSL_RESULT_SUCCESS``` on success. One of ```DSL_RESULT_SOURCE_RESULT``` values on failure.

### *dsl_source_rtmp_new*
TBI

## Destructors
As with all Pipeline components, Sources are deleted by calling [dsl_component_delete](#dsl_component_delete) or [dsl_component_delete_many](#dsl_component_delete_many).

## Methods

### *dsl_source_play*
```C++
DslReturnType dsl_source_play(const char* source);
```
Sets the state of the Source component to Playing. This method tries to change the state of an *in-use* Source component to `DSL_STATE_PLAYING`. The current state of the Source component can be obtained by calling [dsl_source_state_is](#dsl_source_state_is). The Pipeline, when transitioning to a state of `DSL_STATE_PLAYING`, will set each of its Sources' 
state to `DSL_STATE_PLAYING`. An individual Source, once playing, can be paused by calling [dsl_source_pause](#dsl_source_pause).

**Parameters**
* `source` - unique name of the Source to play

**Returns**  - ```DSL_RESULT_SUCCESS``` on success. One of ```DSL_RESULT_SOURCE_RESULT``` values on failure.

### *dsl_source_pause*
```C++
DslReturnType dsl_source_pause(const char* source);
```
Sets the state of the Source component to Paused. This method tries to change the state of an *in-use* Source component to `GST_STATE_PAUSED`. The current state of the Source component can be obtained by calling [dsl_source_state_is](#dsl_source_state_is).

**Parameters**
* `source` - unique name of the Source to pause

**Returns**  - ```DSL_RESULT_SUCCESS``` on success. One of ```DSL_RESULT_SOURCE_RESULT``` values on failure.

### *dsl_source_state_is*
```C++
uint dsl_source_state_is(const char* source);
```
Returns a Source component's current state as defined by the [DSL_STATE](#DSL_STATE) values.

**Parameters**
* `source` - unique name of the Source to query

**Returns**  - One of [DSL_STATE](#) values.


### *dsl_source_is_live*
```C++
boolean dsl_source_is_live(const char* source);
```
Returns `True` if the Source component's stream is live. CSI and V4L2 Camera sources will always return `True`.

**Parameters**
* `source` - unique name of the Source to query

**Returns**  - `True` if Source component's stream is live, `False` otherwise


### *dsl_source_get_num_in_use*
```C++
uint dsl_source_get_num_in_use();
```
Queries DSL for the number of Source components currently in-use by all Pipelines.

**Returns**  - The number of Source components in use.

### *dsl_source_get_num_in_use_max*
```C++
uint dsl_source_get_num_in_use_max();
```
Queries DSL for the number maximum number of in-use sources that can be serviced, as limited by Hardware, 

**Returns**  - The maximum number of Source components that can be in-use.

### *dsl_source_set_num_in_use_max*
```C++
void dsl_source_set_num_in_use_max(uint max);
```
Sets the maximum number of in-use sources that can be serviced. **Note!** it is the responsibility of the client to set max in-use value within the limit imposed by Hardware.

**Parameters**
* `max` - the value for the new maximum
