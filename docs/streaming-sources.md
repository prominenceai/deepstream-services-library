# Streaming Sources
DSL supports four types of Streaming Sources
1. Camera Serial Interface ( CSI )
2. Video For Linux ( V4L2 )
2. Uniform Resource Identifier ( URI )
4. Real-time Messaging Protocol ( RTMP )

Sources are created using one of four type-specific constructors. As with all components, Streaming Sources must be uniquely named 
from all other Pipeline components created. Streaming Sources are added to a Pipeline by calling 
[dsl_pipeline_component_add](#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](#dsl_pipeline_component_add_many) 
and removed with [dsl_pipeline_component_remove](#dsl_pipeline_component_remove) or 
[dsl_pipeline_component_remove_many](dsl_pipeline_component_remove_many). The relation ship between Pipelines and Sources is 
one-to-many. Once added to a Pipeline, a Source must be removed before it can used with another. Sources are deleted by calling 
[dsl_component_delete](#dsl_component_delete) or [dsl_component_delete_many](#dsl_component_delete_many)

## Source API
* [dsl_source_csi_new](#dsl_source_csi_new)
* [dsl_source_v4l2_new](#dsl_source_v4l2_new)
* [dsl_source_uri_new](#dsl_source_uri_new)
* [dsl_source_rtmp_new](#dsl_source_rtmp_new)
* [dsl_source_pause](#dsl_source_pause)
* [dsl_source_play](#dsl_source_play)
* [dsl_source_state_is](#dsl_source_state_is)

## Return Values
Streaming Source Methods use the following return codes
```C++
#define DSL_RESULT_SUCCESS                                          0x00000000

#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00100001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00100010
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00100011
#define DSL_RESULT_SOURCE_NEW_EXCEPTION                             0x00100100
#define DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND                     0x00100101
#define DSL_RESULT_SOURCE_STATE_READY                               0x00100110
#define DSL_RESULT_SOURCE_STATE_PAUSED                              0x00100111
#define DSL_RESULT_SOURCE_STATE_PLAYING                             0x00101000
```
## Constructors

### *dsl_source_csi_new*
```C++
DslReturnType dsl_source_csi_new(const char* name,
    guint width, guint height, guint fps_n, guint fps_d);
```    
Creates a new, uniquely named CSI Camera Source object

**Parameters**
* `name` - unique name for the new Source
* `width` - width of the source in pixels
* `height` - height of the source in pixels
* `fps-n` - frames per second fraction numerator
* `fps-d` - frames per second fraction denominator

**Returns**  DSL_RESULT_SOURCE_RESULT

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
#define DSL_CUDADEC_MEMTYPE_DEVICE   0
#define DSL_CUDADEC_MEMTYPE_PINNED   1
#define DSL_CUDADEC_MEMTYPE_UNIFIED  2
```
**Returns**  DSL_RESULT_SOURCE_RESULT

### *dsl_source_rtmp_new*
TBI

### *dsl_source_pause*
TBI

### *dsl_source_pause*
TBI

## Destructors
As with all Pipeline components, ***dsl_component_delete*** is used to destroy a Streaming Source once created. 
```C++
DslReturnType dsl_component_delete(const char* name);
```
