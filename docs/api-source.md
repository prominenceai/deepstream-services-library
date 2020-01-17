# Sources
Sources are the head components for all DSL Pipelines. Pipelines must have at least one source in use, among other components, to reach a state of Ready. DSL supports four types of Streaming Sources:

**Camera Sources:**
* Camera Serial Interface ( CSI )
* Universal Serial Bus ( USB )

**Decocde Sources:**
* Uniform Resource Identifier ( URI )
* Real-time Streaming Protocol ( RTSP )

Sources are created using one of four type-specific constructors. As with all components, Streaming Sources must be uniquely named from all other Pipeline components created. 

Sources are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all]((api-pipeline.md#dsl_pipeline_component_remove_all).

When adding multiple sources to a Pipeline, all must have the same `is_live` setting; `true` or `false`. The add services will fail on first exception. 

The relationship between Pipelines and Sources is one-to-many. Once added to a Pipeline, a Source must be removed before it can used with another. Sinks are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all). Calling a delete service on a Source (or any Pipeline component) `in-use` by a Pipeline will fail.

There is no (practical) limit to the number of Sources that can be created, just to the number of Sources that can be `in use`(a child of a Pipeline) at one time. The in-use limit is imposed by the Jetson Model in use. 

The maximum number of `in-use` Sources is set to `DSL_DEFAULT_SOURCE_IN_USE_MAX` on DSL initialization. The value can be read by calling [dsl_source_num_in_use_max_get](#dsl_source_num_in_use_max_get) and updated with [dsl_source_num_in_use_max_set](#dsl_source_num_in_use_max_set). The number of Sources in use by all Pipelines can obtained by calling [dsl_source_get_num_in_use](#dsl_source_get_num_in_use). 


## Source API
**Constructors:**
* [dsl_source_csi_new](#dsl_source_csi_new)
* [dsl_source_usb_new](#dsl_source_usb_new)
* [dsl_source_uri_new](#dsl_source_uri_new)
* [dsl_source_rtsp_new](#dsl_source_rtsp_new)

**methods:**
* [dsl_source_dimensions_get](#dsl_source_dimensions_get)
* [dsl_source_framerate get](#dsl_source_framerate_get)
* [dsl_source_is_live](#dsl_source_is_live)
* [dsl_source_pause](#dsl_source_pause)
* [dsl_source_play](#dsl_source_play)
* [dsl_source_sink_add](#dsl_source_sink_add)
* [dsl_source_sink_remove](#dsl_source_sink_remove)
* [dsl_source_decode_dewarper_add](#dsl_source_decode_dewarper_add)
* [dsl_source_decode_dewarper_remove](#dsl_source_decode_dewarper_remove)
* [dsl_source_num_in_use_get](#dsl_source_num_in_use_get)
* [dsl_source_num_in_use_max_get](#dsl_source_num_in_use_max_get)
* [dsl_source_num_in_use_max_set](#dsl_source_num_in_use_max_set)

## Return Values
Streaming Source Methods use the following return codes, in addition to the general [Component API Return Values](api-component.md#Return Values).
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

## Cuda Decode Memory Types
```C++
#define DSL_CUDADEC_MEMTYPE_DEVICE   0
#define DSL_CUDADEC_MEMTYPE_PINNED   1
#define DSL_CUDADEC_MEMTYPE_UNIFIED  2
```

<br>

## Constructors

### *dsl_source_csi_new*
```C++
DslReturnType dsl_source_csi_new(const wchar_t* source,
    guint width, guint height, guint fps_n, guint fps_d);
```
Creates a new, uniquely named CSI Camera Source object. 

**Parameters**
* `source` - [in] unique name for the new Source
* `width` - [in] width of the source in pixels
* `height` - [in] height of the source in pixels
* `fps-n` - [in] frames per second fraction numerator
* `fps-d` - [in] frames per second fraction denominator

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_csi_new('my-csi-source', 1280, 720, 30, 1)
```

<br>

### *dsl_source_usb_new*
TBI

<br>

### *dsl_source_uri_new*
```C++
DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, boolean is_live,
    uint cudadec_mem_type, uint intra_decode, uint drop_frame_interval);
```
This service creates a new, uniquely named URI Source component

**Parameters**
* `name` - [in] unique name for the new Source
* `uri` - [in] fully qualified URI prefixed with `http://`, `https://`,  or `file://` 
* `is_live` [in] `true` if the URI is a live source, `false` otherwise. File URI's will used a fixed value of `false`
* `cudadec_mem_type` - [in] one of the [Cuda Decode Memory Types](#Cuda Decode Memory Types) defined below
* `intra_decode` - [in] 
* `drop_frame_interval` [in] interval to drop frames at. 0 = decode all frames

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_uri_new('dsl_source_uri_new', '../../test/streams/sample_1080p_h264.mp4',
    False, DSL_CUDADEC_MEMTYPE_DEVICE, 0)
```

<br>

### *dsl_source_rtmp_new*
TBI

## Destructors
As with all Pipeline components, Sources are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

## Methods

### *dsl_source_dimensions_get*
```C++
DslReturnType dsl_source_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
This service returns the width and height values of a named source. CSI and USB Camera sources will return the values they were created with. URI and RTSP sources will return 0's while `not-in` and will be updated once the Source has transitioned to a state of `playing`.

**Parameters**
* `source` - [in]unique name of the Source to play
* `width` - [out] width of the Source in pixels.
* `height` - [out] height of the Source in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_source_dimensions_get('my-uri-source')
```

<br>

### *dsl_source_framerate_get*
```C++
DslReturnType dsl_source_frame_rate_get(const wchar_t* name, uint* fps_n, uint* fps_n);
```
This service returns the fractional frames per second as numerator and denominator for a named source. CSI and USB Camera sources will return the values they were created with. URI and RTSP sources will return 0's while `not-in` and will be updated once the Source has transitioned to a state of `playing`.

**Parameters**
* `source` - [in] unique name of the Source to play.
* `fps_n` - [out] width of the Source in pixels.
* `fps_d` - [out] height of the Source in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, fps_n, fps_d = dsl_source_dimensions_get('my-uri-source')
```

<br>

### *dsl_source_is_live*
```C++
DslReturnType dsl_source_is_live(const wchar_t* source, boolean* is_live);
```
Returns `true` if the Source component's stream is live. CSI and USB Camera sources will always be return `True`.

**Parameters**
* `source` - [in] unique name of the Source to query
* `is_live` - [out] `true` if the source is live, false otherwise

* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, is_live = dsl_source_is_live('my-uri-source')
```

<br>

### *dsl_source_play*
```C++
DslReturnType dsl_source_play(const wchar_t* source);
```
Sets the state of a `paused` Source component to `playing`. This method tries to change the state of an `in-use` Source component to `DSL_STATE_PLAYING`. The current state of the Source component can be obtained by calling [dsl_source_state_is](#dsl_source_state_is). The Pipeline, when transitioning to a state of `DSL_STATE_PLAYING`, will set each of its Sources' 
state to `DSL_STATE_PLAYING`. An individual Source, once playing, can be paused by calling [dsl_source_pause](#dsl_source_pause).

<br>

**Parameters**
* `source` - unique name of the Source to play

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_play('my-source')
```

<br>

### *dsl_source_pause*
```C++
DslReturnType dsl_source_pause(const wchar_t* source);
```
Sets the state of the Source component to Paused. This method tries to change the state of an `in-use` Source component to `GST_STATE_PAUSED`. The current state of the Source component can be obtained by calling [dsl_source_state_is](#dsl_source_state_is).

**Parameters**
* `source` - unique name of the Source to pause

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_play('my-source')
```

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

**Python Example**
```Python
retval, state = dsl_source_state_is('my-source')
```

<br>

### *dsl_source_sink_add*
```C++
DslReturnType dsl_source_sink_add(const wchar_t* source, const wchar_t* sink);
```
This service adds a previously constructed [Sink](api-sink.md) component to a named source. The relationshie of Source to child Sink is one to many. The add service will fail if either of the Source or Sink objects is currently `in-use`.

**Parameters**
* `source` - [in] unique name of the Source to update
* `sink` - [in] unique name of the Sink to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, dsl_source_sink_add('my-uri-source', 'my-window-sink')
```

<br>

### *dsl_source_sink_remove*
```C++
DslReturnType dsl_source_sink_remove(const wchar_t* source, const wchar_t* sink);
```
This service removes a named[Sink](api-sink.md) component from a named source component. The remove service will fail if the Source (and therefore the Sink) object is currently `in-use`.

**Parameters**
* `source` - [in] unique name of the Source to update
* `sink` - [in] unique name of the Sink to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, dsl_source_sink_remove('my-uri-source', 'my-window-sink')
```

<br>

### *dsl_source_decode_dewarper_add*
```C++
DslReturnType dsl_source_decode_dewarper_add(const wchar_t* source, const wchar_t* dewarper);
```
This service adds a previously constructed [Dewarper](api-dewarper.md) component to either a named URI or RTSP source. A source can have at most one Dewarper, and calls to add more will fail. Attempts to add a Dewarper to a Source `in use` will fail. 

**Parameters**
* `source` - [in] unique name of the Source to update
* `dewarper` - [in] unique name of the Dewarper to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, dsl_source_decode_dewarper_add('my-uri-source', 'my-dewarper')
```

<br>

### *dsl_source_decode_dewarper_remove*
```C++
DslReturnType dsl_source_decode_dewarper_remove(const wchar_t* source);
```
This service remove a [Dewarper](api-dewarper.md) component, previously added with [dsl_source_decode_dewarper_add](#dsl_source_decode_dewarper_add) to a named URI source. Calls to remove will fail if the Source is currently without a Dewarper or `in use`.

**Parameters**
* `source` - [in] unique name of the Source to update

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, dsl_source_uri_dewarper_remove('my-uri-source')
```

<br>
### *dsl_source_num_in_use_get*
```C++
uint dsl_source_num_in_use_get();
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
