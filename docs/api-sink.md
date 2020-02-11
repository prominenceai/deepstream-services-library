# Sink API
Sinks are the end components for all DSL GStreamer Pipelines. A Pipeline must have at least one sink in use, among other components, to reach a state of Ready. DSL supports five types of Sinks:
* Overlay Sink - renders/overlays video on a Parent display
* Window Sink - renders/overlays video on a Parent XWindow
* File Sink - encodes video to a media container file
* RTSP Sink - streams encoded video on a specifed port
* Fake Sink - consumes/drops all data 

Sinks are created with five type-specific constructors. As with all components, Sinks must be uniquely named from all other components created. 

Sinks are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all]((api-pipeline.md#dsl_pipeline_component_remove_all). 

The relationship between Pipelines and Sinks is one-to-many. Once added to a Pipeline, a Sink must be removed before it can used with another. Sinks are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

There is no (practical) limit to the number of Sinks that can be created, just to the number of Sinks that can be `in use` - a child of a Pipeline - at one time. The in-use limit is imposed by the Jetson Model in use. 

The maximum number of in-use Sinks is set to `DSL_DEFAULT_SINK_IN_USE_MAX` on DSL initialization. The value can be read by calling [dsl_sink_num_in_use_max_get](#dsl_sink_num_in_use_max_get) and updated with [dsl_sink_num_in_use_max_set](#dsl_sink_num_in_use_max_set). The number of Sinks in use by all Pipelines can obtained by calling [dsl_sink_get_num_in_use](#dsl_sink_get_num_in_use). 

## Sink API
**Constructors:**
* [dsl_sink_overlay_new](#dsl_sink_overlay_new)
* [dsl_sink_window_new](#dsl_sink_window_new)
* [dsl_sink_file_new](#dsl_sink_file_new)
* [dsl_sink_rtsp_new](#dsl_sink_rtsp_new)
* [dsl_sink_fake_new](#dsl_sink_fake_new)

**Methods**
* [dsl_sink_overlay_offsets_get](#dsl_sink_overlay_offsets_get)
* [dsl_sink_overlay_offsets_set](#dsl_sink_overlay_offsets_set)
* [dsl_sink_overlay_dimensions_get](#dsl_sink_overlay_dimensions_get)
* [dsl_sink_overlay_dimensions_set](#dsl_sink_overlay_dimensions_set)
* [dsl_sink_window_offsets_get](#dsl_sink_window_offsets_get)
* [dsl_sink_window_offsets_set](#dsl_sink_window_offsets_set)
* [dsl_sink_window_dimensions_get](#dsl_sink_window_dimensions_get)
* [dsl_sink_window_dimensions_set](#dsl_sink_window_dimensions_set)
* [dsl_sink_file_video_formats_get](#dsl_sink_file_video_formats_get)
* [dsl_sink_file_encoder_settings_get](#dsl_sink_file_encoder_settings_get)
* [dsl_sink_file_encoder_settings_set](#dsl_sink_file_encoder_settings_set)
* [dsl_sink_rtsp_server_settings_get](#dsl_sink_rtsp_server_settings_get)
* [dsl_sink_rtsp_encoder_settings_get](#dsl_sink_rtsp_code_settings_get)
* [dsl_sink_rtsp_encoder_settings_set](#dsl_sink_rtsp_code_settings_set)
* [dsl_sink_num_in_use_get](#dsl_sink_num_in_use_get)
* [dsl_sink_num_in_use_max_get](#dsl_sink_num_in_use_max_get)
* [dsl_sink_num_in_use_max_set](#dsl_sink_num_in_use_max_set)

## Return Values
The following return codes are used by the Sink API
```C++
#define DSL_RESULT_SINK_NAME_NOT_UNIQUE                             0x00040001
#define DSL_RESULT_SINK_NAME_NOT_FOUND                              0x00040002
#define DSL_RESULT_SINK_NAME_BAD_FORMAT                             0x00040003
#define DSL_RESULT_SINK_THREW_EXCEPTION                             0x00040004
#define DSL_RESULT_SINK_FILE_PATH_NOT_FOUND                         0x00040005
#define DSL_RESULT_SINK_IS_IN_USE                                   0x00040007
#define DSL_RESULT_SINK_SET_FAILED                                  0x00040008
```
## Codec Types
The following codec types are used by the Sink API
```C++
#define DSL_CODEC_H264                                              0
#define DSL_CODEC_H265                                              1
#define DSL_CODEC_MPEG4                                             2
```
## Video Container Types
The following video container types are used by the File Sink API
```C++
#define DSL_CONTAINER_MPEG4                                         0
#define DSL_CONTAINER_MK4                                           1
```
<br>

## Constructors
### *dsl_sink_overlay_new*
```C++
DslReturnType dsl_sink_overlay_new(const wchar_t* name, 
    uint x_offset, uint y_offset, uint width, uint height);
```
The constructor creates a uniquely named Overlay Sink with given offsets and dimensions. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Overlay Sink to create.
* `x_offset` - [in] offset in the X direction from the upper left corner of the display in pixels
* `y_offset` - [in] offset in the Y direction from the upper left corner of the display in pixels
* `width` - [in] width of the Overlay Sink in pixels
* `height` - [in] height of the Overlay Sink in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_overlay_new('my-overlay-sink', 0, 0, 1280, 720)
```

<br>

### *dsl_sink_window_new*
```C++
DslReturnType dsl_sink_window_new(const wchar_t* name, 
    uint x_offset, uint y_offset, uint width, uint height);
```
The constructor creates a uniquely named Window Sink with given offsets and dimensions. Construction will fail if the name is currently in use. Window Sinks are used render video onto an XWindow's Display. See [Pipeline XWindow Support](api-pipeline.md#pipeline-xwindow-support) for more information.

**Parameters**
* `name` - [in] unique name for the Window Sink to create.
* `x_offset` - [out] offset in the X direction in pixels from the upper left most corner of the parent XWindow.
* `y_offset` - [out] offset in the Y direction in pixels from the upper left most corner of the parent XWindow.
* `width` - [in] width of the Overlay Sink in pixels
* `height` - [in] height of the Overlay Sink in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_window_new('my-window-sink', 0, 0, 1280, 720)
```

<br>

### *dsl_sink_file_new*
```C++
DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* filepath, 
     uint codec, uint container, uint bit_rate, uint interval);
```
The constructor creates a uniquely named File Sink. Construction will fail if the name is currently in use. There are three Codec formats - `H.264`, `H.265`, and `MPEG` - and two video container types - `MPEG4` and `MK4` - supported.

**Parameters**
* `name` - [in] unique name for the File Sink to create.
* `filepath` - [in] absolute or relative filespec for the media file to write to.
* `codec` - [in] on of the [Codec Types](#Codec Types) defined above
* `container` - [in] on of the [Video Container Types](#Video Container Types) defined above
* `bitrate` - [in] bitrate at which to code the video
* `interval` - [in] frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_file_new('my-file-sink', './my-video.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
```

<br>

### *dsl_sink_rtsp_new*
```C++
DslReturnType dsl_sink_rtsp_new(const wchar_t* name, uint port, uint codec, uint bit_rate, uint interval);
```
The constructor creates a uniquely named RTSP Sink. Construction will fail if the name is currently in use. There are three Codec formats - `H.264`, `H.265`, and `MPEG` supported. The RTSP server is configured when the Pipeline owning Sink as called to Play. The server is then started and attached to the Main Loop context once [dsl_main_loop_run](#dsl_main_loop_run) is called. Once attached, the server can accept connects.

Note: the server Mount point will be derived from the unique RTSP Sink name, for example: 
```
http://localhost::8080/my-sink-name
```

**Parameters**
* `name` - [in] unique name for the File Sink to create.
* `port` - [in] UDP port setting for the RTSP server.
* `codec` - [in] on of the [Codec Types](#Codec Types) defined above
* `bitrate` - [in] bitrate at which to code the video
* `interval` - [in] frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_rtsp_new('my-rtsp-sink', 8050, DSL_CODEC_H265, 200000, 0)
```

<br>

## Destructors
As with all Pipeline components, Sources are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

<br>

## Methods
### *dsl_sink_overlay_offsets_get*
```C++
DslReturnType dsl_sink_overlay_offsets_get(const wchar_t* name, 
    uint* x_offset, uint* y_offsetY);
```
This service returns the current X and Y offsets for the uniquely named Overlay Sink.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to query.
* `x_offset` - [out] offset in the X direction in pixels from the upper left most corner of the display.
* `y_offset` - [out] offset in the Y direction in pixels from the upper left most corner of the display.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, x_offset, y_offset = dsl_sink_overlay_offsets_get('my-overlay-sink')
```

<br>

### *dsl_sink_overlay_offsets_set*
```C++
DslReturnType dsl_sink_overlay_offsets_set(const wchar_t* name, 
    uint x_offset, uint y_offset);
```
This service updates the X and Y offsets of a named Overlay Sink. This service will fail if the Overlay Sink is currently `in-use`.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to update.
* `x_offset` - [in] new offset in the X direction in pixels from the upper left most corner of the parent display.
* `y_offset` - [in] new offset in the Y direction in pixels from the upper left most corner of the parent display.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_overlay_dimensions_set('my-overlay-sink', 100, 100)
```

<br>

### *dsl_sink_overlay_dimensions_get*
```C++
DslReturnType dsl_sink_overlay_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service returns the current dimensions for the uniquely named Overlay Sink.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to query.
* `width` - [out] width of the Overlay Sink in pixels.
* `height` - [out] height of the Overlay Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_sink_overlay_dimensions_get('my-overlay-sink')
```

<br>

### *dsl_sink_overlay_dimensions_set*
```C++
DslReturnType dsl_sink_overlay_dimensions_set(const wchar_t* name, 
    uint width, uint height);
```
This service updates the dimensions of a named Overlay Sink. This service will fail if the Overlay Sink is currently `in-use`.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to update.
* `width` - [in] new width setting for the Overlay Sink.
* `height` - [in] new height setting to use on XWindow creation in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_overlay_dimensions_set('my-overlay-sink', 1280, 720)
```

<br>

### *dsl_sink_window_offsets_get*
```C++
DslReturnType dsl_sink_window_offsets_get(const wchar_t* name, 
    uint* x_offset, uint* y_offset);
```
This service returns the current X and Y offsets for the uniquely named Overlay Sink.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to query.
* `x_offset` - [out] offset in the X direction in pixels from the upper left most corner from the parent XWindow.
* `y_offset` - [out] offset in the Y direction in pixels from the upper left most corner from the parent XWindow.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, x_offset, y_offset = dsl_sink_window_offsets_get('my-overlay-sink')
```

<br>

### *dsl_sink_window_offsets_set*
```C++
DslReturnType dsl_sink_window_offsets_set(const wchar_t* name, 
    uint* x_offset, uint* y_offset);
```
This service updates the X and Y offsets of a named Overlay Sink. This service will fail if the Overlay Sink is currently `in-use`.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to update.
* `x_offset` - [in] new offset the X direction in pixels from the upper left most corner of the parent XWindow.
* `y_offset` - [in] new offset the Y direction in pixels from the upper left most corner of the parent XWindow.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_offsets_set('my-window-sink', 100, 100)
```

<br>

### *dsl_sink_window_dimensions_get*
```C++
DslReturnType dsl_sink_window_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service returns the current dimensions for the uniquely named Window Sink.

**Parameters**
* `name` - [in] unique name of the Window Sink to query.
* `width` - [out] width of the Window Sink in pixels.
* `height` - [out] height of the Window Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_sink_window_dimensions_get('my-window-sink')
```

<br>

### *dsl_sink_window_dimensions_set*
```C++
DslReturnType dsl_sink_window_dimensions_set(const wchar_t* name, 
    uint width, uint height);
```
This service updates the dimensions of a named Window Sink. This service will fail if the Window Sink is currently `in-use`.

**Parameters**
* `name` - [in] unique name of the Window Sink to update.
* `width` - [in] new width setting to use for the Window Sink in pixels.
* `height` - [in] new height setting to use for the Window Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_dimensions_set('my-window-sink', 1280, 720)
```

<br>

### *dsl_sink_file_video_formats_get*
This service returns the current video codec and container formats for the uniquely named File Sink.
```C++
DslReturnType dsl_sink_file_video_formats_get(const wchar_t* name, 
    uint* codec, uint* container);
```
**Parameters**
* `name` - [in] unique name of the File Sink to query.
* `codec` - [out] the current [Codec Type](#Codec Types) setting in use.
* `container` - [out] the current [Video Container Type](#Video Container Types) setting in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, codec, container = dsl_sink_file_video_formats_get('my-window-sink')
```

<br>

### *dsl_sink_file_encoder_settings_get*
This service returns the current bitrate and interval settings for the uniquely named File Sink.
```C++
DslReturnType dsl_sink_file_encoder_settings_get(const wchar_t* name, 
    uint* bitrate, uint* interval);
```
**Parameters**
* `name` - [in] unique name of the File Sink to query.
* `bitrate` - [out] current bitrate at which to code the video
* `interval` - [out] current frame interval at which to code the video. 0 equals code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, bitrate, interval = dsl_sink_file_encoder_settings_get('my-file-sink')
```

<br>

### *dsl_sink_file_encoder_settings_set*
This service sets the bitrate and interval settings for the uniquely named File Sink. The service will fail if the File Sink is currently `in-use`.
```C++
DslReturnType dsl_sink_file_encoder_settings_set(const wchar_t* name, 
    uint bitrate, uint interval);
```
**Parameters**
* `name` - [in] unique name of the File Sink to update.
* `bitrate` - [in] new bitrate at which to code the video
* `interval` - [in] new frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_file_encoder_settings_set('my-file-sink', 2000000, 1)
```

<br>

### *dsl_sink_rtsp_server_settings_get*
This service returns the current RTSP video codec and Port settings for the uniquely named RTSP Sink.
```C++
DslReturnType dsl_sink_rtsp_server_settings_get(const wchar_t* name, 
    uint* codec, uint* port);
```
**Parameters**
* `name` - [in] unique name of the RTSP Sink to query.
* `codec` - [out] the current [Codec Type](#Codec Types) setting in use.
* `port` - [out] the current Port number setting in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, codec, port = dsl_sink_rtsp_server_settings_get('my-rtsp-sink')
```

<br>

### *dsl_sink_rtsp_encoder_settings_get*
This service returns the current bitrate and interval settings for the uniquely named RTSP Sink.
```C++
DslReturnType dsl_sink_rtsp_encoder_settings_get(const wchar_t* name, 
    uint* bitrate, uint* interval);
```
**Parameters**
* `name` - [in] unique name of the RTSP Sink to query.
* `bitrate` - [out] current bitrate at which to code the video
* `interval` - [out] current frame interval at which to code the video. 0 equals code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, bitrate, interval = dsl_sink_rtsp_encoder_settings_get('my-rtsp-sink')
```

<br>

### *dsl_sink_rtsp_encoder_settings_set*
This service sets the bitrate and interval settings for the uniquely named RTSP Sink. The service will fail if the File Sink is currently `in-use`.
```C++
DslReturnType dsl_sink_rtsp_encoder_settings_set(const wchar_t* name, 
    uint bitrate, uint interval);
```
**Parameters**
* `name` - [in] unique name of the RTSP Sink to update.
* `bitrate` - [in] new bitrate at which to code the video
* `interval` - [in] new frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_rtsp_encoder_settings_set('my-rtsp-sink', 4000000, 1)
```

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
sinks_in_use = dsl_sink_num_in_use_get()
```

<br>

### *dsl_sink_num_in_use_max_get*
```C++
uint dsl_sink_num_in_use_max_get();
```
This service returns the "maximum number of Sinks" that can be `in-use` at any one time, defined as `DSL_DEFAULT_SINK_NUM_IN_USE_MAX` on service initilization, and can be updated by calling [dsl_sink_num_in_use_max_set](#dsl_sink_num_in_use_max_set). The actual maximum is impossed by the Jetson model in use. It's the responsibility of the client application to set the value correctly.

**Returns**
* The current max number of Sinks that can be `in-use` by all Pipelines at any one time. 

**Python Example**
```Python
max_sinks_in_use = dsl_sink_num_in_use_max_get()
```

<br>

### *dsl_sink_num_in_use_max_set*
```C++
boolean dsl_sink_num_in_use_max_set(uint max);
```
This service sets the "maximum number of Sinks" that can be `in-use` at any one time. The value is defined as `DSL_DEFAULT_SINK_NUM_IN_USE_MAX` on service initilization. The actual maximum is impossed by the Jetson model in use. It's the responsibility of the client application to set the value correctly.

**Returns**
* `false` if the new value is less than the actual current number of Sinks in use, `true` otherwise

**Python Example**
```Python
retval = dsl_sink_num_in_use_max_set(24)
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* **Sink**
* [Component](/docs/api-component.md)

