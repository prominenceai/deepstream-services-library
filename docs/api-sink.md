# Sink API
Sinks are the end components for all DSL GStreamer Pipelines. A Pipeline must have at least one sink in use, along with other certain components, to reach a state of Ready. DSL supports six types of Sinks:
* Overlay Sink - renders/overlays video on a Parent display
* Window Sink - renders/overlays video on a Parent XWindow
* File Sink - encodes video to a media container file
* Record Sink - similar to the File sink but with Start/Stop/Duration control and a cache for pre-start buffering. 
* RTSP Sink - streams encoded video on a specifed port
* Fake Sink - consumes/drops all data 

Sinks are created with six type-specific constructors. As with all components, Sinks must be uniquely named from all other components created. 

Sinks are added to a Pipeline by calling [dsl_pipeline_component_add](api-pipeline.md#dsl_pipeline_component_add) or [dsl_pipeline_component_add_many](api-pipeline.md#dsl_pipeline_component_add_many) and removed with [dsl_pipeline_component_remove](api-pipeline.md#dsl_pipeline_component_remove), [dsl_pipeline_component_remove_many](api-pipeline.md#dsl_pipeline_component_remove_many), or [dsl_pipeline_component_remove_all](api-pipeline.md#dsl_pipeline_component_remove_all). 

The relationship between Pipelines and Sinks is one-to-many. Once added to a Pipeline, a Sink must be removed before it can used with another. Sinks are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

There is no (practical) limit to the number of Sinks that can be created, just to the number of Sinks that can be `in use` - a child of a Pipeline - at one time. The in-use limit is imposed by the Jetson Model in use. 

The maximum number of in-use Sinks is set to `DSL_DEFAULT_SINK_IN_USE_MAX` on DSL initialization. The value can be read by calling [dsl_sink_num_in_use_max_get](#dsl_sink_num_in_use_max_get) and updated with [dsl_sink_num_in_use_max_set](#dsl_sink_num_in_use_max_set). The number of Sinks in use by all Pipelines can obtained by calling [dsl_sink_get_num_in_use](#dsl_sink_get_num_in_use). 

## Sink API
**Types:**
* [dsl_record_info](#dsl_record_info)

**Callback Types:**
* [dsl_record_client_listener_cb](#dsl_record_client_listener_cb)

**Constructors:**
* [dsl_sink_overlay_new](#dsl_sink_overlay_new)
* [dsl_sink_window_new](#dsl_sink_window_new)
* [dsl_sink_file_new](#dsl_sink_file_new)
* [dsl_sink_record_new](#dsl_sink_record_new)
* [dsl_sink_rtsp_new](#dsl_sink_rtsp_new)
* [dsl_sink_fake_new](#dsl_sink_fake_new)

**Methods**
* [dsl_sink_render_offsets_get](#dsl_sink_render_offsets_get)
* [dsl_sink_render_offsets_set](#dsl_sink_render_offsets_set)
* [dsl_sink_render_dimensions_get](#dsl_sink_render_dimensions_get)
* [dsl_sink_render_dimensions_set](#dsl_sink_render_dimensions_set)
* [dsl_sink_window_force_aspect_ratio_get](#dsl_sink_window_force_aspect_ratio_get)
* [dsl_sink_window_force_aspect_ratio_set](#dsl_sink_window_force_aspect_ratio_set)
* [dsl_sink_record_session_start](#dsl_sink_record_session_start)
* [dsl_sink_record_session_stop](#dsl_sink_record_session_stop)
* [dsl_sink_record_outdir_get](#dsl_sink_record_outdir_get)
* [dsl_sink_record_outdir_set](#dsl_sink_record_outdir_set)
* [dsl_sink_record_container_get](#dsl_sink_record_container_get)
* [dsl_sink_record_container_set](#dsl_sink_record_container_set)
* [dsl_sink_record_cache_size_get](#dsl_sink_record_cache_size_get)
* [dsl_sink_record_cache_size_set](#dsl_sink_record_cache_size_set)
* [dsl_sink_record_dimensions_get](#dsl_sink_record_dimensions_get)
* [dsl_sink_record_dimensions_set](#dsl_sink_record_dimensions_set)
* [dsl_sink_record_is_on_get](#dsl_sink_record_is_on_get)
* [dsl_sink_record_reset_done_get](#dsl_sink_record_reset_done_get)
* [dsl_sink_encode_video_formats_get](#dsl_sink_encode_video_formats_get)
* [dsl_sink_encode_settings_get](#dsl_sink_encode_settings_get)
* [dsl_sink_encode_settings_set](#dsl_sink_encode_settings_set)
* [dsl_sink_rtsp_server_settings_get](#dsl_sink_rtsp_server_settings_get)
* [dsl_sink_rtsp_encoder_settings_get](#dsl_sink_rtsp_encoder_settings_get)
* [dsl_sink_rtsp_encoder_settings_set](#dsl_sink_rtsp_encoder_settings_set)
* [dsl_sink_pph_add](#dsl_sink_pph_add)
* [dsl_sink_pph_remove](#dsl_sink_pph_remove)
* [dsl_sink_num_in_use_get](#dsl_sink_num_in_use_get)
* [dsl_sink_num_in_use_max_get](#dsl_sink_num_in_use_max_get)
* [dsl_sink_num_in_use_max_set](#dsl_sink_num_in_use_max_set)

## Return Values
The following return codes are used by the Sink API
```C++
#define DSL_RESULT_SINK_RESULT                                      0x00040000
#define DSL_RESULT_SINK_NAME_NOT_UNIQUE                             0x00040001
#define DSL_RESULT_SINK_NAME_NOT_FOUND                              0x00040002
#define DSL_RESULT_SINK_NAME_BAD_FORMAT                             0x00040003
#define DSL_RESULT_SINK_THREW_EXCEPTION                             0x00040004
#define DSL_RESULT_SINK_FILE_PATH_NOT_FOUND                         0x00040005
#define DSL_RESULT_SINK_IS_IN_USE                                   0x00040007
#define DSL_RESULT_SINK_SET_FAILED                                  0x00040008
#define DSL_RESULT_SINK_CODEC_VALUE_INVALID                         0x00040009
#define DSL_RESULT_SINK_CONTAINER_VALUE_INVALID                     0x0004000A
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK                       0x0004000B
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK                0x0004000C
#define DSL_RESULT_SINK_HANDLER_ADD_FAILED                          0x0004000D
#define DSL_RESULT_SINK_HANDLER_REMOVE_FAILED                       0x0004000E
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

---

## Types:

```C
typedef struct dsl_recording_info
{
    uint sessionId;
    const wchar_t* filename;
    const wchar_t* dirpath;
    uint64_t duration;
    uint containerType;
    uint width;
    uint height;
} dsl_recording_info;
```
Structure typedef used to provide recording session information provided to the client on callback

**Fields**
* `sessionId` - the unique sesions id assigned on record start
* `filename` - filename generated for the completed recording. 
* `directory` - path for the completed recording
* `duration` - duration of the recording in milliseconds
* `containerType` - DSL_CONTAINER_MP4 or DSL_CONTAINER_MP4
* `width` - width of the recording in pixels
* `height` - height of the recording in pixels

**Python Example**
```Python

```

## Callback Types:
### *dsl_record_client_listener_cb*
```C++
typedef void* (*dsl_record_client_listener_cb)(void* info, void* user_data);
```
Callback typedef for clients to listen for a notification that a Recording Session has ended.

**Parameters**
* `info` [in] opaque pointer to the session info, see... dsl_capture_info. 
* `user_data` [in] user_data opaque pointer to client's user data, provided by the client  

---

## Constructors
### *dsl_sink_overlay_new*
```C++
DslReturnType dsl_sink_overlay_new(const wchar_t* name, uint display_id,
    uint depth, uint offset_x, uint offset_y, uint width, uint height);
```
The constructor creates a uniquely named Overlay Sink with given offsets and dimensions. Construction will fail if the name is currently in use. 

**Parameters**
* `name` - [in] unique name for the Overlay Sink to create.
* `display_id` - [in] display Id to overlay, 0 = main display
* `depth` - [in] depth of the overlay for the given display Id.  
* `x_offset` - [in] offset in the X direction from the upper left corner of the display in pixels
* `y_offset` - [in] offset in the Y direction from the upper left corner of the display in pixels
* `width` - [in] width of the Overlay Sink in pixels
* `height` - [in] height of the Overlay Sink in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_overlay_new('my-overlay-sink', 0, 0, 200, 100, 1280, 720)
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
* `codec` - [in] on of the [Codec Types](#codec-types) defined above
* `container` - [in] on of the [Video Container Types](#video-container-types) defined above
* `bitrate` - [in] bitrate at which to code the video
* `interval` - [in] frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_file_new('my-file-sink', './my-video.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
```

### *dsl_sink_record_new*
```C++
DslReturnType dsl_sink_record_new(const wchar_t* name, const wchar_t* outdir, uint codec, 
    uint container, uint bitrate, uint interval, dsl_record_client_listener_cb  client_listener);
```
The constructor creates a uniquely named Record Sink. Construction will fail if the name is currently in use. There are three Codec formats - `H.264`, `H.265`, and `MPEG` - and two video container types - `MPEG4` and `MK4` - supported.

Note: the Sink name is used as the filename prefix, followed by session id and NTP time. 

**Parameters**
* `name` - [in] unique name for the Record Sink to create.
* `outdir` - [in] absolute or relative pathspec for the directory to save the recorded video streams.
* `codec` - [in] on of the [Codec Types](#codec-types) defined above
* `container` - [in] on of the [Video Container Types](#video-container-types) defined above
* `bitrate` - [in] bitrate at which to code the video
* `interval` - [in] frame interval at which to code the video. Set to 0 to code every frame
* `client_listener` - [in] client callback funtion of type [dsl_record_client_listener_cb ](#dsl_record_client_listener_cb)to be called when the recording is complete or stoped.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_record_new('my-record-sink', 
    './', DSL_CODEC_H265, DSL_CONTAINER_MPEG, 20000000, 0, my_client_record_complete_cb)
```

<br>

### *dsl_sink_rtsp_new*
```C++
DslReturnType dsl_sink_rtsp_new(const wchar_t* name, const wchar_t* host, 
     uint udp_port, uint rtmp_port, uint codec, uint bitrate, uint interval);
```
The constructor creates a uniquely named RTSP Sink. Construction will fail if the name is currently in use. There are three Codec formats - `H.264`, `H.265`, and `MPEG` supported. The RTSP server is configured when the Pipeline owning Sink as called to Play. The server is then started and attached to the Main Loop context once [dsl_main_loop_run](#dsl_main_loop_run) is called. Once attached, the server can accept connects.

Note: the server Mount point will be derived from the unique RTSP Sink name, for example: 
```
rtsp://my-jetson.local:8554/rtsp-sink-name
```

**Parameters**
* `name` - [in] unique name for the File Sink to create.
* `host` - [in] host name 
* `udp_port` - [in] UDP port setting for the RTSP server.
* `rtsp_port` - [in] RTSP port setting for the server.
* `codec` - [in] on of the [Codec Types](#codec-types) defined above
* `bitrate` - [in] bitrate at which to code the video
* `interval` - [in] frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
 retVal = dsl_sink_rtsp_new('rtsp-sink', host_uri, 5400, 8554, DSL_CODEC_H264, 4000000,0)
```

### *dsl_sink_fake_new*
```C++
DslReturnType dsl_sink_fake_new(const wchar_t* name);
```
The constructor creates a uniquely named Fake Sink. Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the File Sink to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
 retVal = dsl_sink_fake_new('my-fake-sink')
```

---

<br>

## Destructors
As with all Pipeline components, Sinks are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

<br>

---

## Methods
### *dsl_sink_render_offsets_get*
```C++
DslReturnType dsl_sink_render_offsets_get(const wchar_t* name, 
    uint* x_offset, uint* y_offsetY);
```
This service returns the current X and Y offsets for the uniquely named Render Sink; Overlay or Window.

**Parameters**
* `name` - [in] unique name of the Render Sink to query.
* `x_offset` - [out] offset in the X direction in pixels from the upper left most corner of the display.
* `y_offset` - [out] offset in the Y direction in pixels from the upper left most corner of the display.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, x_offset, y_offset = dsl_sink_render_offsets_get('my-overlay-sink')
```

<br>

### *dsl_sink_render_offsets_set*
```C++
DslReturnType dsl_sink_render_offsets_set(const wchar_t* name, 
    uint x_offset, uint y_offset);
```
This service updates the X and Y offsets of a named Render Sink; Overlay or Window. Offsets can be set while the Pipeline or Player is playing, 

**Parameters**
* `name` - [in] unique name of the Render Sink to update.
* `x_offset` - [in] new offset in the X direction in pixels from the upper left most corner of the parent display.
* `y_offset` - [in] new offset in the Y direction in pixels from the upper left most corner of the parent display.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_render_dimensions_set('my-overlay-sink', 100, 100)
```

<br>

### *dsl_sink_render_dimensions_get*
```C++
DslReturnType dsl_sink_render_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service returns the current dimensions for the uniquely named Render Sink; Overlay or Window

**Parameters**
* `name` - [in] unique name of the Render Sink to query.
* `width` - [out] current width of the Render Sink in pixels.
* `height` - [out] current height of the Render Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_sink_render_dimensions_get('my-overlay-sink')
```

<br>

### *dsl_sink_render_dimensions_set*
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
retval = dsl_sink_render_dimensions_set('my-overlay-sink', 1280, 720)
```

<br>

### *dsl_sink_window_force_aspect_ratio_get*
```C++
DslReturnType dsl_sink_window_force_aspect_ratio_get(const wchar_t* name, 
    boolean* force);
```
This service returns the "force-aspect-ratio" property setting for the uniquely named Window Sink. The Sink's aspect ratio will be maintained on Window resize if set. 

**Parameters**
* `name` - [in] unique name of the Window Sink to query.
* `force` - [out] true if the property is set, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, force = dsl_sink_window_force_aspect_ratio_get('my-window-sink')
```

<br>

### *dsl_sink_window_force_aspect_ratio_set*
```C++
DslReturnType dsl_sink_window_force_aspect_ratio_set(const wchar_t* name, 
    boolean force);
```
This service set the "force-aspect-ratio" property for the uniquely named Window Sink.  The Sink's aspect ratio will be maintained on Window resize if set.

**Parameters**
* `name` - [in] unique name of the Window Sink to query.
* `force` - [in] set true to force the aspect ratio on window resize., false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, force = dsl_sink_window_force_aspect_ratio_get('my-window-sink')
```

<br>

### *dsl_sink_record_session_start*
```C++
DslReturnType dsl_sink_record_session_start(const wchar_t* name, uint* session,
    uint start, uint duration, void* client_data);
```
This services starts a new recording session for the named Record Sink

**Parameters**
 * `name` [in] unique of the Record Sink to start the session
 * `session` [out] unique id for the new session on successful start
 * `start` [in] start time in seconds before the current time should be less that the video cache size
 * `duration` [in] in seconds from the current time to record.
 * `client_data` [in] opaque pointer to client data returned  on callback to the client listener function provided on Sink creation

**Returns**
* `DSL_RESULT_SUCCESS` on successful start. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, session = dsl_sink_record_session_start('my-record-sink', 15, 900, None)
```

<br>

### *dsl_sink_record_session_stop*
```C++
DslReturnType dsl_sink_record_session_stop(const wchar_t* name, uint session);
```
This services stops a current recording in session.

**Parameters**
* `name` [in] unique of the Record Sink to stop 
* `session` [in] unique id for the session to stop

**Returns**
* `DSL_RESULT_SUCCESS` on successful Stop. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_record_session_stop('my-record-sink', current_session)
```

<br>

### *dsl_sink_record_outdir_get*
```C++
DslReturnType dsl_sink_record_outdir_get(const wchar_t* name, const wchar_t** outdir);
```
This service returns the video recording output directory. 

**Parameters**
 * `name` [in] name of the Record Sink to query
 * `outdir` - [out] absolute pathspec for the directory to save the recorded video streams.

**Returns**
 * `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, outdir = dsl_sink_record_outdir_get('my-record-sink')
```

<br>

### *dsl_sink_record_outdir_set*
```C++
DslReturnType dsl_sink_record_outdir_set(const wchar_t* name, const wchar_t* outdir);
```
This service sets the video recording output directory. 

**Parameters**
 * `name` [in] name of the Record Sink to update
 * `outdir` - [in] absolute or relative pathspec for the directory to save the recorded video streams.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_record_outdir_set('my-record-sink', './recordings')
```

<br>

### *dsl_sink_record_container_get*
```C++
DslReturnType dsl_sink_record_container_get(const wchar_t* name, uint* container);
```
This service returns the media container type used when recording. 

**Parameters**
 * `name` [in] name of the Record Sink to query
 * `container` - [out] one of the [Video Container Types](#video-container-types) defined above

**Returns**
* `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, container_type = dsl_sink_record_container_get('my-record-sink')
```

<br>

### *dsl_sink_record_container_set*
```C++
DslReturnType dsl_sink_record_container_set(const wchar_t* name,  uint container);
```
This service sets the media container type to use when recording.

**Parameters**
 * `name` [in] name of the Record Sink to update
 * `container` - [in] on of the [Video Container Types](#video-container-types) defined above

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_record_container_set('my-record-sink', DSL_CONTAINER_MP4)
```

<br>

### *dsl_sink_record_cache_size_get*
```C++
DslReturnType dsl_sink_record_cache_size_get(const wchar_t* name, uint* cache_size);
```
This service returns the video recording cache size in units of seconds. A fixed size cache is created when the Pipeline is linked and played. The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC.

**Parameters**
 * `name` [in] name of the Record Sink to query
 * `cache_size` [out] current cache size setting

**Returns**
* `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, cache_size = dsl_sink_record_cache_size_get('my-record-sink')
```

<br>

### *dsl_sink_record_cache_size_set*
```C++
DslReturnType dsl_sink_record_cache_size_set(const wchar_t* name, uint cache_size);
```
This service sets the video recording cache size in units of seconds. A fixed size cache is created when the Pipeline is linked and played. The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC

**Parameters**
 * `name` [in] name of the Record Sink to query
 * `cache_size` [in] new cache size setting to use on Pipeline play

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_record_cache_size_set('my-record-sink', 15)
```

<br>

### *dsl_sink_record_dimensions_get
```C++
DslReturnType dsl_sink_record_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
This service returns the dimensions, width and height, used for the video recordings

**Parameters**
 * `name`[in] name of the Record Sink to query
 * `width`[out] current width of the video recording in pixels
 * `height` [out] current height of the video recording in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_sink_record_dimensions_get('my-record-sink')
```

<br>

### *dsl_sink_record_dimensions_set*
```C++
DslReturnType dsl_sink_record_dimensions_set(const wchar_t* name, uint width, uint height);
```
This services sets the dimensions, width and height, for the video recordings created values of zero (default) indicate no-transcode

**Parameters**
 * `name` [in] name of the Record Sink to update
 * `width` [in] width to set the video recording in pixels
 * `height` [in] height to set the video in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_record_dimensions_set('my-record-sink', 1280, 720)
```

<br>

### *dsl_sink_record_is_on_get*
```C++
DslReturnType dsl_sink_record_is_on_get(const wchar_t* name, boolean* is_on);
```
Thsi service returns the current recording state of the Record Sink.

**Parameters**
 * `name` [in] name of the Record Sink to query
 * `is_on` [out] true if the Record Sink is currently recording a session, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, is_on = dsl_sink_record_is_on_get('my-record-sink')
```

<br>

### *dsl_sink_record_reset_done_get*
```C++
DslReturnType dsl_sink_record_reset_done_get(const wchar_t* name, boolean* reset_done);
```
This service returns the current state of the Reset Done flag

**Parameters**
 * `name` [in] name of the Record Sink to query
 * `reset_done` [out] true if Reset has been done, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, reset_done = dsl_sink_record_reset_done_get('my-record-sink')
```

<br>


### *dsl_sink_encode_video_formats_get*
This service returns the current video codec and container formats for the uniquely named Enocde Sink
```C++
DslReturnType dsl_sink_encode_video_formats_get(const wchar_t* name, 
    uint* codec, uint* container);
```
**Parameters**
* `name` - [in] unique name of the Encode Sink to query.
* `codec` - [out] the current [Codec Type](#Codec Types) setting in use.
* `container` - [out] the current [Video Container Type](#Video Container Types) setting in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, codec, container = dsl_sink_encode_video_formats_get('my-record-sink')
```

<br>

### *dsl_sink_encode_settings_get*
This service returns the current bitrate and interval settings for the uniquely named Encode Sink.
```C++
DslReturnType dsl_sink_file_encode_settings_get(const wchar_t* name, 
    uint* bitrate, uint* interval);
```
**Parameters**
* `name` - [in] unique name of the Encode Sink to query.
* `bitrate` - [out] current bitrate at which to code the video
* `interval` - [out] current frame interval at which to code the video. 0 equals code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, bitrate, interval = dsl_sink_encode_settings_get('my-file-sink')
```

<br>

### *dsl_sink_encode_settings_set*
This service sets the bitrate and interval settings for the uniquely named File or Record Encode Sink. The service will fail if the Encode Sink is currently `in-use`.
```C++
DslReturnType dsl_sink_encode_settings_set(const wchar_t* name, 
    uint bitrate, uint interval);
```
**Parameters**
* `name` - [in] unique name of the Encode Sink to update.
* `bitrate` - [in] new bitrate at which to code the video
* `interval` - [in] new frame interval at which to code the video. Set to 0 to code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_encode_settings_set('my-file-sink', 2000000, 1)
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
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

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

### *dsl_sink_pph_add*
```C++
DslReturnType dsl_sink_pph_add(const wchar_t* name, const wchar_t* handler);
```
This services adds a Pad-Probe-Handler (PPH) to the sink pad of the Named Sink component. The PPH will be invoked on every buffer-ready event for the sink pad. More than one PPH can be added to a single Sink Component.

**Parameters**
 * `name` [in] unique name of the Tee to update
 * `handler` [in] uninque name of the PPH to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_pph_add('my-window-sink', 'my-meter-pph')
```

<br>

### *dsl_sink_pph_remove*
```C++
DslReturnType dsl_sink_pph_remove(const wchar_t* name, const wchar_t* handler);
```
This services removes a Pad-Probe-Handler from a named Sink

**Parameters**
 * `name` [in] unique name of the Sink to update.
 * `handler` [in] unique name of the Pad probe handler to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_pph_remove('my-window-sink', 'my-meter-pph')
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
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIE](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Splitter and Demuxer](/docs/api-tee.md)
* **Sink**
* [Pad Probe Handler](/docs/api-pad-probe-handler.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [SMTP Services](/docs/api-smtp.md)
