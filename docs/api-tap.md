# Tap API
Taps are used to "Tap" into a single RTSP source pre-decode so that the original source stream can be preserved. As with all components, Taps must be uniquely named from all other components created. There is only one Tap type at this time, a Record Tap -- similar in operation to the [Record Sink](/docs/api-sink.md) -- with Start/Stop/Duration control and a cache for pre-start buffering. 

### Tap Construction and Destruction
Taps are created by calling a type-specific constructor. Taps are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all). Attempting to delete a Tap while in use will fail. 

### Adding/removing a Tap
Taps are added to an RTSP Source by calling [dsl_source_rtsp_tap_add](/docs/api-source-md#dsl_source_rtsp_tap_add) and removed with [dsl_source_rtsp_tap_remove](/docs/api-source-md#dsl_source_rtsp_tap_remove).  The relationship between Taps and RTSP Sources is one-to-one. Once added to a Source, a Tap must be removed before it can used with another.

Note: Adding a Tap component to a Pipeline or Branch directly will fail.


## Tap API
**Types:**
* [dsl_recording_info](#dsl_recording_info)

**Callback Types:**
* [dsl_record_client_listner_cb](#dsl_record_client_listner_cb)

**Constructors:**
* [dsl_tap_record_new](#dsl_tap_record_new)

**Methods**
* [dsl_tap_record_session_start](#dsl_tap_record_session_start)
* [dsl_tap_record_session_stop](#dsl_tap_record_session_stop)
* [dsl_tap_record_outdir_get](#dsl_tap_record_outdir_get)
* [dsl_tap_record_outdir_set](#dsl_tap_record_outdir_set)
* [dsl_tap_record_container_get](#dsl_tap_record_container_get)
* [dsl_tap_record_container_set](#dsl_tap_record_container_set)
* [dsl_tap_record_cache_size_get](#dsl_tap_record_cache_size_get)
* [dsl_tap_record_cache_size_set](#dsl_tap_record_cache_size_set)
* [dsl_tap_record_dimensions_get](#dsl_tap_record_dimensions_get)
* [dsl_tap_record_dimensions_set](#dsl_tap_record_dimensions_set)
* [dsl_tap_record_is_on_get](#dsl_tap_record_is_on_get)
* [dsl_tap_record_reset_done_get](#dsl_tap_record_reset_done_get)
* [dsl_tap_record_video_player_add](#dsl_tap_record_video_player_add)
* [dsl_tap_record_video_player_remove](#dsl_tap_record_video_player_remove)
* [dsl_tap_record_mailer_add](#dsl_tap_record_mailer_add)
* [dsl_tap_record_mailer_remove](#dsl_tap_record_mailer_remove)


## Return Values
The following return codes are used by the Tap API
```C++
#define DSL_RESULT_TAP_RESULT                                       0x00300000
#define DSL_RESULT_TAP_NAME_NOT_UNIQUE                              0x00300001
#define DSL_RESULT_TAP_NAME_NOT_FOUND                               0x00300002
#define DSL_RESULT_TAP_THREW_EXCEPTION                              0x00300003
#define DSL_RESULT_TAP_IN_USE                                       0x00300004
#define DSL_RESULT_TAP_SET_FAILED                                   0x00300005
#define DSL_RESULT_TAP_COMPONENT_IS_NOT_TAP                         0x00300006
#define DSL_RESULT_TAP_FILE_PATH_NOT_FOUND                          0x00300007
#define DSL_RESULT_TAP_CONTAINER_VALUE_INVALID                      0x00300008
#define DSL_RESULT_TAP_PLAYER_ADD_FAILED                            0x00300009
#define DSL_RESULT_TAP_PLAYER_REMOVE_FAILED                         0x0030000A
#define DSL_RESULT_TAP_MAILER_ADD_FAILED                            0x0030000B
#define DSL_RESULT_TAP_MAILER_REMOVE_FAILED                         0x0030000C
```

## Video Container Types
The following video container types are used by the Record Tap API
```C++
#define DSL_CONTAINER_MPEG4                                         0
#define DSL_CONTAINER_MK4                                           1
```
## Recording Events
The following Event Type identifiers are used by the Recording Tap API
```C++
#define DSL_RECORDING_EVENT_START                                   0
#define DSL_RECORDING_EVENT_END                                     1
```
<br>

## Types
### *dsl_recording_info*
```C
typedef struct dsl_recording_info
{
    uint recording_event;
    uint sessionId;
    const wchar_t* filename;
    const wchar_t* dirpath;
    uint64_t duration;
    uint containerType;
    uint width;
    uint height;
} dsl_recording_info;
```
Structure typedef used to provide recording session information to the client on callback

**Fields**
* `recording_event` - one of DSL_RECORDING_EVENT_START or DSL_RECORDING_EVENT_END
* `sessionId` - the unique sessions id assigned on record start
* `filename` - filename generated for the completed recording. 
* `directory` - path for the completed recording
* `duration` - duration of the recording in milliseconds
* `containerType` - DSL_CONTAINER_MP4 or DSL_CONTAINER_MP4
* `width` - width of the recording in pixels
* `height` - height of the recording in pixels

**Python Example**
```Python
## 
# Function to be called on recording start and complete
## 
def recording_event_listener(session_info_ptr, client_data):
    print(' ***  Recording Event  *** ')
    
    session_info = session_info_ptr.contents
    print('event type: ', recording_info.recording_event)
    print('session_id: ', recording_info.session_id)
    print('filename:   ', recording_info.filename)
    print('dirpath:    ', recording_info.dirpath)
    print('duration:   ', recording_info.duration)
    print('container:  ', recording_info.container_type)
    print('width:      ', recording_info.width)
    print('height:     ', recording_info.height)
    
    return None
```

## Callback Types
### *dsl_record_client_listner_cb*
```C++
typedef void* (*dsl_record_client_listner_cb)(void* info, void* user_data);
```
Callback typedef for a client to listen for the notification that a Recording Session has ended.

**Parameters**
* `info` [in] opaque pointer to the session info, see... NvDsSRRecordingInfo in gst-nvdssr.h 
* `user_data` [in] user_data opaque pointer to client's user data, provided by the client  

---

## Constructors
### *dsl_tap_record_new*
```C++
DslReturnType dsl_tap_record_new(const wchar_t* name, const wchar_t* outdir, 
    uint container, dsl_record_client_listner_cb client_listener);
```
The constructor creates a uniquely named Record Tap. Construction will fail if the name is currently in use. There are two video container types - `MPEG4` and `MK4` - supported.

Note: the Tap name is used as the filename prefix, followed by session id and NTP time. 

**Parameters**
* `name` - [in] unique name for the Record Tap to create.
* `outdir` - [in] absolute or relative pathspec for the directory to save the recorded video streams.
* `container` - [in] one of the [Video Container Types](#video-container-types) defined above
* `client_listener` - [in] client callback function to be called when the recording is complete or stopped.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_new('my-record-tap', './', DSL_CONTAINER_MPEG, my_client_record_complete_cb)
```

<br>

---

## Destructors
As with all Pipeline components, Taps are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all)

<br>

---

## Methods

### *dsl_tap_record_session_start*
```C++
DslReturnType dsl_tap_record_session_start(const wchar_t* name, uint* session,
    uint start, uint duration, void* client_data);
```
This services starts a new recording session for the named Record Tap

**Parameters**
 * `name` [in] unique of the Record Tap to start the session
 * `session` [out] unique id for the new session on successful start
 * `start` [in] start time in seconds before the current time should be less that the video cache size
 * `duration` [in] in seconds from the current time to record.
 * `client_data` [in] opaque pointer to client data returned  on callback to the client listener function provided on Tap creation

**Returns**
* `DSL_RESULT_SUCCESS` on successful start. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, session = dsl_tap_record_session_start('my-record-tap', 15, 900, None)
```

<br>

### *dsl_tap_record_session_stop*
```C++
DslReturnType dsl_tap_record_session_stop(const wchar_t* name, uint session);
```
This services stops a current recording in session.

**Parameters**
* `name` [in] unique of the Record Tap to stop 
* `session` [in] unique id for the session to stop

**Returns**
* `DSL_RESULT_SUCCESS` on successful Stop. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_session_stop('my-record-tap', current_session)
```
<br>

### *dsl_tap_record_outdir_get*
```C++
DslReturnType dsl_tap_record_outdir_get(const wchar_t* name, const wchar_t** outdir);
```
This service returns the video recording output directory. 

**Parameters**
 * `name` [in] name of the Record Tap to query
 * `outdir` - [out] absolute pathspec for the directory to save the recorded video streams.

**Returns**
 * `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, outdir = dsl_tap_record_outdir_get('my-record-tap')
```

<br>

### *dsl_tap_record_outdir_set*
```C++
DslReturnType dsl_tap_record_outdir_set(const wchar_t* name, const wchar_t* outdir);
```
This service sets the video recording output directory. 

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `outdir` - [in] absolute or relative pathspec for the directory to save the recorded video streams.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_outdir_set('my-record-tap', './recordings')
```

<br>

### *dsl_tap_record_container_get*
```C++
DslReturnType dsl_tap_record_container_get(const wchar_t* name, uint* container);
```
This service returns the media container type used when recording. 

**Parameters**
 * `name` [in] name of the Record Tap to query
 * `container` - [out] one of the [Video Container Types](#video-container-types) defined above

**Returns**
* `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, container_type = dsl_tap_record_container_get('my-record-tap')
```

<br>

### *dsl_tap_record_container_set*
```C++
DslReturnType dsl_tap_record_container_set(const wchar_t* name,  uint container);
```
This service sets the media container type to use when recording.

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `container` - [in] on of the [Video Container Types](#video-container-types) defined above

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_container_set('my-record-tap', DSL_CONTAINER_MP4)
```

<br>

### *dsl_tap_record_cache_size_get*
```C++
DslReturnType dsl_tap_record_cache_size_get(const wchar_t* name, uint* cache_size);
```
This service returns the video recording cache size in units of seconds. A fixed size cache is created when the Pipeline is linked and played. The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC.

**Parameters**
 * `name` [in] name of the Record Tap to query
 * `cache_size` [out] current cache size setting

**Returns**
* `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, cache_size = dsl_tap_record_cache_size_get('my-record-tap')
```

<br>

### *dsl_tap_record_cache_size_set*
```C++
DslReturnType dsl_tap_record_cache_size_set(const wchar_t* name, uint cache_size);
```
This service sets the video recording cache size in units of seconds. A fixed size cache is created when the Pipeline is linked and played. The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC

**Parameters**
 * `name` [in] name of the Record Tap to query
 * `cache_size` [in] new cache size setting to use on Pipeline play

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_cache_size_set('my-record-tap', 15)
```

<br>

### *dsl_tap_record_dimensions_get*
```C++
DslReturnType dsl_tap_record_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
This service returns the dimensions, width and height, used for the video recordings

**Parameters**
 * `name`[in] name of the Record Tap to query
 * `width`[out] current width of the video recording in pixels
 * `height` [out] current height of the video recording in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_tap_record_dimensions_get('my-record-tap')
```

<br>

### *dsl_tap_record_dimensions_set*
```C++
DslReturnType dsl_tap_record_dimensions_set(const wchar_t* name, uint width, uint height);
```
This services sets the dimensions, width and height, for the video recordings created values of zero (default) indicate no-transcode

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `width` [in] width to set the video recording in pixels
 * `height` [in] height to set the video in pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_dimensions_set('my-record-tap', 1280, 720)
```

<br>

### *dsl_tap_record_is_on_get*
```C++
DslReturnType dsl_tap_record_is_on_get(const wchar_t* name, boolean* is_on);
```
Thsi service returns the current recording state of the Record Tap.

**Parameters**
 * `name` [in] name of the Record Tap to query
 * `is_on` [out] true if the Record Tap is currently recording a session, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, is_on = dsl_tap_record_is_on_get('my-record-tap')
```

<br>

### *dsl_tap_record_reset_done_get*
```C++
DslReturnType dsl_tap_record_reset_done_get(const wchar_t* name, boolean* reset_done);
```
This service returns the current state of the Reset Done flag

**Parameters**
 * `name` [in] name of the Record Tap to query
 * `reset_done` [out] true if Reset has been done, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, reset_done = dsl_tap_record_reset_done_get('my-record-tap')
```

<br>

### *dsl_tap_record_video_player_add*
```C++
DslReturnType dsl_tap_record_video_player_add(const wchar_t* name, 
    const wchar_t* player)
```
This services adds a [Video Player](/docs/api-player.md), Render or RTSP type, to a named Recording Tap. Once added, each recorded video's file_path will be added (or queued) with the Video Player to be played according to the Players settings. 

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `player` [in] name of the Video Player to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_video_player_add('my-record-tap, 'my-video-render-player')
```

<br>

### *dsl_tap_record_video_player_remove*
```C++
DslReturnType dsl_tap_record_video_player_remve(const wchar_t* name, 
    const wchar_t* player)
```
This services removes a [Video Player](/docs/api-player.md), Render or RTSP type, from a named Recording Tap. 

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `player` [in] player name of the Video Player to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_video_player_remove('my-record-tap', 'my-video-render-player'
```

### *dsl_tap_record_mailer_add*
```C++
DslReturnType dsl_tap_record_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject);
```
This services adds a [Mailer](/docs/api-mailer.md) to a named Recording Tap. Once added, the file_name, location, and specifics of each recorded video will be emailed by the Mailer according to its current settings. 

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `mailer` [in] name of the Mailer to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_mailer_add('my-record-tap, 'my-mailer', 'New recording for my-record-tap')
```

<br>

### *dsl_tap_record_mailer_remove*
```C++
DslReturnType dsl_tap_record_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer)
```
This services removes a [Mailer](/docs/api-mailer.md) from a named Recording Tap. 

**Parameters**
 * `name` [in] name of the Record Tap to update
 * `mailer` [in] name of the Mailer to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_tap_record_mailer_remove('my-record-tap', 'my-mailer')
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* [Source](/docs/api-source.md)
* **TAP**
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Splitter and Demuxer](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
