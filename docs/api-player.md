# Player API Reference
Players are simplified/specialized Pipelines, created with a single [Source](/docs/api-source.md) and single [Sink](/docs/api-sink.md) component. Once created, the Source and Sink attributes can be updated, however, the components cannot be removed or replaced.

#### Player Construction and Destruction
There are three types of Players that can be created. 
1. The **Basic Player** that can be created with any one Source and Sink by calling [dsl_player_new](#dsl_player_new). Both the Source and Sink components must exists at the time of Player construction
2. Two Render Players that create and manage their own Sink and Source Components
   1. An **Image Render Player**, created by calling  [dsl_player_render_image_new](#dsl_player_render_image_new), that creates an Image Source and Render Sink; Overlay or Window as specified.
   1. A **Video Render Player**, created by calling  [dsl_player_render_video_new](#dsl_player_render_video_new), that creates a File Source and Render Sink of choice.

Players are destructed by calling [dsl_player_delete](#dsl_player_delete) or [dsl_player_delete_all](#dsl_player_delete_all).

#### Playing, Pausing and Stopping a Players
Players can be `played` by calling [dsl_player_play](#dsl_player_play), `paused` (non-live sources only) by calling [dsl_player_pause](#dsl_player_pause) and `stopped` by calling [dsl_player_stop](#dsl_player_stop).

#### Player Client-Listener Notifications
Clients can be notified of **Player Termination** on **End-of-Stream** and **Window Deletion** events by registering/deregistering one or more callback functions with [dsl_player_termination_event_listener_add](#dsl_player_termination_event_listener_add) / [dsl_player_termination_event_listener_remove](#dsl_player_termination_event_listener_remove). 

---
## Player API
**Client CallBack Typedefs**
* [dsl_player_termination_event_listener_cb](#dsl_player_termination_event_listener_cb)

**Constructors**
* [dsl_player_new](#dsl_player_new)
* [dsl_player_render_image_new](#dsl_player_render_image_new)
* [dsl_player_render_video_new](#dsl_player_render_video_new)

**Destructors**
* [dsl_player_delete](#dsl_player_delete)
* [dsl_player_delete_all](#dsl_player_delete_all)

**Methods**
* [dsl_player_render_file_path_get](#dsl_player_render_file_path_get)
* [dsl_player_render_file_path_set](#dsl_player_render_file_path_set)
* [dsl_player_render_file_path_queue](#dsl_player_render_file_path_queue)
* [dsl_player_render_offsets_get](#dsl_player_render_offsets_get)
* [dsl_player_render_offsets_set](#dsl_player_render_offsets_set)
* [dsl_player_render_zoom_get](#dsl_player_render_zoom_get)
* [dsl_player_render_zoom_set](#dsl_player_render_zoom_set)
* [dsl_player_render_image_timeout_get](#dsl_player_render_image_timeout_get)
* [dsl_player_render_image_timeout_set](#dsl_player_render_image_timeout_set)
* [dsl_player_render_video_repeat_enabled_get](#dsl_player_render_video_repeat_enabled_get)
* [dsl_player_render_video_repeat_enabled_set](#dsl_player_render_video_repeat_enabled_set)
* [dsl_player_termination_event_listener_add](#dsl_player_termination_event_listener_add)
* [dsl_player_termination_event_listener_remove](#dsl_player_termination_event_listener_remove)
* [dsl_player_play](#dsl_player_play)
* [dsl_player_pause](#dsl_player_pause)
* [dsl_player_stop](#dsl_player_stop)
* [dsl_player_state_get](#dsl_player_state_get)
* [dsl_player_exists](#dsl_player_exists)
* [dsl_player_list_size](#dsl_player_list_size)

---
## Return Values
The following return codes are used by the Player API
```C++
#define DSL_RESULT_PLAYER_RESULT                                    0x00400000
#define DSL_RESULT_PLAYER_NAME_NOT_UNIQUE                           0x00400001
#define DSL_RESULT_PLAYER_NAME_NOT_FOUND                            0x00400002
#define DSL_RESULT_PLAYER_NAME_BAD_FORMAT                           0x00400003
#define DSL_RESULT_PLAYER_IS_NOT_RENDER_PLAYER                      0x00400004
#define DSL_RESULT_PLAYER_STATE_PAUSED                              0x00400005
#define DSL_RESULT_PLAYER_STATE_RUNNING                             0x00400006
#define DSL_RESULT_PLAYER_THREW_EXCEPTION                           0x00400007
#define DSL_RESULT_PLAYER_XWINDOW_GET_FAILED                        0x00400008
#define DSL_RESULT_PLAYER_XWINDOW_SET_FAILED                        0x00400009
#define DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED                       0x0040000A
#define DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED                    0x0040000B
#define DSL_RESULT_PLAYER_FAILED_TO_PLAY                            0x0040000C
#define DSL_RESULT_PLAYER_FAILED_TO_PAUSE                           0x0040000D
#define DSL_RESULT_PLAYER_FAILED_TO_STOP                            0x0040000E
#define DSL_RESULT_PLAYER_SET_FAILED                                0x00400011
```

---
## Client Callback Typedefs
### *dsl_player_termination_event_listener_cb*
```C++
typedef void (*dsl_player_termination_event_listener_cb)(void* client_data);
```
Callback typedef for a client termination event listener function. Functions of this type are added to a Player by calling [dsl_player_termination_event_listener_add](#dsl_player_termination_event_listener_add). Once added, the function will be called on the event of Pipeline end-of-stream (EOS) or Window Deletion. The listener function is removed by calling [dsl_player_termination_event_listener_remove](#dsl_pipeline_eos_listener_remove) . 

**Parameters**
* `client_data` - [in] opaque pointer to client's user data, passed into the Player on callback add

<br>

---
## Constructors
### *dsl_player_new*
```C++
DslReturnType dsl_player_new(const wchar_t* name,
    const wchar_t* source, const wchar_t* sink);;
```
The constructor creates a uniquely named Player. Construction will fail if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Player to create.
* `source` - [in] unique name of the Source component to use for this Player.
* `sink` - [in] unique name of the Sink component to use for this Player.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_new('my-player', 'my-rtsp-source', my-window-sink)
```

<br>

### *dsl_player_render_video_new*
```C++
DslReturnType dsl_player_render_video_new(const wchar_t* name,  const wchar_t* file_path, 
   uint render_type, uint offset_x, uint offset_y, uint zoom, boolean repeat_enabled);
```
The constructor creates a uniquely named Player that creates and manages its own File Source and Render Sink. Construction will fail
if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Player to create.
* `render_type` - [in] one of DSL_RENDER_TYPE_OVERLAY or DSL_RENDER_TYPE_WINDOW
* `offset_x` - [in] offset for the Render Sink in the X direction in units of pixels.
* `offset_y` - [in] offset for the Render Sink in the Y direction in units of pixels.
* `zoom` - [in] zoom factor for the Render Sink in units of %
* `repeat_enabled` - [in] set to true to repeat the video on end-of-stream.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_video_new('my-player', './streams/my-video.mp4',
  100, 100, 150, false)
```

<br>

### *dsl_player_render_image_new*
```C++
DslReturnType dsl_player_render_image_new(const wchar_t* name, const wchar_t* file_path,
    uint render_type, uint offset_x, uint offset_y, uint zoom, uint timeout)
```
The constructor creates a uniquely named Player that creates and manages its own Image Source and Render Sink. Construction will fail
if the name is currently in use.

**Parameters**
* `name` - [in] unique name for the Player to create.
* `render_type` - [in] one of DSL_RENDER_TYPE_OVERLAY or DSL_RENDER_TYPE_WINDOW
* `offset_x` - [in] offset for the Render Sink in the X direction in units of pixels.
* `offset_y` - [in] offset for the Render Sink in the Y direction in units of pixels.
* `zoom` - [in] zoom factor for the Render Sink in units of %
* `timeout` - [in] time to render the image before generating an end-of-stream event and terminating, in uints of seconds. Set to 0 for no timeout

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_image_new('my-player', './streams/my-image.jpg',
  100, 100, 150, 0)
```

<br>

---
## Destructors
### *dsl_player_delete*
```C++
DslReturnType dsl_player_delete(const wchar_t* pipeline);
```
This destructor deletes a single, uniquely named Player. Basic Players will set the Source and Sink components to a state of `not-in-use`. Render Players will delete the Source and Sink under their management. 

**Parameters**
* `name` - [in] unique name for the Player to delete

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_player_delete('my-player')
```

<br>

### *dsl_player_delete_all*
```C++
DslReturnType dsl_player_delete_all();
```
This destructor deletes all Players currently in memory  All Source and Sinks provided to Basic Players will move to a state of `not-in-use`. All Render Players will delete their own Source and Sink components.

**Returns**
* `DSL_RESULT_SUCCESS` on successful deletion. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_player_delete_all()
```

<br>

---
## Methods
### *dsl_player_render_file_path_get*
```C
DslReturnType dsl_player_render_file_path_get(const wchar_t* name, 
    const wchar_t** file_path);
```
This service gets the current file path in use by the Render Player

**Parameters**
* `name` - [in] unique name of the Render Player to query
* `file_path` - [out] current file path in use by the Render Player

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, file_path = dsl_player_render_file_path_get('my-video-player')
```
<br>

### *dsl_player_render_file_path_set*
```C
DslReturnType dsl_player_render_file_path_set(const wchar_t* name, 
    const wchar_t* file_path);
```
This service sets the File path for a named Render Player to use. 

**Parameters**
* `name` - [in] unique name of the Source to update
* `file_path` - [in] path to the Video or Image file to play

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_file_path_set('my-render-player', './streams/sample_1080p_h264.mp4')
```

<br>
