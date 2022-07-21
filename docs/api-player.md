# Player API Reference
Players are simplified/specialized Pipelines, created with a single [Source](/docs/api-source.md) and single [Sink](/docs/api-sink.md). Once created, the Source and Sink attributes can be updated, however, the components cannot be removed or replaced.

#### Player Construction and Destruction
There are three types of Players that can be created. 
* The **Basic Player**, created with any one Source and Sink by calling [dsl_player_new](#dsl_player_new). The Source and Sink components must exists at the time of Player construction
* Two Render Players that create and manage their own Sink and Source Components
  * An **Image Render Player**, created by calling  [dsl_player_render_image_new](#dsl_player_render_image_new), that creates an Image Source and Render Sink; Overlay or Window as specified.
  * A **Video Render Player**, created by calling  [dsl_player_render_video_new](#dsl_player_render_video_new), that creates a Video File Source and Render Sink as specified.

Players are destructed by calling [dsl_player_delete](#dsl_player_delete) or [dsl_player_delete_all](#dsl_player_delete_all).

#### Adding an Image Render Player to an ODE Capture Action
Images Players can be added to either a Frame or Object Capture Action for auto-play on ODE occurrence by calling [dsl_ode_action_capture_image_player_add](/docs/api-ode-action.md#dsl_ode_action_capture_image_player_add) and removed by calling [dsl_ode_action_capture_image_player_remove](/docs/api-ode-action.md#dsl_ode_action_capture_image_player_remove)

#### Adding a Video Render Player to a Smart Recording Tap or Sink
Video Players can be added to Smart Recording Taps and Sinks for auto-play on recording complete by calling [dsl_tap_record_video_player_add](/docs/api-tap.md#dsl_tap_record_video_player_add) and [dsl_sink_record_video_player_add](/docs/api-sink.md#dsl_sink_record_video_player_add) respectively and removed by calling [dsl_tap_record_video_player_remove](/docs/api-tap.md#dsl_tap_record_video_player_remove) and [dsl_sink_record_video_player_remove](/docs/api-sink.md#dsl_sink_record_video_player_remove)

#### Playing, Pausing and Stopping Players
Players can be played by calling [dsl_player_play](#dsl_player_play), paused (non-live sources only) by calling [dsl_player_pause](#dsl_player_pause), and stopped by calling [dsl_player_stop](#dsl_player_stop).

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
* [dsl_player_render_reset](#dsl_player_render_reset)
* [dsl_player_render_image_timeout_get](#dsl_player_render_image_timeout_get)
* [dsl_player_render_image_timeout_set](#dsl_player_render_image_timeout_set)
* [dsl_player_render_video_repeat_enabled_get](#dsl_player_render_video_repeat_enabled_get)
* [dsl_player_render_video_repeat_enabled_set](#dsl_player_render_video_repeat_enabled_set)
* [dsl_player_termination_event_listener_add](#dsl_player_termination_event_listener_add)
* [dsl_player_termination_event_listener_remove](#dsl_player_termination_event_listener_remove)
* [dsl_player_xwindow_handle_get](#dsl_player_xwindow_handle_get)
* [dsl_player_xwindow_handle_set](#dsl_player_xwindow_handle_set)
* [dsl_player_xwindow_key_event_handler_add](#dsl_player_xwindow_key_event_handler_add)
* [dsl_player_xwindow_key_event_handler_remove](#dsl_player_xwindow_key_event_handler_remove)
* [dsl_player_play](#dsl_player_play)
* [dsl_player_pause](#dsl_player_pause)
* [dsl_player_stop](#dsl_player_stop)
* [dsl_player_render_next](#dsl_player_render_next)
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
#define DSL_RESULT_PLAYER_RENDER_FAILED_TO_PLAY_NEXT                0x00400010
#define DSL_RESULT_PLAYER_SET_FAILED                                0x00400011
```

## DSL State Values
```C
#define DSL_STATE_NULL                                              1
#define DSL_STATE_READY                                             2
#define DSL_STATE_PAUSED                                            3
#define DSL_STATE_PLAYING                                           4
```

## Constants
The following symbolic constants are used by the ODE Trigger API.

**Note: the Overlay Sink is only available on the Jetson Platform.**

```C++
#define DSL_RENDER_TYPE_OVERLAY                                     0
#define DSL_RENDER_TYPE_WINDOW                                      1
```

---

## Client Callback Typedefs
### *dsl_player_termination_event_listener_cb*
```C++
typedef void (*dsl_player_termination_event_listener_cb)(void* client_data);
```
Callback typedef for a client termination event listener function. Functions of this type are added to a Player by calling [dsl_player_termination_event_listener_add](#dsl_player_termination_event_listener_add). Once added, the function will be called on the event of Player end-of-stream (EOS) or Window Deletion. The listener function is removed by calling [dsl_player_termination_event_listener_remove](#dsl_pipeline_eos_listener_remove) . 

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
The constructor creates a uniquely named Basic Player. Construction will fail if the name is currently in use.

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

### *dsl_player_render_image_new*
```C++
DslReturnType dsl_player_render_image_new(const wchar_t* name, const wchar_t* file_path,
    uint render_type, uint offset_x, uint offset_y, uint zoom, uint timeout)
```
The constructor creates a uniquely named Player that creates and manages its own Image Source and Render Sink. Construction will fail
if the name is currently in use. **Note: the Overlay Sink is only available on the Jetson Platform.**


**Parameters**
* `name` - [in] unique name for the Player to create.
* `render_type` - [in] one of DSL_RENDER_TYPE_OVERLAY or DSL_RENDER_TYPE_WINDOW
* `offset_x` - [in] offset for the Render Sink in the X direction in units of pixels.
* `offset_y` - [in] offset for the Render Sink in the Y direction in units of pixels.
* `zoom` - [in] zoom factor for the Render Sink in units of %
* `timeout` - [in] time to render the image before generating an end-of-stream event and terminating, in units of seconds. Set to 0 for no timeout

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_image_new('my-player', './streams/my-image.jpg',
  100, 100, 150, 0)
```

<br>

### *dsl_player_render_video_new*
```C++
DslReturnType dsl_player_render_video_new(const wchar_t* name,  const wchar_t* file_path, 
   uint render_type, uint offset_x, uint offset_y, uint zoom, boolean repeat_enabled);
```
The constructor creates a uniquely named Player that creates and manages its own Video File Source and Render Sink. Construction will fail
if the name is currently in use. **Note: the Overlay Sink is only available on the Jetson Platform.**

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
This destructor deletes all Players currently in memory  All Source and Sinks provided to Basic Players will move to a state of `not-in-use`. All Render Players will delete their own Source and Sink components.

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
This service gets the current file path in use by the Render Player. 

**Parameters**
* `name` - [in] unique name of the Render Player to query
* `file_path` - [out] current file path in use by the Render Player

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, file_path = dsl_player_render_file_path_get('my-render-player')
```
<br>

### *dsl_player_render_file_path_set*
```C
DslReturnType dsl_player_render_file_path_set(const wchar_t* name, 
    const wchar_t* file_path);
```
This service sets the File path for a named Render Player to use. The file path cannot be updated while the Render Player is playing.

**Parameters**
* `name` - [in] unique name of the Render Player to update
* `file_path` - [in] path to the Video or Image file to play

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_file_path_set('my-render-player', './streams/sample_1080p_h264.mp4')
```

<br>

### *dsl_player_render_file_path_queue*
```C
DslReturnType dsl_player_render_file_path_queue(const wchar_t* name, 
    const wchar_t* file_path);
```
This service queues a new File path for a named Image or Video Render Player to play on end-of-stream (EOS). 

**Parameters**
* `name` - [in] unique name of the Source to update
* `file_path` - [in] path to the Video or Image file to play

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_file_path_queue('my-render-player', './streams/sample_1080p_h264.mp4')
```

<br>

### *dsl_player_render_offsets_get*
```c++
DslReturnType dsl_player_render_offsets_get(const wchar_t* name, 
    uint* offset_x, uint* offset_y);
```
This service gets the current X and Y offsets for the named Render Player. 

**Parameters**
* `name` - [in] unique name of the Render Player to query.
* `x_offset` - [out] Current offset in the X direction in units of pixels
* `y_offset` - [out] Current offset in the Y direction in units of pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, offset_x, offset_y = dsl_player_render_offsets_get('my-render-player')
```

<br>

### *dsl_player_render_offsets_set*
```c++
DslReturnType dsl_player_render_offsets_set(const wchar_t* name, 
    uint offset_x, uint offset_y);
```
This service sets the X and Y offsets for the named Render Player. The X and Y offsets can be updated while the Player is player.

**Parameters**
* `name` - [in] unique name of the Render Player to update.
* `x_offset` - [in] Offset in the X direction in units of pixels
* `y_offset` - [in] Offset in the Y direction in units of pixels

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, dsl_player_render_offsets_set('my-on-screen-display', 200, 100)
```

<br>

### *dsl_player_render_zoom_get*
```C
DslReturnType dsl_player_render_zoom_get(const wchar_t* name, uint* zoom);
```
This service gets the current zoom setting in use by the named Render Player

**Parameters**
* `name` - [in] unique name of the Render Player to query
* `zoom` - [out] current zoom setting in use in units of %

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, zoom = dsl_player_render_zoom_get('my-render-player')
```
<br>

### *dsl_player_render_zoom_set*
```C
DslReturnType dsl_player_render_zoom_set(const wchar_t* name, uint zoom);
```
This service sets the zoom setting to use for the named Render Player. The zoom setting can be updated while the player.

**Parameters**
* `name` - [in] unique name of the Render Player to update
* `zoom` - [in] new zoom setting to use in units of %

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_zoom_set('my-render-player', 50)
```

<br>

### *dsl_player_render_reset*
```C++
DslReturnType dsl_player_render_reset(const wchar_t* name);
```
This service resets the named Render Player causing it to close it's rendering surface. This service will fail if the Render Player is currently `playing` or `paused`.

**Parameters**
* `name` - [in] unique name of the Render Player to update; Overlay or Window type.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_stop('my-player')
retval = dsl_player_render_reset('my-player)

# The rendering surface (Overlay or Window) will be recreated on next Play
retval = dsl_player_play('my-player')
```

<br>

### *dsl_player_render_image_timeout_get*
```C
DslReturnType dsl_player_render_image_timeout_get(const wchar_t* name, uint* timeout);
```
This service gets the current timeout setting in use by the named Image Render Player. If set, the Image Source will generate an end-of-stream (EOS) event on timeout. The Player will play the next Image if a new file path has been queued. 

**Parameters**
* `name` - [in] unique name of the Image Render Player to query
* `timeout` - [out] current timeout setting in use in units of seconds

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, timeout = dsl_player_render_image_timeout_get('my-image-render-player')
```
<br>

### *dsl_player_render_image_timeout_set*
```C
DslReturnType dsl_player_render_image_timeout_set(const wchar_t* name, uint timeout);
```
This service sets the timeout setting to use for the named Image Render Player. If set, the Image Source will generate an end-of-stream (EOS) event on timeout. The Player will play the next Image if a new file path has been queued. The timeout setting cannot be changed while the Player is playing.

**Parameters**
* `name` - [in] unique name of the Image Render Player to update
* `timeout` - [in] new timeout setting to use in units of seconds

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_image_timeout_set('my-image-render-player', 30)
```

<br>

### *dsl_player_render_video_repeat_enabled_get*
```C
DslReturnType dsl_player_render_video_repeat_enabled_get(const wchar_t* name, 
    boolean* repeat_enabled);
```
This service gets the current repeat-enabled setting in use by the named Video Render Player. If enabled, the Video File Source will repeat the video on end-of-stream (EOS), i.e. at the end of the video. 

**Parameters**
* `name` - [in] unique name of the Video Render Player to query
* `repeat_enabled` - [out] current repeat_enabled setting in use..

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, timeout = dsl_player_render_video_repeat_enabled_get('my-video-render-player')
```
<br>

### *dsl_player_render_video_repeat_enabled_set*
```C
DslReturnType dsl_player_render_video_repeat_enabled_set(const wchar_t* name, 
    boolean repeat_enabled);
```
This service sets the repeat-enable setting to use for the named Image Render Player. If enabled, the Video File Source will repeat the video on end-of-stream (EOS), i.e. at the end of the video. The repeat-enabled setting cannot be changed while the Player is playing.
**Parameters**
* `name` - [in] unique name of the Source to update
* `repeat_enabled` - [in] set to true to enable repeat on EOS, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_video_repeat_enabled_set('my-video-render-player', 30)
```

<br>

### *dsl_player_termination_event_listener_add*
```C++
DslReturnType dsl_player_termination_event_listener_add(const wchar_t* name, 
    dsl_player_termination_event_listener_cb listener, void* client_data);
```
This service adds a callback function of type [dsl_player_termination_event_listener_cb](#dsl_player_termination_event_listener_cb) to a Player identified by its unique name. The function will be called on a Player Termination event, either end-of-stream (eos) or Window Delete. Multiple callback functions can be registered with one Player, and one callback function can be registered with multiple Players.

**Parameters**
* `name` - [in] unique name of the Player to update.
* `listener` - [in] Termination listener callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the listener when called back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_termination_event_listener_add('my-player', listener_cb, None)
```

<br>

### *dsl_player_termination_event_listener_remove*
```C++
DslReturnType dsl_player_termination_event_listener_remove(const wchar_t* name, 
    dsl_player_termination_event_listener_cb listener);
```
This service removes a callback function of type [dsl_player_termination_event_listener_cb](#dsl_player_termination_event_listener_cb) from a Player identified by it's unique name.

**Parameters**
* `name` - [in] unique name of the Player to update.
* `listener` - [in] termination listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_termination_event_listener_remove('my-player', listener_cb)
```

<br>

### *dsl_player_xwindow_handle_get*
```C++
DslReturnType dsl_player_xwindow_handle_get(const wchar_t* name, uint64_t* xwindow);
```
This service returns the current XWindow handle in use by the named Player. The handle is set to `Null` on Player creation and will remain `Null` until,
1. The Player creates an internal XWindow synchronized with a Window-Sink on Transition to a state of playing, or
2. The Client Application passes an XWindow handle into the Player by calling [dsl_player_xwindow_handle_set](#dsl_player_xwindow_handle_set).

**Parameters**
* `name` - [in] unique name for the Player to query.
* `handle` - [out] XWindow handle in use by the named Player.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, x_window = dsl_player_xwindow_handle_get('my-player')
```
<br>

### *dsl_player_xwindow_handle_set*
```C++
DslReturnType dsl_player_xwindow_handle_set(const wchar_t* name, uint64_t window);
```
This service sets the the XWindow for the named Player to use. Must be called prior to playing the Player

**Parameters**
* `name` - [in] unique name for the Player to update.
* `handle` - [in] XWindow handle to use by the Window-Sink in use by this Pipeline.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_player_xwindow_handle_set('my-player', x_window)
```
<br>

### *dsl_player_xwindow_key_event_handler_add*
```C++
DslReturnType dsl_player_xwindow_key_event_handler_add(const wchar_t* name, 
    dsl_xwindow_key_event_handler_cb handler, void* client_data);
```
This service adds a callback function of type [dsl_xwindow_key_event_handler_cb](#dsl_xwindow_key_event_handler_cb) to a Player identified by it's unique name. The function will be called on every Player XWindow `KeyReleased` event with Key string and the client provided `client_data`. Multiple callback functions can be registered with one Player, and one callback function can be registered with multiple Players.

**Note** Client XWindow Callback functions will only be called if the Player owns an XWindow, which requires a Window Sink component.

**Parameters**
* `name` - [in] unique name of the Player to update.
* `handler` - [in] XWindow event handler callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns** 
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def key_event_handler(key_string, client_data):
    print('key pressed = ', key_string)
    
retval = dsl_player_xwindow_key_event_handler_add('my-pipeline', key_event_handler, None)
```

<br>

### *dsl_player_xwindow_key_event_handler_remove*
```C++
DslReturnType dsl_player_xwindow_key_event_handler_remove(const wchar_t* name, 
    dsl_xwindow_key_event_handler_cb handler);
```
This service removes a Client XWindow key event handler callback that was added previously with [dsl_player_xwindow_key_event_handler_add](#dsl_player_xwindow_key_event_handler_add)

**Parameters**
* `name` - [in] unique name of the Player to update
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_pipeline_xwindow_key_event_handler_remove('my-pipeline', key_event_handler)
```

<br>

### *dsl_player_play*
```C++
DslReturnType dsl_player_play(wchar_t* name);
```
This service is used to play a named Player.  

**Parameters**
* `name` - [in] unique name for the Player to play.

**Returns** 
* `DSL_RESULT_SUCCESS` if the named Player is able to successfully transition to a state of `playing`, one of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_play('my-player')
```

<br>

### *dsl_player_pause*
```C++
DslReturnType dsl_player_pause(wchar_t* name);
```
This service is used to pause a named Player. The service is supported by Players with a `non-live` Source only, including the Image and Video Render Players.

**Parameters**
* `name` - [in] unique name for the Player to pause.

**Returns** 
* `DSL_RESULT_SUCCESS` if the named Player is able to successfully transition to a state of `paused`, one of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_pause('my-player')
```

<br>

### *dsl_player_stop*
```C++
DslReturnType  dsl_player_stop(wchar_t* name);
```
This service is used to stop a named Player and return it to a state of `ready`. The service will fail if the Player is not currently in a `playing` or `paused` state.

**Parameters**
* `name` - [in] unique name for the Player to stop.

**Returns**
* `DSL_RESULT_SUCCESS` if the named Player is able to successfully transition to a state of `stopped`, one of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_stop('my-player')
```

<br>

### *dsl_player_render_next*
```C++
DslReturnType dsl_player_render_next(const wchar_t* name);
```
This service is used to stop a named Player and Play the next queued `file_path` if one exists. The service is supported by Render Players only.
**Parameters**
* `name` - [in] unique name for the Player to stop and play next.

**Returns**
* `DSL_RESULT_SUCCESS` if the named Render Player is able to plan the next queued file, one of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_player_render_next('my-player')
```

<br>

### *dsl_player_state_get*
```C++
DslReturnType dsl_player_state_get(wchar_t* name, uint* state);
```
This service returns the current [state](#dsl-state-values) of the named Player The service fails if the named Player was not found.  

**Parameters**
* `name` - [in] unique name for the Player to query.
* `state` - [out] the current [state](#dsl-state-values) of the named Player

**Returns**
* `DSL_RESULT_SUCCESS` on successful get. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, state = dsl_player_state_get('my-player')
```

<br>

### *dsl_player_exists*
```C++
boolean dsl_player_exists(const wchar_t* name);
```
This service is used to determine if a named Player currently exists in memory.

**Parameters**
* `name` - [in] unique name for the Player to query for.
* 
**Returns** 
* true if the named Player exists, false otherwise.

**Python Example**
```Python
if dsl_player_exists():
   # handle true case
```

<br>

### *dsl_player_list_size*
```C++
uint dsl_player_list_size();
```
This service returns the size of the current list of Players in memory

**Returns** 
* The number of Players currently in memory

**Python Example**
```Python
player_count = dsl_player_list_size()
```

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* **Player**
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](/docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pad-probe-handler.md)
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
