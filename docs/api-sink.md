
# Sink API Reference
Sinks are the end components for all DSL GStreamer Pipelines. A Pipeline must have at least one sink in use, along with other certain components, to reach a state of Ready. 

All Sinks are derived from the "Component" class, therefore all [component methods](/docs/api-component.md) can be called with any Sink.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── `sink`

DSL supports ten (12) different types of Sinks:
* [Overlay Sink](#dsl_sink_overlay_new) - renders/overlays video on a Parent display **(Jetson Platform Only)**
* [Window Sink](#dsl_sink_window_new) - renders/overlays video on a Parent XWindow
* [File Sink](#dsl_sink_file_new) - encodes video to a media container file
* [Record Sink](#dsl_sink_record_new) - similar to the File sink but with Start/Stop/Duration control and a cache for pre-start buffering.
* [RTSP Sink](#dsl_sink_rtsp_new) - streams encoded video on a specified port
* [WebRTC Sink](#dsl_sink_webrtc_new) - streams encoded video to a web browser or mobile application. **(Requires GStreamer 1.18 or later)**
* [Message Sink](dsl_sink_message_new) - converts Object Detection Event (ODE) metadata into a message payload and sends it to the server using a specified communication protocol.
* [Application Sink](#dsl_sink_app_new) - allows the application to receive buffers or samples from a DSL Pipeline.
* [Interpipe Sink](#dsl_sink_interpipe_new) -  allows pipeline buffers and events to flow to other independent pipelines, each with an [Interpipe Source](/docs/api-source.md#dsl_source_interpipe_new). Disabled by default, requires additional [install/build steps](/docs/installing-dependencies.md).
* [Multi-Image Sink](#dsl_sink_image_multi_new) - encodes and saves video frames to JPEG files at specified dimensions and frame-rate.
* [Frame-Capture Sink](#dsl_sink_frame_capture_new) - encodes and saves video frames to JPEG files on application/user demand. Disabled by default, requires additional [install/build steps](/docs/installing-dependencies.md).
* [Fake Sink](#dsl_sink_fake_new) - consumes/drops all data.

### Sink Construction and Destruction
Sinks are created by calling one of the type-specific constructors. As with all components, Sinks must be uniquely named from all other components created. 

The relationship between [Pipelines](/docs/api-pipeline.md) and Sinks is one-to-many. The same relationship exists between [Branches](/docs/api-branch.md) and Sinks. Once added to a Pipeline, a Sink must be removed before it can be used with another. Sinks are deleted by calling [`dsl_component_delete`](/docs/api-component.md#dsl_component_delete), [`dsl_component_delete_many`](/docs/api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](/docs/api-component.md#dsl_component_delete_all)

### Adding and Removing
When adding a Sink(s) to a Pipeline or Branch, DSL automatically inserts a Splitter Tee between the last component and the Sink(s) as show in the image below, even if there is only one.  This ensures that additional Sinks can be added (and removed) once the Pipeline is playing. 

<img src="/Images/multi-sink-splitter-tee.png"/>

Sinks are added to a Pipeline by calling [`dsl_pipeline_component_add`](/docs/api-pipeline.md#dsl_pipeline_component_add) or [`dsl_pipeline_component_add_many`](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [`dsl_pipeline_component_remove`](/docs/api-pipeline.md#dsl_pipeline_component_remove), [`dsl_pipeline_component_remove_many`](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [`dsl_pipeline_component_remove_all`](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

A similar set of Services are used when adding/removing a to/from a branch: [`dsl_branch_component_add`](api-branch.md#dsl_branch_component_add), [`dsl_branch_component_add_many`](/docs/api-branch.md#dsl_branch_component_add_many), [`dsl_branch_component_remove`](/docs/api-branch.md#dsl_branch_component_remove), [`dsl_branch_component_remove_many`](/docs/api-branch.md#dsl_branch_component_remove_many), and [`dsl_branch_component_remove_all`](/docs/api-branch.md#dsl_branch_component_remove_all).

**IMPORTANT!:** A Sink Component can be added as a branch to either a [Splitter or Demuxer Tee](/docs/api-tee.md).

### Common Sink Properties
All Sinks<sup id="a1">[1](#f1)</sup> support the following common properties accessible through corresponding get/set base [Sink Methods](#sink-methods). (_Note: the follow bullets are quotes from the [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/base/gstbasesink.html?gi-language=c)_)
* **`sync`** : Each Sink sets a timestamp for when a frame should be played, if `sync=true` it will block the pipeline and only play the frame after that time. This is useful for playing from a video file, or other non-live sources. If you play a video file with `sync=false` it will play back as fast as it can be read and processed. See [`dsl_sink_sync_enabled_get`](#dsl_sink_sync_enabled_get) and [`dsl_sink_sync_enabled_set`](#dsl_sink_sync_enabled_set).
As a general rule
   * Set `sync=true` if using non-live sources and/or if the stream is to be rendered and viewed.
   * Set `sync=false` if using live sources and/or if significantly processing the stream.
* **`async`** : If `async=true`, the Sink will perform asynchronous state changes. When `async=false`, the Sink will not signal the parent when it prerolls. Use this option when dealing with sparse streams or when synchronization is not required. See [`dsl_sink_async_enabled_get`](#dsl_sink_async_enabled_get) and [`dsl_sink_async_enabled_set`](#dsl_sink_async_enabled_set).
* **`max-lateness`** : The max-lateness property affects how the Sink deals with buffers that arrive too late. A buffer arrives too late in the Sink when the presentation time (as a combination of the last segment, buffer timestamp and element base_time) plus the duration is before the current time of the clock. If the frame is later than max-lateness (in nanoseconds), the sink will drop the buffer without calling the render method. This feature is disabled if `sync=false`. See [`dsl_sink_max_lateness_get`](#dsl_sink_max_lateness_get) and [`dsl_sink_max_lateness_se`t](#dsl_sink_max_lateness_set).
* **`qos`** :If `qos=true`, the property will enable the quality-of-service features of the Sink which gather statistics about the real-time performance of the clock synchronization. For each buffer received in the Sink, statistics are gathered and a QOS event is sent upstream with these numbers. This information can then be used by upstream elements to reduce their processing rate, for example. See [`dsl_sink_qos_enabled_get`](#dsl_sink_qos_enabled_get) and [`dsl_sink_qos_enabled_set`](#dsl_sink_qos_enabled_set).

**IMPORTANT!** All DSL Sink Components use the default property values assigned to their GStreamer (GST) Sink Plugin as defined in the table below.

#### Default common property values
| Sink               |  GST Plugin   | sync  | async | max-lateness |  qos  |
| -------------------|---------------|-------|------ | ------------ | ----- |
| Overlay Sink       | nvoverlaysink | true  | true  |   20000000   | true  |
| Window Sink        | nveglglessink | true  | true  |   20000000   | true  |
| File Sink          | filesink      | false | true  |      -1      | false |
| Record Sink        | n/a           |  n/a  |  n/a  |      n/a     |  n/a  |
| RTSP Sink          | udpsink       | true  | true  |      -1      | false |
| WebRTC Sink        | fakesink      | false | true  |      -1      | false |
| Message Sink       | nvmsgbroker   | true  | true  |      -1      | false |
| App Sink           | appsink       | true  | true  |      -1      | false |
| Interpipe Sink     | interpipesink | false | true  |      -1      | false |
| Multi-Image Sink   | multifilesink | false | true  |      -1      | false |
| Frame-Capture Sink | appsink       | true  | true  |      -1      | false |
| Fake Sink          | fakesink      | false | true  |      -1      | false |

<b id="f1">1</b> _The NVIDIA Smart Recording Bin - used by the Record Sink - does not support/extern any of the common sink properties._ [↩](#a1)

## Sink API
**Types:**
* [`dsl_recording_info`](#dsl_recording_info)

**Callback Types:**
* [`dsl_sink_app_new_data_handler_cb`](#dsl_sink_app_new_data_handler_cb)
* [`dsl_sink_window_key_event_handler_cb`](#dsl_sink_window_key_event_handler_cb)
* [`dsl_sink_window_button_event_handler_cb`](#dsl_sink_window_button_event_handler_cb)
* [`dsl_sink_window_delete_event_handler_cb`](#dsl_sink_window_delete_event_handler_cb)
* [`dsl_record_client_listener_cb`](#dsl_record_client_listener_cb)
* [`dsl_sink_webrtc_client_listener_cb`](#dsl_sink_webrtc_client_listener_cb)

**Constructors:**
* [`dsl_sink_app_new`](#dsl_sink_app_new)
* [`dsl_sink_overlay_new`](#dsl_sink_overlay_new)
* [`dsl_sink_window_new`](#dsl_sink_window_new)
* [`dsl_sink_file_new`](#dsl_sink_file_new)
* [`dsl_sink_record_new`](#dsl_sink_record_new)
* [`dsl_sink_rtsp_new`](#dsl_sink_rtsp_new)
* [`dsl_sink_webrtc_new`](#dsl_sink_webrtc_new)
* [`dsl_sink_message_new`](#dsl_sink_message_new)
* [`dsl_sink_interpipe_new`](#dsl_sink_interpipe_new)
* [`dsl_sink_image_multi_new`](#dsl_sink_image_multi_new)
* [`dsl_sink_frame_capture_new`](#dsl_sink_frame_capture_new)
* [`dsl_sink_fake_new`](#dsl_sink_fake_new)

**Sink Methods**
* [`dsl_sink_sync_enabled_get`](#dsl_sink_sync_enabled_get)
* [`dsl_sink_sync_enabled_set`](#dsl_sink_sync_enabled_set)
* [`dsl_sink_async_enabled_get`](#dsl_sink_async_enabled_get)
* [`dsl_sink_async_enabled_set`](#dsl_sink_async_enabled_set)
* [`dsl_sink_max_lateness_get`](#dsl_sink_max_lateness_get)
* [`dsl_sink_max_lateness_set`](#dsl_sink_max_lateness_set)
* [`dsl_sink_qos_enabled_get`](#dsl_sink_qos_enabled_get)
* [`dsl_sink_qos_enabled_set`](#dsl_sink_qos_enabled_set)
* [`dsl_sink_pph_add`](#dsl_sink_pph_add)
* [`dsl_sink_pph_remove`](#dsl_sink_pph_remove)

**App Sink Methods**
* [`dsl_sink_app_data_type_get`](#dsl_sink_app_data_type_get)
* [`dsl_sink_app_data_type_set`](#dsl_sink_app_data_type_set)

**Render Sink Methods**
* [`dsl_sink_render_offsets_get`](#dsl_sink_render_offsets_get)
* [`dsl_sink_render_offsets_set`](#dsl_sink_render_offsets_set)
* [`dsl_sink_render_dimensions_get`](#dsl_sink_render_dimensions_get)
* [`dsl_sink_render_dimensions_set`](#dsl_sink_render_dimensions_set)

**Window Sink Methods**
* [`dsl_sink_window_handle_get`](#dsl_sink_window_handle_get)
* [`dsl_sink_window_handle_set`](#dsl_sink_window_handle_set)
* [`dsl_sink_window_force_aspect_ratio_get`](#dsl_sink_window_force_aspect_ratio_get)
* [`dsl_sink_window_force_aspect_ratio_set`](#dsl_sink_window_force_aspect_ratio_set)
* [`dsl_sink_window_fullscreen_enabled_get`](#dsl_sink_window_fullscreen_enabled_get)
* [`dsl_sink_window_fullscreen_enabled_set`](#dsl_sink_window_fullscreen_enabled_set)
* [`dsl_sink_window_key_event_handler_add`](#dsl_sink_window_key_event_handler_add)
* [`dsl_sink_window_key_event_handler_remove`](#dsl_sink_window_key_event_handler_remove)
* [`dsl_sink_window_button_event_handler_add`](#dsl_sink_window_button_event_handler_add)
* [`dsl_sink_window_button_event_handler_remove`](#dsl_sink_window_button_event_handler_remove)
* [`dsl_sink_window_delete_event_handler_add`](#dsl_sink_window_delete_event_handler_add)
* [`dsl_sink_window_delete_event_handler_remove`](#dsl_sink_window_delete_event_handler_remove)

**Encode Sink Methods**
* [`dsl_sink_encode_settings_get`](#dsl_sink_encode_settings_get)
* [`dsl_sink_encode_settings_set`](#dsl_sink_encode_settings_set)
* [`dsl_sink_encode_dimensions_get`](#dsl_sink_encode_dimensions_get)
* [`dsl_sink_encode_dimensions_set`](#dsl_sink_encode_dimensions_set)

**Smart-Record Sink Methods**
* [`dsl_sink_record_session_start`](#dsl_sink_record_session_start)
* [`dsl_sink_record_session_stop`](#dsl_sink_record_session_stop)
* [`dsl_sink_record_outdir_get`](#dsl_sink_record_outdir_get)
* [`dsl_sink_record_outdir_set`](#dsl_sink_record_outdir_set)
* [`dsl_sink_record_container_get`](#dsl_sink_record_container_get)
* [`dsl_sink_record_container_set`](#dsl_sink_record_container_set)
* [`dsl_sink_record_cache_size_get`](#dsl_sink_record_cache_size_get)
* [`dsl_sink_record_cache_size_set`](#dsl_sink_record_cache_size_set)
* [`dsl_sink_record_dimensions_get`](#dsl_sink_record_dimensions_get)
* [`dsl_sink_record_dimensions_set`](#dsl_sink_record_dimensions_set)
* [`dsl_sink_record_is_on_get`](#dsl_sink_record_is_on_get)
* [`dsl_sink_record_video_player_add`](#dsl_sink_record_video_player_add)
* [`dsl_sink_record_video_player_remove`](#dsl_sink_record_video_player_remove)
* [`dsl_sink_record_mailer_add`](#dsl_sink_record_mailer_add)
* [`dsl_sink_record_mailer_remove`](#dsl_sink_record_mailer_remove)
* [`dsl_sink_record_reset_done_get`](#dsl_sink_record_reset_done_get)

**RTSP Sink Methods**
* [`dsl_sink_rtsp_server_settings_get`](#dsl_sink_rtsp_server_settings_get)

**WebRTC Sink Methods**
* [`dsl_sink_webrtc_connection_close`](#dsl_sink_webrtc_connection_close)
* [`dsl_sink_webrtc_servers_get`](#dsl_sink_webrtc_servers_get)
* [`dsl_sink_webrtc_servers_set`](#dsl_sink_webrtc_servers_set)
* [`dsl_sink_webrtc_client_listener_add`](#dsl_sink_webrtc_client_listener_add)
* [`dsl_sink_webrtc_client_listener_remove`](#dsl_sink_webrtc_client_listener_remove)

**IOT Message Converter Sink Methods**
* [`dsl_sink_message_converter_settings_get`](#dsl_sink_message_converter_settings_get)
* [`dsl_sink_message_converter_settings_set`](#dsl_sink_message_converter_settings_set)
* [`dsl_sink_message_broker_settings_get`](#dsl_sink_message_broker_settings_get)
* [`dsl_sink_message_broker_settings_set`](#dsl_sink_message_broker_settings_set)

**Interpipe Sink Methods**
* [`dsl_sink_interpipe_forward_settings_get`](#dsl_sink_interpipe_forward_settings_get)
* [`dsl_sink_interpipe_forward_settings_set`](#dsl_sink_interpipe_forward_settings_set)
* [`dsl_sink_interpipe_num_listeners_get`](#dsl_sink_interpipe_num_listeners_get)

**Multi-Image Sink Methods**
* [`dsl_sink_image_multi_file_path_get`](#dsl_sink_image_multi_file_path_get)
* [`dsl_sink_image_multi_file_path_set`](#dsl_sink_image_multi_file_path_set)
* [`dsl_sink_image_multi_dimensions_get`](#dsl_sink_image_multi_dimensions_get)
* [`dsl_sink_image_multi_dimensions_set`](#dsl_sink_image_multi_dimensions_set)
* [`dsl_sink_image_multi_frame_rate_get`](#dsl_sink_image_multi_frame_rate_get)
* [`dsl_sink_image_multi_frame_rate_set`](#dsl_sink_image_multi_frame_rate_set)
* [`dsl_sink_image_multi_file_max_get`](#dsl_sink_image_multi_file_max_get)

**Frame-Capture Sink Methods**
* [`dsl_sink_frame_capture_initiate`](#dsl_sink_frame_capture_initiate)

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
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_RENDER_SINK                0x0004000D
#define DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED             0x0004000E
#define DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED          0x0004000F
#define DSL_RESULT_SINK_HANDLER_ADD_FAILED                          0x00040010
#define DSL_RESULT_SINK_HANDLER_REMOVE_FAILED                       0x00040011
#define DSL_RESULT_SINK_PLAYER_ADD_FAILED                           0x00040012
#define DSL_RESULT_SINK_PLAYER_REMOVE_FAILED                        0x00040013
#define DSL_RESULT_SINK_MAILER_ADD_FAILED                           0x00040014
#define DSL_RESULT_SINK_MAILER_REMOVE_FAILED                        0x00040015
#define DSL_RESULT_SINK_OVERLAY_NOT_SUPPORTED                       0x00040016
#define DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_ADD_FAILED           0x00040017
#define DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_REMOVE_FAILED        0x00040018
#define DSL_RESULT_SINK_WEBRTC_CONNECTION_CLOSED_FAILED             0x00040019
#define DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND               0x00040020
#define DSL_RESULT_SINK_COMPONENT_IS_NOT_MESSAGE_SINK               0x00040021
```

## Codec Types
The following codec types are used by the Sink API
```C
#define DSL_CODEC_H264                                              0
#define DSL_CODEC_H265                                              1
```

## Video Container Types
The following video container types are used by the File Sink API
```C
#define DSL_CONTAINER_MPEG4                                         0
#define DSL_CONTAINER_MK4                                           1
```

## Smart Recording Events
```C
#define DSL_RECORDING_EVENT_START                                   0
#define DSL_RECORDING_EVENT_END                                     1
```

## Valid return values for the dsl_sink_app_new_data_handler_cb
```C
#define DSL_FLOW_OK                                                 0
#define DSL_FLOW_EOS                                                1
#define DSL_FLOW_ERROR                                              2
```

## Data types provided by the APP Sink
```C
#define DSL_SINK_APP_DATA_TYPE_SAMPLE                               0
#define DSL_SINK_APP_DATA_TYPE_BUFFER                               1
```

## WebRTC Connection States
Used by the WebRTC Sink to communicate its current state to listening clients
```C
#define DSL_SOCKET_CONNECTION_STATE_CLOSED                          0
#define DSL_SOCKET_CONNECTION_STATE_INITIATED                       1
#define DSL_SOCKET_CONNECTION_STATE_OPENED                          2
#define DSL_SOCKET_CONNECTION_STATE_FAILED                          3
```

## Message Converter Payload Schema Types
Defines the Payload schema types that can be used with the Message Sink
```C
#define DSL_MSG_PAYLOAD_DEEPSTREAM                                  0
#define DSL_MSG_PAYLOAD_DEEPSTREAM_MINIMAL                          1
```

## NVIDIA Installed Protocol Adapter Paths
The following -D flags are defined in the DSL Makefile
```C
-DNVDS_AZURE_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_azure_proto.so"' \
-DNVDS_AZURE_EDGE_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_azure_edge_proto.so \
-DNVDS_AMQP_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_amqp_proto.so \
-DNVDS_KAFKA_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_kafka_proto.so \
-DNVDS_REDIS_PROTO_LIB='L"$(LIB_INSTALL_DIR)/libnvds_redis_edge_proto.so
 ```

<br>

---

## Types:

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
Structure typedef used to provide recording session information provided to the client on callback

**Fields**
* `recording_event` - specifies which recording event has occurred. One of DSL_RECORDING_EVENT_START or DSL_RECORDING_EVENT_END
* `sessionId` - the unique sessions id assigned on record start
* `filename` - filename generated for the completed recording. Null on recording start.
* `directory` - path for the completed recording. Null on recording start.
* `duration` - duration of the recording in milliseconds. 0 on recording start.
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
    print('event type: ', session_info.recording_event)
    print('session_id: ', session_info.session_id)
    print('filename:   ', session_info.filename)
    print('dirpath:    ', session_info.dirpath)
    print('duration:   ', session_info.duration)
    print('container:  ', session_info.container_type)
    print('width:      ', session_info.width)
    print('height:     ', session_info.height)
```

### *dsl_webrtc_connection_data*
```C
typedef struct _dsl_webrtc_connection_data
{
    uint current_state;

} dsl_webrtc_connection_data;
```

A structure typedef used to provide connection date for a given WebRTC Sink

**Fields**
* `current_state` - the current state of the WebRTC Sink's WebSocket connection, one of the defined [WebRTC Connection States](webrtc_connection_states).

<br>

## Callback Types:


### *dsl_sink_app_new_data_handler_cb*
```C++
typedef uint (*dsl_sink_app_new_data_handler_cb)(uint data_type, 
    void* data, void* client_data);
```
Callback typedef for the App Sink Component. The function is registered when the App Sink is created with [dsl_sink_app_new](#dsl_sink_app_new). Once the Pipeline is playing, the function will be called when new data is available to process. The type of data is specified with the App Sink constructor.

**Parameters**
* `data_type` [in] either `DSL_SINK_APP_DATA_TYPE_SAMPLE` or `DSL_SINK_APP_DATA_TYPE_BUFFER`. See [App Sink data-types](#data-types-provided-by-the-app-sink).
* `client_data` [in] opaque pointer to client's user data, provided by the client.

**Returns**
* One of the [DSL_FLOW](#valid-return-values-for-the-dsl_sink_app_new_data_handler_cb) constant values.

<br>

### *dsl_sink_window_key_event_handler_cb*
```C++
typedef void (*dsl_sink_window_key_event_handler_cb)(const wchar_t* key, void* client_data);
```
Callback typedef for a client XWindow `KeyRelease` event handler function. Functions of this type are added to a Window Sink by calling [dsl_sink_window_key_event_handler_add](#dsl_sink_window_key_event_handler_add). Once added, the function will be called on every XWindow `KeyRelease` event. The handler function is removed by calling  [dsl_sink_window_key_event_handler_remove](#dsl_sink_window_key_event_handler_remove).

**Parameters**
* `key` - [in] UNICODE key string for the key pressed
* `client_data` - [in] opaque pointer to client's user data, passed into the Window Sink on callback add

<br>

### *dsl_sink_window_button_event_handler_cb*
```C++
typedef void (*dsl_sink_window_button_event_handler_cb)(uint button, uint xpos, uint ypos, void* client_data);
```
Callback typedef for a client XWindow `ButtonPress` event handler function. Functions of this type are added to a Window Sink by calling [dsl_sink_window_button_event_handler_add](#dsl_sink_window_button_event_handler_add). Once added, the function will be called on every XWindow `ButtonPress` event. The handler function is removed by calling [dsl_sink_window_button_event_handler_remove](#dsl_sink_window_button_event_handler_remove).

**Parameters**
* `button` - [in] one of [DSL_BUTTON_ID](#) indicating which mouse button was pressed
* `xpos` - [in] positional X-offset from the XWindow's upper left corner in pixels
* `ypos` - [in] positional Y-offset from the XWindow's upper left corner in pixels
* `client_data` - [in] opaque pointer to client's user data, passed into the Window Sink on callback add

<br>

### *dsl_sink_window_delete_event_handler_cb*
```C++
typedef void (*dsl_sink_window_delete_event_handler_cb)(void* client_data);
```
Callback typedef for a client XWindow `Delete` event handler function. Functions of this type are added to a Pipeline by calling [dsl_sink_window_delete_event_handler_add](#dsl_sink_window_delete_event_handler_add). Once added, the function will be called on XWindow `Delete` event. The handler function is removed by calling [dsl_sink_window_button_event_handler_remove](#dsl_sink_window_button_event_handler_remove).

**Parameters**
* `client_data` - [in] opaque pointer to client's user data, passed into the Window Sink on callback add

<br>

### *dsl_record_client_listener_cb*
```C++
typedef void* (*dsl_record_client_listener_cb)(void* info, void* user_data);
```
Callback typedef for clients to listen for a notification that a Recording Session has started or ended.

**Parameters**
* `info` [in] opaque pointer to the connection info, see... see [dsl_capture_info](#dsl_capture_info).
* `client_data` [in] opaque pointer to client's user data, provided by the client.

<br>

### *dsl_sink_webrtc_client_listener_cb*
```C++
typedef void (*dsl_sink_webrtc_client_listener_cb)(dsl_webrtc_connection_data* data,
    void* client_data);
```
Callback typedef for a client to listen for WebRTC Sink connection events.

**IMPORTANT:** The WebRTC Sink implementation requires GStreamer 1.18 or later.

**Parameters**
* `info` [in] opaque pointer to the session info, see [dsl_webrtc_connection_data](#dsl_webrtc_connection_data).
* `client_data` [in] opaque pointer to client's user data, provided by the client.

---

## Constructors
### *dsl_sink_app_new*
```C++
DslReturnType dsl_sink_app_new(const wchar_t* name, uint data_type,
    dsl_sink_app_new_data_handler_cb client_handler, void* client_data);
```
The constructor creates a new, uniquely named App Sink component. Construction will fail if the name is currently in use.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `app sink`

**Parameters**
* `name` - [in] unique name for the App Sink to create.
* `data_type` - [in]  either `DSL_SINK_APP_DATA_TYPE_SAMPLE` or `DSL_SINK_APP_DATA_TYPE_BUFFER`. See [App Sink data-types](#data-types-provided-by-the-app-sink).
* `client_handler` - [in] client [callback function](#dsl_sink_app_new_data_handler_cb) to be called with each new buffer received.
* `client_data` [in] opaque pointer to client data returned on callback to the `client_handler` function.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_app_new('my-app-sink', DSL_SINK_APP_DATA_TYPE_BUFFER,
    my_data_handler, NULL)
```

<br>

### *dsl_sink_overlay_new*
```C++
DslReturnType dsl_sink_overlay_new(const wchar_t* name, uint display_id,
    uint depth, uint offset_x, uint offset_y, uint width, uint height);
```
The constructor creates a uniquely named Overlay Sink with given offsets and dimensions. Construction will fail if the name is currently in use.

**IMPORTANT:** The Overlay Sink is only available on the Jetson platform.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`render sink`](#render-sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `overlay sink`

**Parameters**
* `name` - [in] unique name for the Overlay Sink to create.
* `display_id` - [in] display Id to overlay, 0 = main display.
* `depth` - [in] depth of the overlay for the given display Id.  
* `x_offset` - [in] offset in the X direction from the upper left corner of the display in pixels.
* `y_offset` - [in] offset in the Y direction from the upper left corner of the display in pixels.
* `width` - [in] width of the Overlay Sink in pixels.
* `height` - [in] height of the Overlay Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

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
The constructor creates a uniquely named Window Sink with given offsets and dimensions. Construction will fail if the name is currently in use. Window Sinks are used to render video onto an XWindows. See [Pipeline XWindow Support](api-pipeline.md#pipeline-xwindow-support) for more information.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`render sink`](#render-sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `window sink`

**Parameters**
* `name` - [in] unique name for the Window Sink to create.
* `x_offset` - [out] offset in the X direction in pixels from the upper left most corner of the parent XWindow.
* `y_offset` - [out] offset in the Y direction in pixels from the upper left most corner of the parent XWindow.
* `width` - [in] width of the Window Sink in pixels.
* `height` - [in] height of the Window Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

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
The constructor creates a uniquely named File Sink. Construction will fail if the name is currently in use. There are two Codec formats - `H.264` and `H.265` - and two video container types - `MPEG4` and `MK4` - supported.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`encode sink`](#encode-sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `file sink`

**Parameters**
* `name` - [in] unique name for the File Sink to create.
* `filepath` - [in] absolute or relative filespec for the media file to write to.
* `codec` - [in] one of the [Codec Types](#codec-types) defined above.
* `container` - [in] one of the [Video Container Types](#video-container-types) defined above.
* `bitrate` - [in] bitrate at which to encode the video. Set to 0 to use the encoder's default (4Mbps).
* `interval` - [in] frame interval at which to code the video. Set to 0 to code every frame.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_file_new('my-file-sink', './my-video.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 0, 0)
```

<br>

### *dsl_sink_record_new*
```C++
DslReturnType dsl_sink_record_new(const wchar_t* name, const wchar_t* outdir, uint codec,
    uint container, uint bitrate, uint interval, dsl_record_client_listener_cb  client_listener);
```
The constructor creates a uniquely named Record Sink. Construction will fail if the name is currently in use. There are two Codec formats - `H.264` and `H.265` - and two video container types - `MPEG4` and `MK4` - supported.

Note: the Sink name is used as the filename prefix, followed by session id and NTP time.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`encode sink`](#encode-sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `record sink`

**Parameters**
* `name` - [in] unique name for the Record Sink to create.
* `outdir` - [in] absolute or relative pathspec for the directory to save the recorded video streams.
* `codec` - [in] one of the [Codec Types](#codec-types) defined above.
* `container` - [in] one of the [Video Container Types](#video-container-types) defined above.
* `bitrate` - [in] bitrate at which to encode the video. Set to 0 to use the encoder's default (4Mbps).
* `interval` - [in] frame interval at which to encode the video. Set to 0 to code every frame.
* `client_listener` - [in] client callback function of type [dsl_record_client_listener_cb ](#dsl_record_client_listener_cb) to be called when a [Recording Event}(#smart-recording-events) occurs.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_new('my-record-sink',
    './', DSL_CODEC_H265, DSL_CONTAINER_MPEG, 0, 0, my-client-recording-event-cb)
```

<br>

### *dsl_sink_rtsp_new*
```C++
DslReturnType dsl_sink_rtsp_new(const wchar_t* name, const wchar_t* host,
     uint udp_port, uint rtmp_port, uint codec, uint bitrate, uint interval);
```
The constructor creates a uniquely named RTSP Sink. Construction will fail if the name is currently in use. There are two Codec formats - `H.264` and `H.265` - supported. The RTSP server is configured when the Pipeline is called to Play. The server is then started and attached to the g-main-loop context once [dsl_main_loop_run](#dsl_main_loop_run) is called. Once attached, the server can accept connections.

**Important Note:** the URI is derived from the device identification, `rtmp_port`, and server mount point which is derived from the unique RTSP Sink name, 

**When the client and DSL application are both running locally:**
```
rtsp://<device-name>.local:<rtmp-port-number>/<rtsp-sink-name>
```
for example:
```
rtsp://my-jetson-device.local:8554/my-rtsp-sink
```

**When the client is running remotely from the DSL application:**
```
rtsp://<user-name>:<password>@<ip-address>:<rtmp-port-number>/<rtsp-sink-name>
```
for example:
```
rtsp://admin:12345@192.168.1.64:8554/my-rtsp-sink
```

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`encode sink`](#encode-sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `rtsp sink`

**Parameters**
* `name` - [in] unique name for the File Sink to create.
* `host` - [in] host name
* `udp_port` - [in] UDP port setting for the RTSP server.
* `rtsp_port` - [in] RTSP port setting for the server.
* `codec` - [in] one of the [Codec Types](#codec-types) defined above.
* `bitrate` - [in] bitrate at which to encode the video. Set to 0 to use the encoder's default (4Mbps).
* `interval` - [in] frame interval at which to encode the video. Set to 0 to code every frame.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
 retVal = dsl_sink_rtsp_new('rtsp-sink', host_uri, 5400, 8554, DSL_CODEC_H264, 4000000,0)
```

<br>

### *dsl_sink_webrtc_new*
```C++
DslReturnType dsl_sink_webrtc_new(const wchar_t* name, const wchar_t* stun_server,
    const wchar_t* turn_server, uint codec, uint bitrate, uint interval);
```
The constructor creates a uniquely named WebRTC Sink. Construction will fail if the name is currently in use. There are two Codec formats - `H.264` and `H.265`. The WebRTC Sink Implements a Signaling Transceiver which is automatically added and removed from the WebSocket Server when added and removed from a Pipeline or Branch. Refer to the [WebSocket Server API Reference](/docs/api-ws-server.md) for more information.

 **IMPORTANT:** The WebRTC Sink implementation requires GStreamer 1.18 or later.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`encode sink`](#encode-sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `webrtc sink`

**Parameters**
* `name` - [in] unique name for the WebRTC Sink to create.
* `stun_server` - [in] STUN server to use - of the form stun://hostname:port. Set to NULL to omit if using TURN server(s).
* `turn_server` - [in] TURN server(s) to use - of the form turn(s)://username:password@host:port. Set to NULL to omit if using a STUN server.
* `codec` - [in] one of the [Codec Types](#codec-types) defined above.
* `bitrate` - [in] bitrate at which to encode the video. Set to 0 to use the encoder's default (4Mbps).
* `interval` - [in] frame interval at which to encode the video. Set to 0 to code every frame.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
STUN_SERVER = "stun://stun.l.google.com:19302"
retval = dsl_sink_webrtc_new('my-webrtc-sink', STUN_SERVER, DSL_CODEC_H264, 0, 0)
```

<br>

### *dsl_sink_message_new*
```C++
DslReturnType dsl_sink_message_new(const wchar_t* name,
    const wchar_t* converter_config_file, uint payload_type,
    const wchar_t* broker_config_file, const wchar_t* protocol_lib,
    const wchar_t* connection_string, const wchar_t* topic);
```
The constructor creates a uniquely named Message Sink. Construction will fail if the name is currently in use. The Message Sink converts ODE metadata -- generated by an [Add Message Meta ODE Action](/docs/api-ode-action.md#dsl_ode_action_message_meta_add_new) -- into a message payload and sends it to the server using a specified communication protocol.

**Important:** refer to the DeepStream Plugin Guide for information on the [Message Converter](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html#gst-nvmsgconv) and [Message Broker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgbroker.html), and for information on configuration and use of the different Protocol Adapters.  

**Note:** refer to [DSL Message Broker API reference](/docs/api-msg-broker.md) and [DSL - Azure MQTT Protocol Adapter Libraries](/docs/proto-lib-azure.md#azure-mqtt-protocol-adapter-libraries) for additional information. 

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `message sink`

**Parameters**
* `name` - [in] unique name for the Message Sink to create.
* `converter_config_file` - [in] absolute or relative path to a message-converter configuration file of type text or csv.
* `payload_type` - [in]  one of the [Message Converter payload schema type constants](#message-converter-payload-schema-types) defined above.
* `broker_config_file` - [in] broker_config_file absolute or relative path to a message-broker configuration file required by nvds_msgapi_* interface.
* `protocol_lib` - [in] absolute or relative path to the protocol adapter that implements the nvds_msgapi_* interface. See the [NVIDIA Installed Protocol Adapter Paths](nvidia-installed-protocol-adapter-paths) defined above.
* `connection_string` - [in] endpoint for communication with server of the format `<name>;<port>;<specifier>`
* `topic` - [in] (optional) message topic for each message sent to the server.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
CONNECTION_STRING = 'HostName=my-hub.azure-devices.net;DeviceId=nano-1;SharedAccessKey=abcd1234EFGH5678ijkl'
retval = dsl_sink_message_new('my-message-sink', converter_config_file, DSL_MSG_PAYLOAD_DEEPSTREAM,
    broker_config_file, NVDS_AZURE_PROTO_LIB, CONNECTION_STRING, '/ode/data')
```

<br>

### *dsl_sink_interpipe_new*
```C++
DslReturnType dsl_sink_interpipe_new(const wchar_t* name,
    boolean forward_eos, boolean forward_events);
```
The constructor creates a new, uniquely named Inter-Pipe Sink component. Construction will fail if the name is currently in use.

**Important!** The Interpipe Services are disabled by default and require additional [install/build steps](/docs/installing-dependencies.md).

Refer to the [Interpipe Services](/docs/overview.md#interpipe-services) overview for more information.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `interpipe sink`

**Parameters**
* `name` - [in] unique name for the Interpipe Sink to create.
* `forward_eos` - [in]  set to true to forward the EOS event to all listeners. False to not forward.
* `forward_events` - [in] set to true to forward downstream events to all listeners (except for EOS). False to not forward.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retVal = dsl_sink_interpipe_new('my-interpipe-sink',
    True, True)
```

<br>

### *dsl_sink_image_multi_new*
```C++
DslReturnType dsl_sink_image_multi_new(const wchar_t* name, 
    const wchar_t* file_path, uint width, uint height,
    uint fps_n, uint fps_d);
```
The constructor creates a new, uniquely named Multi-Image Sink. Construction will fail if the name is currently in use. 

The Sink Encodes each frame into a JPEG image and saves it to a file specified by a given folder/filename-pattern. The size can be scaled by setting the width and height to non-zero values. The rate can be controlled by setting the fps_n and fps_d to non-zero values as well. The maximum rate at which the sink can encode and save is hardware/platform dependent. It is up to the user to manage the maximum setting. Exceeding the limit may lead to corrupted files.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `multi-image sink`

**Parameters**
* `name` - [in] unique name for the Multi-Image Sink to create.
* `file_path` - [in]  use the printf style %d in the absolute or relative path. Example: `"./my_images/image.%04d.jpg"`, will create files in `"./my_images/"` named `"image.0000.jpg"`, `"image.0001.jpg"`, `"image.0002.jpg"` etc.
* `width` - [in] width of the images to save in pixels. Set to 0 for no transcode.
* `height` - [in] height of the images to save in pixels. Set to 0 for no transcode.
* `fps_n` - [in] fps_n frames/second fraction numerator. Set to 0 for no rate-change.
* `fps_d` - [in] fps_d frames/second fraction denominator. Set to 0 for no rate-change.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retVal = dsl_sink_image_multi_new('my-multi-image-sink',
    './frame_%04d.jpg', 640, 360, 1, 10)
```

<br>

### *dsl_sink_frame_capture_new*
```C++
DslReturnType dsl_sink_frame_capture_new(const wchar_t* name, 
    const wchar_t* frame_capture_action);
```
The constructor creates a new, uniquely named Frame-Capture Sink. Construction will fail if the name is currently in use. The Sink is created with an [ODE Frame-Capture Action](/docs/api-ode-action.md#dsl_ode_action_capture_frame_new) which performs the image encoding and saving. All captured frames are copied and buffered in the Sink's processing thread. The encoding and saving of each buffered frame is done in the g-idle-thread context. 

The Application initiates a frame-capture by calling [dsl_sink_frame_capture_initiate](#dsl_sink_frame_capture_initiate).

[Capture-complete-listeners](/docs/api-ode-action.md#dsl_capture_complete_listener_cb) (to notify on completion), [Image Players](/docs/api-player.md) (to auto-play the new image) and [SMTP Mailers](/docs/api-mailer.md) (to mail the new image) can be added to the Capture Action as well.

**Important!** The Frame-Capture Sink Services are disabled by default and require additional [install/build steps](/docs/installing-dependencies.md) to use.

**Note:** The first capture may cause a noticeable short pause to the stream while cuda dependencies are loaded and cached. 

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `frame-capture sink`

**Parameters**
* `name` - [in] unique name for the Frame-Capture Sink to create.
* `frame_capture_action` - [in] unique name of the [ODE Frame-Capture Action](/docs/api-ode-action.md#dsl_ode_action_capture_frame_new) to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retVal = dsl_sink_frame_capture_new('my-frame-capture-sink',
    'my-frame-capture-action')
```

<br>

### *dsl_sink_fake_new*
```C++
DslReturnType dsl_sink_fake_new(const wchar_t* name);
```
The constructor creates a uniquely named Fake Sink. Construction will fail if the name is currently in use.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`sink`](#sink-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `fake sink`

**Parameters**
* `name` - [in] unique name for the Fake Sink to create.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retVal = dsl_sink_fake_new('my-fake-sink')
```

<br>

---

## Destructors
As with all Pipeline components, Sinks are deleted by calling [dsl_component_delete](api-component.md#dsl_component_delete), [dsl_component_delete_many](api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](api-component.md#dsl_component_delete_all).

<br>

---

## Sink Methods

### *dsl_sink_sync_enabled_get*
```C++
DslReturnType dsl_sink_sync_enabled_get(const wchar_t* name, boolean* enabled);
```
This service gets the current `sync` enabled property in use by the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Sink to query.
* `enabled` - [out] true if the `sync` property for the named Sink is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_sink_sync_enabled_get('my-overlay-sink')
```

<br>

### *dsl_sink_sync_enabled_set*
```C++
DslReturnType dsl_sink_sync_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the `sync` enabled property for the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Message Sink to update.
* `enabled` - [in] set to true to enable the `async` property, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_sync_enabled_set('my-overlay-sink', False)
```

<br>

### *dsl_sink_async_enabled_get*
```C++
DslReturnType dsl_sink_async_enabled_get(const wchar_t* name, boolean* enabled);
```
This service gets the current `sync` enabled property in use by the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Sink to query.
* `enabled` - [out] true if the `async` property for the named Sink is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_sink_async_enabled_get('my-overlay-sink')
```

<br>

### *dsl_sink_async_enabled_set*
```C++
DslReturnType dsl_sink_async_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the `async` enabled property for the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Message Sink to update.
* `enabled` - [in] set to true to enable the `async` property, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_async_enabled_set('my-overlay-sink', False)
```

<br>

### *dsl_sink_max_lateness_get*
```C++
DslReturnType dsl_sink_max_lateness_get(const wchar_t* name, int64_t* max_lateness);
```
This service gets the current value of the `max_lateness` property in use by the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Sink to query.
* `max_lateness` - [out] the maximum number of nanoseconds a buffer can be late. -1 equals unlimited lateness.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, max_lateness = dsl_sink_max_lateness_get('my-overlay-sink')
```

<br>

### *dsl_sink_max_lateness_set*
```C++
DslReturnType dsl_sink_max_lateness_set(const wchar_t* name, int64_t max_lateness);
```
This service sets the `max_lateness` property for the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Message Sink to update.
* `max_lateness` - [in] the maximum number of nanoseconds a buffer can be late. Set to -1 for unlimited lateness.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_max_lateness_set('my-overlay-sink', -1)
```

<br>

### *dsl_sink_qos_enabled_get*
```C++
DslReturnType dsl_sink_qos_enabled_get(const wchar_t* name, boolean* enabled);
```
This service gets the current `qos` enabled property in use by the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Sink to query.
* `enabled` - [out] true if the `qos` property for the named Sink is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_sink_qos_enabled_get('my-overlay-sink')
```

<br>

### *dsl_sink_qos_enabled_set*
```C++
DslReturnType dsl_sink_qos_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the `qos` enabled property for the named Sink. See the [Common Sink Properties](#common-sink-properties) for more information.

**Parameters**
* `name` - [in] unique name of the Message Sink to update.
* `enabled` - [in] set to true to enable the `qos` property, false to disable.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_qos_enabled_set('my-overlay-sink', False)
```

<br>

### *dsl_sink_pph_add*
```C++
DslReturnType dsl_sink_pph_add(const wchar_t* name, const wchar_t* handler);
```
This service adds a Pad-Probe-Handler (PPH) to the sink pad of the Named Sink component. The PPH will be invoked on every buffer-ready event for the sink pad. More than one PPH can be added to a single Sink Component.

**Parameters**
 * `name` [in] unique name of the Tee to update.
 * `handler` [in] unique name of the PPH to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_pph_add('my-window-sink', 'my-meter-pph')
```

<br>

### *dsl_sink_pph_remove*
```C++
DslReturnType dsl_sink_pph_remove(const wchar_t* name, const wchar_t* handler);
```
This service removes a Pad-Probe-Handler from a named Sink

**Parameters**
 * `name` [in] unique name of the Sink to update.
 * `handler` [in] unique name of the Pad probe handler to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_pph_remove('my-window-sink', 'my-meter-pph')
```

<br>


### *dsl_sink_app_data_type_get*
```C++
DslReturnType dsl_sink_app_data_type_get(const wchar_t* name, uint* data_type);
```
This service gets the current data-type setting in use by a named App Sink Component.

**Parameters**
* `name` - [in] unique name of the App Sink to query.
* `data_type` - [out] either `DSL_SINK_APP_DATA_TYPE_SAMPLE` or `DSL_SINK_APP_DATA_TYPE_BUFFER`. See [App Sink data-types](#data-types-provided-by-the-app-sink).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, data_type = dsl_sink_app_data_type_get('my-app-sink')
```

<br>

## App Sink Methods

### *dsl_sink_app_data_type_set*
```C++
DslReturnType dsl_sink_app_data_type_set(const wchar_t* name, uint data_type);
```
This service sets the data-type setting for the named App Sink Component to use.

**Parameters**
* `name` - [in] unique name of the App Sink to update.
* `data_type` - [in] either `DSL_SINK_APP_DATA_TYPE_SAMPLE` or `DSL_SINK_APP_DATA_TYPE_BUFFER`. See [App Sink data-types](#data-types-provided-by-the-app-sink).

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_app_data_type_set('my-app-sink', DSL_SINK_APP_DATA_TYPE_BUFFER)
```

<br>

## Render Sink Methods

### *dsl_sink_render_offsets_get*
```C++
DslReturnType dsl_sink_render_offsets_get(const wchar_t* name,
    uint* x_offset, uint* y_offsetY);
```
This service returns the current X and Y offsets for the named Render Sink; Overlay or Window.

**Parameters**
* `name` - [in] unique name of the Render Sink to query.
* `x_offset` - [out] offset in the X direction in pixels from the upper left most corner of the sink's parent.
* `y_offset` - [out] offset in the Y direction in pixels from the upper left most corner of the sink's parent.

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
This service updates the X and Y offsets of a named Render Sink; Overlay or Window. Note: this service will fail if the Sink is currently linked.

**Parameters**
* `name` - [in] unique name of the Render Sink to update.
* `x_offset` - [in] new offset in the X direction in pixels from the upper left most corner of the sink's parent.
* `y_offset` - [in] new offset in the Y direction in pixels from the upper left most corner of the sink's parent.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_render_offests_set('my-overlay-sink', 100, 100)
```

<br>

### *dsl_sink_render_dimensions_get*
```C++
DslReturnType dsl_sink_render_dimensions_get(const wchar_t* name,
    uint* width, uint* height);
```
This service returns the current dimensions for the named Render Sink; Overlay or Window

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
DslReturnType dsl_sink_render_dimensions_set(const wchar_t* name,
    uint width, uint height);
```
This service updates the dimensions of a named Render Sink; Overlay or Window. This service will fail if the Sink is currently linked.

**Parameters**
* `name` - [in] unique name of the Overlay Sink to update.
* `width` - [in] new width setting for the Render Sink in pixels.
* `height` - [in] new height setting for the Render Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_render_dimensions_set('my-overlay-sink', 1280, 720)
```

<br>

## Window Sink Methods
### *dsl_sink_window_handle_get*
```C++
DslReturnType dsl_sink_window_handle_get(const wchar_t* name, uint64_t* handle);
```
This service returns the current XWindow handle in use by the named Window Sink. The handle is set to `Null` on Wink creation and will remain `Null` until,
1. The Sink creates an internal XWindow synchronized on Transition to a state of playing, or
2. The Client Application passes an XWindow handle into the Pipeline by calling [dsl_sink_window_handle_set](#dsl_sink_window_handle_set).

**Parameters**
* `name` - [in] unique name for the Window Sink to query.
* `handle` - [out] XWindow handle in use by the named Window Sink

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, handle = dsl_sink_window_handle_get('my-window-sink')
```
<br>

### *dsl_sink_window_handle_set*
```C++
DslReturnType dsl_sink_window_handle_set(const wchar_t* name, uint64_t handle);
```
This service sets the XWindow for the named Window Sink to use. Setting the handle to 0 will clear any previously provided handle. The Window Sink will create its own XWindow if the handle is set to 0.

**Parameters**
* `name` - [in] unique name for the Window Sink to update.
* `handle` - [in] XWindow handle for the Window-Sink to use, or 0 to clear the value.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_sink_window_handle_set('my-window-sink', handle)
```
<br>

### *dsl_sink_window_force_aspect_ratio_get*
```C++
DslReturnType dsl_sink_window_force_aspect_ratio_get(const wchar_t* name,
    boolean* force);
```
This service returns the `force-aspect-ratio` property setting for the named Window Sink. The Sink's aspect ratio will be maintained on Window resize if set.

**Parameters**
* `name` - [in] unique name of the Window Sink to query.
* `force` - [out] true if the property is set, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

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
This service sets the `force-aspect-ratio` property for the named Window Sink. The Sink's aspect ratio will be maintained on Window resize if set. This service will fail if the Sink is currently linked.

**Parameters**
* `name` - [in] unique name of the Window Sink to update.
* `force` - [in] set true to force the aspect ratio on window resize., false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_force_aspect_ratio_get('my-window-sink', True)
```

<br>

### *dsl_sink_window_fullscreen_enabled_get*
```C++
DslReturnType dsl_sink_window_fullscreen_enabled_get(const wchar_t* name, 
    boolean* enabled);
```
This service gets the current full-screen-enabled setting for the named Window Sink's XWindow.

**Parameters**
* `name` - [in] unique name of the Window Sink to query
* `enabled` - [out] true if the XWindow's full-screen mode is enabled, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_sink_window_fullscreen_enabled_get('my-window-sink')
```

<br>

### *dsl_sink_window_fullscreen_enabled_set*
```C++
DslReturnType dsl_sink_window_fullscreen_enabled_set(const wchar_t* name, 
    boolean enabled);
```
This service sets the current full-screen-enabled setting for the Window Sink's XWindow.

**Parameters**
* `name` - [in] unique name of the Window Sink to update
* `enabled` - [in] set to true to enable the XWindow's full-screen mode, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_fullscreen_enabled_set('my-window-sink', enabled=True)
```

<br>

### *dsl_sink_window_key_event_handler_add*
```C++
DslReturnType dsl_sink_window_key_event_handler_add(const wchar_t* name, 
    dsl_sink_window_key_event_handler_cb handler, void* client_data);
```
This service adds a callback function of type [dsl_sink_window_key_event_handler_cb](#dsl_sink_window_key_event_handler_cb) to a named Window Sink. The function will be called on every XWindow `KeyReleased` event with Key string and the client provided `client_data`. Multiple callback functions can be registered with one Window Sink, and one callback function can be registered with multiple Window Sinks.

**Parameters**
* `name` - [in] unique name of the Window Sink to update.
* `handler` - [in] XWindow event handler callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful  addition. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def key_event_handler(key_string, client_data):
    print('key pressed = ', key_string)
   
retval = dsl_sink_window_key_event_handler_add('my-window-sink',
    key_event_handler, None)
```

<br>

### *dsl_sink_window_key_event_handler_remove*
```C++
DslReturnType dsl_sink_window_key_event_handler_remove(const wchar_t* name, 
    dsl_sink_window_key_event_handler_cb handler);
```
This service removes a function of type [dsl_sink_window_key_event_handler_cb](#dsl_sink_window_key_event_handler_cb) that was previously added with [dsl_sink_window_key_event_handler_add](#dsl_sink_window_key_event_handler_add).

**Parameters**
* `name` - [in] unique name of the Window Sink to update
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_key_event_handler_remove('my-window-sink',
    key_event_handler)
```

<br>

### *dsl_sink_window_button_event_handler_add*
```C++
DslReturnType dsl_sink_window_button_event_handler_add(const wchar_t* name, 
    dsl_sink_window_button_event_handler_cb handler, void* client_data);
```
This service adds a callback function of type [dsl_sink_window_button_event_handler_cb](#dsl_sink_window_button_event_handler_cb) to a named Window Sink. The function will be called on every XWindow `ButtonPressed` event with Button ID, X and Y positional offsets, and the client provided `client_data`. Multiple callback functions can be registered with one Window Sink, and one callback function can be registered with multiple Window Sinks.

**Parameters**
* `name` - [in] unique name of the Window Sink to update.
* `handler` - [in] XWindow event handler callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful  addition . One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def button_event_handler(button, xpos, ypos, client_data):
    print('button = ', button)
    print('xpos = ', xpos)
    print('ypos = ', ypos)
   
retval = dsl_sink_window_button_event_handler_add('my-window-sink',
    button_event_handler, None)
```

<br>

### *dsl_sink_window_button_event_handler_remove*
```C++
DslReturnType dsl_sink_window_button_event_handler_remove(const wchar_t* name, 
    dsl_sink_window_button_event_handler_cb handler);
```
This service removes a function of type [dsl_sink_window_button_event_handler_cb](#dsl_sink_window_button_event_handler_cb) that was previously added with [dsl_sink_window_button_event_handler_add](#dsl_sink_window_button_event_handler_add).

**Parameters**
* `name` - [in] unique name of the Window Sink to update.
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_button_event_handler_remove('my-window-sink',
    button_event_handler)
```

<br>

### *dsl_sink_window_delete_event_handler_add*
```C++
DslReturnType dsl_sink_window_delete_event_handler_add(const wchar_t* name, 
    dsl_sink_window_delete_event_handler_cb handler, void* client_data);
```
This service adds a callback function of type [dsl_sink_window_delete_event_handler_cb](#dsl_sink_window_delete_event_handler_cb) to a named Window Sink. The function will be called on when the XWindow is closed/deleted. Multiple callback functions can be registered with one Window Sink, and one callback function can be registered with multiple Window Sink.

**Parameters**
* `name` - [in] unique name of the Window Sink to update.
* `handler` - [in] XWindow event handler callback function to add.
* `client_data` - [in] opaque pointer to user data returned to the handler when called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful addition. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def xwindow_delete_event_handler(client_data):
    dsl_pipeline_stop('my-pipeline')
    dsl_main_loop_quit()

retval = dsl_sink_window_delete_event_handler_add('my-window-sink',
    xwindow_delete_event_handler, None)
```

<br>

### *dsl_sink_window_delete_event_handler_remove*
```C++
DslReturnType dsl_sink_window_delete_event_handler_remove(const wchar_t* name, 
    dsl_sink_window_delete_event_handler_cb handler);
```
This service removes a function of type [dsl_sink_window_delete_event_handler_cb](#dsl_sink_window_delete_event_handler_cb) that was previously added with [dsl_sink_window_delete_event_handler_add](#dsl_sink_window_delete_event_handler_add).

**Parameters**
* `name` - [in] unique name of the Window Sink to update
* `handler` - [in] XWindow event handler callback function to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_window_delete_event_handler_remove('my-pipeline',
    xwindow_delete_event_handler)
```

<br>

## Encode Sink Methods

### *dsl_sink_encode_settings_get*
```C++
DslReturnType dsl_sink_encode_settings_get(const wchar_t* name,
    uint* codec, uint* bitrate, uint* interval);
```
This service returns the current bitrate and interval settings for the named Encode Sink; File, Record, RTSP OR WebRTC.

**Parameters**
* `name` - [in] unique name of the Encode Sink to query.
* `codec` - [out] current codec in use, one of the [Codec Types](#codec-types) defined above.
* `bitrate` - [out] current bit rate at which to code the video.
* `interval` - [out] current frame interval at which to encode the video. 0 equals code every frame

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, bitrate, interval = dsl_sink_encode_settings_get('my-file-sink')
```

<br>

### *dsl_sink_encode_settings_set*
```C++
DslReturnType dsl_sink_encode_settings_set(const wchar_t* name,
    uint codec, uint bitrate, uint interval);
```
This service sets the bitrate and interval settings for the named Encode Sink; File, Record, RTSP OR WebRTC. The service will fail if the Encode Sink is currently linked.

**Parameters**
* `name` - [in] unique name of the Encode Sink to update.
* `codec` - [in] new codec to use, one of the [Codec Types](#codec-types) defined above.
* `bitrate` - [in] bitrate at which to encode the video. Set to 0 to use the encoder's default (4Mbps).
* `interval` - [in] new frame interval at which to encode the video. Set to 0 to code every frame.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_encode_settings_set('my-file-sink', 4000000, 1)
```

<br>

### *dsl_sink_encode_dimensions_get*
```C++
DslReturnType dsl_sink_encode_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service gets the current input dimensions for the named Encode Sink; [File](#dsl_sink_file_new), [Smart-Record](#dsl_sink_record_new), [RTSP](#dsl_sink_rtsp_new), or [WebRTC](dsl_sink_webrtc_new).  Values of 0 indicate NO scaling.

The dimensions, if non-zero, are used to scale the video input into the Sinks's encoder element. 

**Parameters**
* `name` - [in] unique name of the Encode Sink to query.
* `width` - [out] current width of the Encode Sink input in pixels.
* `height` - [out] current height of the Encode Sink input pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_sink_encode_dimensions_get('my-rtsp-sink')
```

<br>

### *dsl_sink_encode_dimensions_set*
```C++
DslReturnType dsl_sink_encode_dimensions_set(const wchar_t* name, 
    uint width, uint height);
```
This service updates the dimensions of a named Encode Sink; [File](#dsl_sink_file_new), [Smart-Record](#dsl_sink_record_new), [RTSP](#dsl_sink_rtsp_new), or [WebRTC](dsl_sink_webrtc_new).  Set the values to 0 to indicate NO scaling.


**Parameters**
* `name` - [in] unique name of the Overlay Sink to update.
* `width` - [in] new width setting for the Encode Sink in pixels.
* `height` - [in] new height setting for the Encode Sink in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_encode_dimensions_set('my-rtsp-sink', 1280, 720)
```

<br>

## Smart-Record Sink Methods

### *dsl_sink_record_session_start*
```C++
DslReturnType dsl_sink_record_session_start(const wchar_t* name,
    uint start, uint duration, void* client_data);
```
This service starts a new recording session for the named Record Sink

**Parameters**
 * `name` [in] unique name of the Record Sink to start the session.
 * `start` [in] start time in seconds before the current time should be less than the video cache size.
 * `duration` [in] in seconds from the current time to record.
 * `client_data` [in] opaque pointer to client data returned on callback to the client listener function provided on Sink creation.

**Returns**
* `DSL_RESULT_SUCCESS` on successful start. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_session_start('my-record-sink', 15, 900, None)
```

<br>

### *dsl_sink_record_session_stop*
```C++
DslReturnType dsl_sink_record_session_stop(const wchar_t* name);
```
This service stops a current recording in session.

**Parameters**
* `name` [in] unique name of the Record Sink to stop.

**Returns**
* `DSL_RESULT_SUCCESS` on successful Stop. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_session_stop('my-record-sink')
```

<br>

### *dsl_sink_record_outdir_get*
```C++
DslReturnType dsl_sink_record_outdir_get(const wchar_t* name, const wchar_t** outdir);
```
This service returns the video recording output directory.

**Parameters**
 * `name` [in] name of the Record Sink to query.
 * `outdir` - [out] absolute pathspec for the directory to save the recorded video streams.

**Returns**
 * `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure.

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
 * `name` [in] name of the Record Sink to update.
 * `outdir` - [in] absolute or relative pathspec for the directory to save the recorded video streams.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

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
 * `name` [in] name of the Record Sink to query.
 * `container` - [out] one of the [Video Container Types](#video-container-types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure.

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
 * `name` [in] name of the Record Sink to update.
 * `container` - [in] on of the [Video Container Types](#video-container-types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

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
 * `name` [in] name of the Record Sink to query.
 * `cache_size` [out] current cache size setting.

**Returns**
* `DSL_RESULT_SUCCESS` on successful Query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, cache_size = dsl_sink_record_cache_size_get('my-record-sink')
```

<br>

### *dsl_sink_record_cache_size_set*
```C++
DslReturnType dsl_sink_record_cache_size_set(const wchar_t* name, uint cache_size);
```
This service sets the video recording cache size in units of seconds. A fixed size cache is created when the Pipeline is linked and played. The default cache size is set to DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC.

**Parameters**
 * `name` [in] name of the Record Sink to query.
 * `cache_size` [in] new cache size setting to use on Pipeline play.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_cache_size_set('my-record-sink', 15)
```

<br>

### dsl_sink_record_dimensions_get
```C++
DslReturnType dsl_sink_record_dimensions_get(const wchar_t* name, uint* width, uint* height);
```
This service returns the dimensions, width and height, used for the video recordings. Values of zero (default) indicate no-transcode.

**Parameters**
 * `name`[in] name of the Record Sink to query.
 * `width`[out] current width of the video recording in pixels.
 * `height` [out] current height of the video recording in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, width, height = dsl_sink_record_dimensions_get('my-record-sink')
```

<br>

### *dsl_sink_record_dimensions_set*
```C++
DslReturnType dsl_sink_record_dimensions_set(const wchar_t* name, uint width, uint height);
```
This service sets the dimensions, width and height, for the video recordings created. Values of zero (default) indicate no-transcode.

**Parameters**
 * `name` [in] name of the Record Sink to update.
 * `width` [in] width to set the video recording in pixels.
 * `height` [in] height to set the video in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_dimensions_set('my-record-sink', 1280, 720)
```

<br>

### *dsl_sink_record_is_on_get*
```C++
DslReturnType dsl_sink_record_is_on_get(const wchar_t* name, boolean* is_on);
```
This service returns the current recording state of the Record Sink.

**Parameters**
 * `name` [in] name of the Record Sink to query.
 * `is_on` [out] true if the Record Sink is currently recording a session, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, is_on = dsl_sink_record_is_on_get('my-record-sink')
```

<br>

### *dsl_sink_record_reset_done_get*
```C++
DslReturnType dsl_sink_record_reset_done_get(const wchar_t* name, boolean* reset_done);
```
This service returns the current state of the Reset Done flag.

**Parameters**
 * `name` [in] name of the Record Sink to query.
 * `reset_done` [out] true if Reset has been done, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, reset_done = dsl_sink_record_reset_done_get('my-record-sink')
```

<br>

### *dsl_sink_record_video_player_add*
```C++
DslReturnType dsl_sink_record_video_player_add(const wchar_t* name,
    const wchar_t* player)
```
This service adds a [Video Player](/docs/api-player.md), Render or RTSP type, to a named Recording Sink. Once added, each recorded video's file_path will be added (or queued) with the Video Player to be played according to the Players settings.

**Parameters**
 * `name` [in] name of the Record Sink to update.
 * `player` [in] name of the Video Player to add.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_video_player_add('my-record-sink, 'my-video-render-player')
```

<br>

### *dsl_sink_record_video_player_remove*
```C++
DslReturnType dsl_sink_record_video_player_remove(const wchar_t* name,
    const wchar_t* player)
```
This service removes a [Video Player](/docs/api-player.md), Render or RTSP type, from a named Recording Sink.

**Parameters**
 * `name` [in] name of the Record Sink to update
 * `player` [in] player name of the Video Player to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_video_player_remove('my-record-sink', 'my-video-render-player'
```

### *dsl_sink_record_mailer_add*
```C++
DslReturnType dsl_sink_record_mailer_add(const wchar_t* name,
    const wchar_t* mailer, const wchar_t* subject);
```
This service adds a [Mailer](/docs/api-mailer.md) to a named Recording Sink. Once added, the file_name, location, and specifics of each recorded video will be emailed by the Mailer according to its current settings.

**Parameters**
 * `name` [in] name of the Record Sink to update.
 * `mailer` [in] name of the Mailer to add.
 * `subject` [in] subject line to use for all outgoing mail.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_mailer_add('my-record-sink, 'my-mailer', "New Recording Complete!")
```

<br>

### *dsl_sink_record_mailer_remove*
```C++
DslReturnType dsl_sink_record_mailer_remove(const wchar_t* name,
    const wchar_t* mailer)
```
This service removes a [Mailer](/docs/api-mailer.md) from a named Recording Sink.

**Parameters**
 * `name` [in] name of the Record Sink to update.
 * `mailer` [in] name of the Mailer to remove.

**Returns**
* `DSL_RESULT_SUCCESS` on successful remove. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_record_mailer_remove('my-record-sink', 'my-mailer')
```

<br>

## RTSP Sink Methods

### *dsl_sink_rtsp_server_settings_get*
This service returns the current RTSP video codec and Port settings for the named RTSP Sink.
```C++
DslReturnType dsl_sink_rtsp_server_settings_get(const wchar_t* name,
    uint* udp_port, uint* rtsp_port);
```
**Parameters**
* `name` - [in] unique name of the RTSP Sink to query.
* `udp_port` - [out] the current UDP Port number in use.
* `rtsp_port` - [out] the current RTSP Port number in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, udp_port, rtsp_port = dsl_sink_rtsp_server_settings_get('my-rtsp-sink')
```

<br>

## WebRTC Sink Methods

### *dsl_sink_webrtc_connection_close*
```C++
DslReturnType dsl_sink_webrtc_connection_close(const wchar_t* name);
```
This service closes a currently open WebSocket connection for the named WebRTC Sink.

 **IMPORTANT:** The WebRTC Sink implementation requires GStreamer 1.18 or later.

**Parameters**
* `name` [in] unique name of the WebRTC Sink to update.

**Returns**  `DSL_RESULT_SUCCESS` on successful close. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_webrtc_connection_close('my-webrtc-sink')
```

<br>

### *dsl_sink_webrtc_servers_get*
```C++
DslReturnType dsl_sink_webrtc_servers_get(const wchar_t* name,
    const wchar_t** stun_server, const wchar_t** turn_server);
```
This service queries a named WebRTC Sink component for its current STUN or TURN server(s) in use.

 **IMPORTANT:** The WebRTC Sink implementation requires DS 1.18.0 or later.

**Parameters**
* `name` [in] unique name of the WebRTC Sink to query.
* `stun_server` [out] current STUN server in use, NULL if omitted.
* `turn_server` [out] current TURN server in use, NULL if omitted.

**Returns**  `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, stun_server, turn_server = dsl_sink_webrtc_servers_get('my-webrtc-sink')
```

<br>

### *dsl_sink_webrtc_servers_set*
```C++
DslReturnType dsl_sink_webrtc_servers_set(const wchar_t* name,
    const wchar_t* stun_server, const wchar_t* turn_server);
```
This service updates a named WebRTC Sink component with either a new STUN or TURN server(s) to use.

 **IMPORTANT:** The WebRTC Sink implementation requires GStreamer 1.18 or later.

**Parameters**
* `name` [in] unique name of the WebRTC Sink to query.
* `stun_server` - [in] STUN server to use - of the form stun://hostname:port. Set to NULL to omit if using TURN server(s).
* `turn_server` - [in] TURN server(s) to use - of the form turn(s)://username:password@host:port. Set to NULL to omit if using a STUN server.

**Returns**  `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_webrtc_servers_get('my-webrtc-sink', NULL, turn_server)
```

<br>

### *dsl_sink_webrtc_client_listener_add*
```C++
DslReturnType dsl_sink_webrtc_client_listener_add(const wchar_t* name,
    dsl_sink_webrtc_client_listener_cb listener, void* client_data);
```
This service adds a callback function of type [dsl_sink_webrtc_client_listener_cb](#dsl_sink_webrtc_client_listener_cb) to the WebRTC Sink. The function will be called on all changes of WebSocket connection state. Multiple callback functions can be added to the WebRTC Sink.

 **IMPORTANT:** The WebRTC Sink implementation requires GStreamer 1.18 or later.

**Parameters**
* `name` [in] unique name of the WebRTC Sink to update.
* `listener` - [in] listener callback function to add.
* `client_data` - [in] opaque pointer to user data, returned to the listener on call back

**Returns**  `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_webrtc_client_listener_add('my-webrtc-sink', client_listener_cb, None)
```

<br>

### *dsl_sink_webrtc_client_listener_remove*
```C++
DslReturnType dsl_sink_webrtc_client_listener_remove(const wchar_t* name,
    dsl_sink_webrtc_client_listener_cb listener);
```
This service removes a callback function of type [dsl_sink_webrtc_client_listener_cb](#dsl_sink_webrtc_client_listener_cb) from the WebRTC Sink.

 **IMPORTANT:** The WebRTC Sink implementation requires GStreamer 1.18 or later.

**Parameters**
* `name` [in] unique name of the WebRTC Sink to update.
* `listener` - [in] listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_webrtc_client_listener_remove('my-webrtc-sink', client_listener_cb)
```

<br>

## IOT Message Converter Sink Methods
### *dsl_sink_message_converter_settings_get*
```C++
DslReturnType dsl_sink_message_converter_settings_get(const wchar_t* name,
    const wchar_t** converter_config_file, uint* payload_type);
```
This service gets the current Message Converter settings in use by the named Message Sink.

**Important** refer to the Deepstream Plugin Guide for information on the [Message Converter](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html#gst-nvmsgconv) configuration file settings.

**Parameters**
* `name` - [in] unique name of the Encode Sink to query.
* `converter_config_file` - [out] absolute file-path to the current Message Converter config file in use.
* `payload_type` - [out] the current payload schema type in use, one of the [Message Converter payload schema type constants](#message-converter-payload-schema-types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, converter_config_file, payload_type = dsl_sink_message_converter_settings_get('my-message-sink')
```

<br>

### *dsl_sink_message_converter_settings_set*
```C++
DslReturnType dsl_sink_message_converter_settings_set(const wchar_t* name,
    const wchar_t* converter_config_file, uint payload_type);
```
This service sets the Message Converter settings to be used by the named Message Sink.

**Important** refer to the DeepStream Plugin Guide for information on the [Message Converter](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html#gst-nvmsgconv) configuration file settings.

**Parameters**
* `name` - [in] unique name of the Encode Sink to update.
* `converter_config_file` - [in] relative or absolute file-path to a Message Converter config file to use.
* `payload_type` - [in] the current payload schema type in use, one of the [Message Converter payload schema type constants](#message-converter-payload-schema-types) defined above.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_message_converter_settings_get('my-message-sink',
    converter_config_file, DSL_MSG_PAYLOAD_DEEPSTREAM)
```

<br>

### *dsl_sink_message_broker_settings_get*
```C++
DslReturnType dsl_sink_message_broker_settings_get(const wchar_t* name,
    const wchar_t** broker_config_file, const wchar_t** protocol_lib,
    const wchar_t** connection_string, const wchar_t** topic);
```
This service gets the current Message Broker settings in use by the named Message Sink.

**Important** refer to the DeepStream Plugin Guide for information on the [Message Broker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgbroker.html) configuration file and connection string, and for information on configuration and use of the different Protocol Adapters.  

**Parameters**
* `name` - [in] unique name of the Message Sink to query.
* `broker_config_file` - [out] absolute file-path to the current Message Broker config file in use.
* `protocol_lib` - [out] absolute file-path to the current protocol adapter library in use.
* `connection_string` - [out] current connection string in use.
* `topic` - [out] (optional) current message topic in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, broker_config_file, protocol_lib, connection_string, topic =
        dsl_sink_message_broker_settings_get('my-message-sink')
```

<br>

### *dsl_sink_message_broker_settings_set*
```C++
DslReturnType dsl_sink_message_broker_settings_set(const wchar_t* name,
    const wchar_t* broker_config_file, const wchar_t* protocol_lib,
    const wchar_t* connection_string, const wchar_t* topic);
```
This service sets the Message Broker settings to be used by the named Message Sink.

**Important** refer to the DeepStream Plugin Guide for information on the [Message Broker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgbroker.html) configuration file and connection string, and for information on configuration and use of the different Protocol Adapters.  

**Parameters**
* `name` - [in] unique name of the Message Sink to update.
* `broker_config_file` - [in] absolute or relative file-path to a new Message Broker config file to use.
* `protocol_lib` - [in] absolute or relative file-path to a new protocol adapter library to use.
* `connection_string` - [in] new connection string to use.
* `topic` - [in] (optional) new message topic to use for all messages sent.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_message_broker_settings_get('my-message-sink',
    broker_config_file, protocol_lib, connection_string, new_topic)
```

<br>

## Interpipe Sink Methods

### *dsl_sink_interpipe_forward_settings_get*
```C++
DslReturnType dsl_sink_interpipe_forward_settings_get(const wchar_t* name,
    boolean* forward_eos, boolean* forward_events); 
```
This service gets the current forward settings in use for the named Interpipe Sink component.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to query.
* `forward_eos` - [out] if true, the EOS event will be forwarded to all listeners. False otherwise.
* `forward_events` - [out] if true, all downstream events will be forwarded to all the listeners (except for EOS). False otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, forward_eos, forward_events = dsl_sink_interpipe_forward_settings_get(
    'my-interpipe-sink')
```

<br>

### *dsl_sink_interpipe_forward_settings_set*
```C++
DslReturnType dsl_sink_interpipe_forward_settings_set(const wchar_t* name,
    boolean forward_eos, boolean forward_events); 
```
This service sets the forward settings for named Interpipe Sink component.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to update.
* `forward_eos` - [in] set to true to forward the EOS event to all listeners. False to not forward.
* `forward_events` - [in] set to true to forward downstream events to all listeners (except for EOS). False to not forward.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_interpipe_forward_settings_set('my-interpipe-sink',
    True, True)
```

<br>

### *dsl_sink_interpipe_num_listeners_get*
```C++
DslReturnType dsl_sink_interpipe_num_listeners_get(const wchar_t* name,
    uint* num_listeners);
```
This service gets the current number of Interpipe Sources listening to the named Interpipe Sink component.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to query.
* `num_listeners` - [out]  current number of Interpipe Sources listening.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, num_listeners = dsl_sink_interpipe_num_listeners_get('my-interpipe-sink')
```

<br>

## Multi-Image Sink Methods

### *dsl_sink_image_multi_file_path_get*
```C++
DslReturnType dsl_sink_image_multi_file_path_get(const wchar_t* name, 
    const wchar_t** file_path);
```
This service gets the current output file-path in use by the named Multi-Image Sink component.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to query.
* `file_path` - [out] current file-path in use by the Multi-Image Sink.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, file_path = dsl_sink_image_multi_file_path_get('my-multi-image-sink')
```

<br>

### *dsl_sink_image_multi_file_path_set*
```C++
DslReturnType dsl_sink_image_multi_file_path_set(const wchar_t* name, 
    const wchar_t* file_path);
```
This service sets the output file-path for the named Multi-Image Sink to use.

**Parameters**
* `name` - [in] unique name of the Multi-Image Sink to update.
* `file_path` - [in]  use the printf style %d in the absolute or relative path. Example: `"./my_images/image.%04d.jpg"`, will create files in `"./my_images/"` named `"image.0000.jpg"`, `"image.0001.jpg"`, `"image.0002.jpg"` etc.


**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_image_multi_file_path_set('my-multi-image-sink',
    "./my_images/image.%04d.jpg")
```

<br>

### *dsl_sink_image_multi_dimensions_get*
```C++
DslReturnType dsl_sink_image_multi_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service gets the current dimensions in use by the named Multi-Image Sink component. Values of 0 indicate no transcode.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to query.
* `width` - [out] current width to save images in pixels.
* `height` - [out] current height to save images in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, width, height = dsl_sink_image_multi_dimensions_get(
    'my-multi-image-sink')
```

<br>

### *dsl_sink_image_multi_dimensions_set*
```C++
DslReturnType dsl_sink_image_multi_dimensions_set(const wchar_t* name, 
    uint width, uint height);
```
This service sets the dimensions for the named Multi-Image Sink to use.

**Parameters**
* `name` - [in] unique name of the Multi-Image Sink to update.
* `width` - [in] new width setting in units of pixels. Set to 0 for no transcode. 
* `height` - [in] new width setting in units of pixels. Set to 0 for no transcode. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_image_multi_dimensions_set('my-multi-image-sink',
    640, 360)
```

<br>

### *dsl_sink_image_multi_frame_rate_get*
```C++
DslReturnType dsl_sink_image_multi_frame_rate_get(const wchar_t* name, 
    uint* fps_n, uint* fps_d);
```
This service gets the current frame-rate in use by the named Multi-Image Sink component. Values of 0 indicate no rate-change.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to query.
* `fps_n` - [out] current frames per second numerator.
* `fps_d` - [out] current frames per second denominator.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, fps_n, fps_d = dsl_sink_image_multi_frame_rate_get(
    'my-multi-image-sink')
```

<br>

### *dsl_sink_image_multi_frame_rate_set*
```C++
DslReturnType dsl_sink_image_multi_frame_rate_set(const wchar_t* name, 
    uint fps_n, uint fps_d);
```
This service sets the frame-rate for the named Multi-Image Sink to use.

**Parameters**
* `name` - [in] unique name of the Multi-Image Sink to update.
* `fps_n` - [in] new frames per second numerator. Set to 0 for no rate-change. 
* `fps_d` - [in] new frames per second denominator. Set to 0 for no rate-change. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_image_multi_frame_rate_set(my-multi-image-sink',
    1, 30)
```

<br>

### *dsl_sink_image_multi_file_max_get*
```C++
DslReturnType dsl_sink_image_multi_file_max_get(const wchar_t* name, 
    uint* max);
```
This service gets the current max-file setting in use by the named Multi-Image Sink component. Once the maximum is reached, old files start to be deleted to make room for new ones. Default = 0 - no max.

**Parameters**
* `name` - [in] unique name of the Interpipe Sink to query.
* `max` - [out] current max-file setting in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, max = dsl_sink_image_multi_file_max_get(
    'my-multi-image-sink')
```

<br>

### *dsl_sink_image_multi_file_max_set*
```C++
DslReturnType dsl_sink_image_multi_file_max_set(const wchar_t* name, 
    uint max);
```
This service sets the max-file setting for named Multi-Image Sink to use. Once the maximum is reached, old files start to be deleted to make room for new ones. Default = 0 - no max.

**Parameters**
* `name` - [in] unique name of the Multi-Image Sink to update.
* `max` - [in] new max-files setting to use. Set to 0 for no maximum. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_image_multi_file_max_set('my-multi-image-sink', 100)
```

<br>

## Frame-Capture Sink Methods

### *dsl_sink_frame_capture_initiate*
```C++
DslReturnType dsl_sink_frame_capture_initiate(const wchar_t* name);
```
This service initiates a "frame-capture" action to capture the next buffer processed by the named Frame-Capture Sink.

 All captured frames are copied and buffered in the Sink's processing thread. The encoding and saving of each buffered frame is done in the g-idle-thread context. 
 
 Note: The first capture may cause a noticeable short pause to the stream while cuda dependencies are loaded and cached.  

**Parameters**
* `name` - [in] unique name of the Frame-Capture Sink to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_sink_frame_capture_initiate('my-frame-capture-sink')
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
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer, Remxer, and Splitter Tees](/docs/api-tee.md)
* [On-Screen Display](/docs/api-osd.md)
* **Sink**
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Acton](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)
