# Source API Reference
Sources are the head components for all DSL [Pipelines](/docs/api-pipeline.md) and [Players](docs/api-player.md). Pipelines must have at least one Source and one [Sink](/docs/api-sink.md) to transition to a state of PLAYING.  All Sources are derived from the "Component" class, therefore all [component methods](/docs/api-component.md) can be called with any Source.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── `source`

### Source Construction and Destruction
Sources are created by calling one of the type-specific [source constructors](#constructors). As with all components, Sources must be uniquely named from all other Pipeline components created.

Sources are added to a Pipeline by calling [`dsl_pipeline_component_add`](/docs/api-pipeline.md#dsl_pipeline_component_add) or [`dsl_pipeline_component_add_many`](/docs/api-pipeline.md#dsl_pipeline_component_add_many) and removed with [`dsl_pipeline_component_remove`](/docs/api-pipeline.md#dsl_pipeline_component_remove), [`dsl_pipeline_component_remove_many`](/docs/api-pipeline.md#dsl_pipeline_component_remove_many), or [`dsl_pipeline_component_remove_all`](/docs/api-pipeline.md#dsl_pipeline_component_remove_all).

When adding multiple sources to a Pipeline, all must have the same `is_live` setting; `true` or `false`. The add services will fail on the first exception. A Source's `is_live` setting can be queried by calling [`dsl_source_is_live`](#dsl_source_is_live).

The relationship between Pipelines and Sources is one-to-many. Once added to a Pipeline, a Source must be removed before it can be used with another. All sources are deleted by calling [`dsl_component_delete`](api-component.md#dsl_component_delete), [`dsl_component_delete_many`](api-component.md#dsl_component_delete_many), or [`dsl_component_delete_all`](api-component.md#dsl_component_delete_all). Calling a delete service on a Source `in-use` by a Pipeline will fail.

### Source Stream-Ids and Unique-Ids
All Sources are assigned two identifiers when added to a Pipeline.
#### **`stream-id`**
The stream-id identifies the Source's stream within a unique Pipeline. Stream-ids are assigned to the Sources in the order they are added to the Pipeline starting with 0. The stream-id identifies the Streammuxer sink (input) pad-id the Source will connect with when the Pipeline transitions to a state of PLAYING. When using multiple Pipelines, the first source added to each Pipeline will be given same stream-id=0, meaning that stream-ids are only unique for a given Pipeline. A source's stream-id can be queried by calling [`dsl_source_stream_id_get`](#dsl_source_stream_id_get). 

When not added to a Pipeline, a Source's `stream-id` will be set to `-1`. 

#### **`unique-Id`**
The unique-id uniquely identifies a Source from all other Sources. The Source's unique-id is calculated by offsetting the Source's stream-id with the Pipeline's unique 0-based id.  The following constant defines the positional offset for the Pipeline's unique-id.
```c
#define DSL_PIPELINE_SOURCE_UNIQUE_ID_OFFSET_IN_BITS  16

unique-id = unique-pipeline-id << DSL_PIPELINE_SOURCE_UNIQUE_ID_OFFSET_IN_BITS | stream-id
```
Examples:
```
unique-id    | description
-------------|---------------------------
0x00000000   | pipeline-id:0, stream-id:0
0x00010000   | pipeline-id:1, stream-id:0
0x00030002   | pipeline-id:3, stream-id:2
```
A source's unique-id can be queried by calling [`dsl_source_unique_id_get`](#dsl_source_unique_id_get). A Source's unique name can be obtained by calling [`dsl_source_name_get`](#dsl_source_name_get) with a unique source-id. This can be important when reading source-id's while processing frame-metadata in a [Custom PPH](/docs/api-pph.md#custom-pad-probe-handler).

When not added to a Pipeline, a Source's `unique-id` will be set to `-1`. 

### Source Services
A Source can be queried for it's media type -- `video/x-raw`, `audio/x-raw`, or both -- by calling [`dsl_source_media_type_get`](#dsl_source_media_type_get). A Source's framerate can be queried by calling [`dsl_source_framerate get`](#dsl_source_framerate_get). Some Sources need to transition to a state of `PLAYING` before their framerate is known.

### New Buffer Timeout
A Source's production of new buffers can be monitored for timeout by adding a [New Buffer Timeout Pad Probe Handler (PPH)](/docs/api-pph.md#new-buffer-timeout-pad-probe-handler) to the Source's src-pad -- as shown in the image below -- by calling [`dsl_source_pph_add`](#dsl_source_pph_add). The handler will call the client provided callback function on timeout. 

<img src="/Images/new-buffer-timeout.png" width="600" />

**Important** The [RTSP Source](#dsl_source_rtsp_new) implements its own [new-buffer-timeout and reconnection management](/docs/overview.md#rtsp-stream-connection-management) that supersedes the need for a New Buffer Timeout PPH. 

## Audio Sources
... currently under design.

## Video Sources
There are eleven Video Source components supported, three of which are [Image Video Sources](image-video-sources):

* [App Source](#dsl_source_app_new) - Allows the application to insert raw samples or buffers into a DSL Pipeline.
* [CSI Source](#dsl_source_csi_new) - Camera Serial Interface (CSI) Source - Jetson platform only.
* [USB Source](#dsl_source_usb_new) - Universal Serial Bus (USB) Source.
* [URI Source](#dsl_source_uri_new) - Uniform Resource Identifier ( URI ) Source.
* [File Source](#dsl_source_file_new) - Derived from URI Source with fixed inputs.
* [RTSP Source](#dsl_source_rtsp_new) - Real-time Streaming Protocol ( RTSP ) Source - supports transport over TCP or UDP in unicast or multicast mode
* [Interpipe Source](#dsl_source_interpipe_new) - Receives pipeline buffers and events from an [Interpipe Sink](/docs/api-sink.md#dsl_sink_interpipe_new). Disabled by default, requires additional [install/build steps](/docs/installing-dependencies.md).
* [Single Image Source](#dsl_source_image_single_new) - Single frame to EOS.
* [Multi Image Source](#dsl_source_image_multi_new) - Streamed at one image file per frame.
* [Streaming Image Source](#dsl_source_image_stream_new)  - Single image streamed at a given frame rate. Disabled by default, requires additional [install/build steps](/docs/installing-dependencies.md).
* [Duplicate Source](#dsl_source_duplicate_new) - Used to duplicate another Video Source so the stream can be processed differently and in parallel with the original.

All Video Sources are derived from the base "Source" class (as show in the hierarchy below), therefore all [source methods](#source-methods) can be called with any Video Source.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── `video source`

### Video Buffer Conversion
All Video Sources include a [Video Converter](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvvideoconvert.html) providing programmatic control over the **formatting**, **scaling**, **cropping**, and **orienting** of the Source's output-buffers.

#### buffer-out-format
All Video Source's set their buffer-out-format to `DSL_VIDEO_FORMAT_NV12` by default. The format can be set to any one of the [DSL Video Format Types](#dsl-video-format-types) by calling [`dsl_source_video_buffer_out_format_set`](#dsl_source_video_buffer_out_format_set) when the Source is not PLAYING. The current setting can be read at any time by calling [`dsl_source_video_buffer_out_format_get`](#dsl_source_video_buffer_out_format_get). 

**Note:** NVIDIA's nvstreammux plugin, which is linked to each source in a Pipeline, limits the format types that can be used to `"I420"`, `"NV12"`, and `"RGBA"`.  (`"I420"` is identical to `"IYUV"`).

**Important!** A Video Source will automatically set/fix its buffer-out-format to `DSL_VIDEO_FORMAT_RGBA` if a [Dewarper](#video-dewarping) component is added. 
 
#### buffer-out-dimensions 
The output dimensions (width and height) can be scaled by calling [`dsl_source_video_buffer_out_dimensions_set`](#dsl_source_video_buffer_out_dimensions_set) when the Source is not PLAYING. The default values are set to 0, i.e. "no scaling". The current values can be read at any time by calling [`dsl_source_video_buffer_out_dimensions_get`](#dsl_source_video_buffer_out_dimensions_get).

#### buffer-out-frame-rate
The output frame-rate can be scaled up or down by calling [`dsl_source_video_buffer_out_frame_rate_set`](#dsl_source_video_buffer_out_frame_rate_set) when the Source is not PLAYING. The default values are set to 0, i.e. "no scaling". The current values can be read at any time by calling [`dsl_source_video_buffer_out_frame_rate_get`](#dsl_source_video_buffer_out_frame_rate_get). 

#### buffer-out-crop-rectangles
Each buffer can be cropped in two different ways by calling [`dsl_source_video_buffer_out_crop_rectangle_set`](#dsl_source_video_buffer_out_crop_rectangle_set) when the source is not PLAYING. The method of cropping is specified by the `crop_at` parameter to one of the [crop constant values](#video-source-buffer-out-crop-constants):
* `DSL_VIDEO_CROP_AT_SRC ` = left, top, width, and height of the input image which will be cropped and transformed into the output buffer.  
* `DSL_VIDEO_CROP_AT_DEST` = left, top, width, and height as the location in the output buffer where the input image will be transformed to.

<img src="/Images/video-crop-at-types.png" width="600" />

The current values can read at any time by calling [`dsl_source_video_buffer_out_crop_rectangle_get`](#dsl_source_video_buffer_out_crop_rectangle_get).

#### buffer-out-orientation
There are seven different [video orientation constants](#dsl-video-source-buffer-out-orientation-constants) that can be used to rotate or flip a Video Source's output by calling [`dsl_source_video_buffer_out_orientation_set`](#dsl_source_video_buffer_out_orientation_set) when the Source is not PLAYING. The default setting is `DSL_VIDEO_ORIENTATION_NONE`. The current setting can be read by calling [`dsl_source_video_buffer_out_orientation_get`](#dsl_source_video_buffer_out_orientation_get) at any time.

#### Video Sources and Demuxers
When using a [Demuxer](/docs/api-tiler.md), vs. a Tiler component, each demuxed source stream must have one or more downstream [Sink](/docs/api-sink) components to end the stream.

### Video Dewarping
A [Video Dewarper](/docs/api-dewarper.md), capable of 360 degree and perspective dewarping, can be added to a Video Source by calling [`dsl_source_video_dewarper_add`](#dsl_source_video_dewarper_add) and removed with [`dsl_source_video_dewarper_remove`](#dsl_source_video_dewarper_remove).

### Image Video Sources
Image Video Sources are used to decode JPEG image files into `video/x-raw' buffers. PNG files will be supported in a future release. Derived from the "Video Source" class, Image Video Sources can be called with any [Video Source Method](#video-source-methods)

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `image source`

## Source API
**Typedefs**
* [`dsl_rtsp_connection_data`](#dsl_rtsp_connection_data)

**Client Callback Typedefs**
* [`dsl_source_app_need_data_handler_cb`](#dsl_source_app_need_data_handler_cb)
* [`dsl_source_app_enough_data_handler_cb`](#dsl_source_app_enough_data_handler_cb)
* [`dsl_state_change_listener_cb`](#dsl_state_change_listener_cb)

**Constructors:**
* [`dsl_source_app_new`](#dsl_source_app_new)
* [`dsl_source_csi_new`](#dsl_source_csi_new)
* [`dsl_source_usb_new`](#dsl_source_usb_new)
* [`dsl_source_uri_new`](#dsl_source_uri_new)
* [`dsl_source_file_new`](#dsl_source_file_new)
* [`dsl_source_rtsp_new`](#dsl_source_rtsp_new)
* [`dsl_source_interpipe_new`](#dsl_source_interpipe_new)
* [`dsl_source_image_single_new`](#dsl_source_image_single_new)
* [`dsl_source_image_multi_new`](#dsl_source_image_multi_new)
* [`dsl_source_image_stream_new`](#dsl_source_image_stream_new)
* [`dsl_source_duplicate_new`](#dsl_source_duplicate_new)

**Source Methods:**
* [`dsl_source_unique_id_get`](#dsl_source_unique_id_get)
* [`dsl_source_stream_id_get`](#dsl_source_stream_id_get)
* [`dsl_source_name_get`](#dsl_source_name_get)
* [`dsl_source_media_type_get`](#dsl_source_media_type_get)
* [`dsl_source_framerate get`](#dsl_source_framerate_get)
* [`dsl_source_is_live`](#dsl_source_is_live)
* [`dsl_source_pause`](#dsl_source_pause)
* [`dsl_source_resume`](#dsl_source_resume)
* [`dsl_source_pph_add`](#dsl_source_pph_add)
* [`dsl_source_pph_remove`](#dsl_source_pph_remove)

**Video Source Methods:**
* [`dsl_source_video_dimensions_get`](#dsl_source_video_dimensions_get)
* [`dsl_source_video_buffer_out_format_get`](#dsl_source_video_buffer_out_format_get)
* [`dsl_source_video_buffer_out_format_set`](#dsl_source_video_buffer_out_format_set)
* [`dsl_source_video_buffer_out_dimensions_get`](#dsl_source_video_buffer_out_dimensions_get)
* [`dsl_source_video_buffer_out_dimensions_set`](#dsl_source_video_buffer_out_dimensions_set)
* [`dsl_source_video_buffer_out_frame_rate_get`](#dsl_source_video_buffer_out_frame_rate_get)
* [`dsl_source_video_buffer_out_frame_rate_set`](#dsl_source_video_buffer_out_frame_rate_set)
* [`dsl_source_video_buffer_out_crop_rectangle_get`](#dsl_source_video_buffer_out_crop_rectangle_get)
* [`dsl_source_video_buffer_out_crop_rectangle_set`](#dsl_source_video_buffer_out_crop_rectangle_set)
* [`dsl_source_video_buffer_out_orientation_get`](#dsl_source_video_buffer_out_orientation_get)
* [`dsl_source_video_buffer_out_orientation_set`](#dsl_source_video_buffer_out_orientation_set)
* [`dsl_source_video_dewarper_add`](#dsl_source_video_dewarper_add)
* [`dsl_source_video_dewarper_remove`](#dsl_source_video_dewarper_remove)

**App Source Methods:**
* [`dsl_source_app_data_handlers_add`](#dsl_source_app_data_handlers_add)
* [`dsl_source_app_data_handlers_remove`](#dsl_source_app_data_handlers_remove)
* [`dsl_source_app_buffer_push`](#dsl_source_app_buffer_push)
* [`dsl_source_app_sample_push`](#dsl_source_app_sample_push)
* [`dsl_source_app_eos`](#dsl_source_app_eos)
* [`dsl_source_app_stream_format_get`](#dsl_source_app_stream_format_get)
* [`dsl_source_app_stream_format_set`](#dsl_source_app_stream_format_set)
* [`dsl_source_app_block_enabled_get`](#dsl_source_app_block_enabled_get)
* [`dsl_source_app_block_enabled_set`](#dsl_source_app_block_enabled_set)
* [`dsl_source_app_current_level_bytes_get`](#dsl_source_app_current_level_bytes_get)
* [`dsl_source_app_max_level_bytes_get`](#dsl_source_app_max_level_bytes_get)
* [`dsl_source_app_max_level_bytes_set`](#dsl_source_app_max_level_bytes_set)
* [`dsl_source_app_do_timestamp_get`](#dsl_source_app_do_timestamp_get)
* [`dsl_source_app_do_timestamp_set`](#dsl_source_app_do_timestamp_set)

**CSI Source Methods**
* [`dsl_source_csi_sensor_id_get`](#dsl_source_csi_sensor_id_get)
* [`dsl_source_csi_sensor_id_set`](#dsl_source_csi_sensor_id_set)

**USB Source Methods**
* [`dsl_source_usb_device_location_get`](#dsl_source_usb_device_location_get)
* [`dsl_source_usb_device_location_set`](#dsl_source_usb_device_location_set)

**URI Source Methods**
* [`dsl_source_uri_uri_get`](#dsl_source_uri_uri_get)
* [`dsl_source_uri_uri_set`](#dsl_source_uri_uri_set)

**File Source Methods**
* [`dsl_source_file_file_path_get`](#dsl_source_file_file_path_get)
* [`dsl_source_file_file_path_set`](#dsl_source_file_file_path_set)
* [`dsl_source_file_repeat_enabled_get`](#dsl_source_file_repeat_enabled_get)
* [`dsl_source_file_repeat_enabled_set`](#dsl_source_file_repeat_enabled_set)

**RTSP Source Methods**
* [`dsl_source_rtsp_uri_get`](#dsl_source_rtsp_uri_get)
* [`dsl_source_rtsp_uri_set`](#dsl_source_rtsp_uri_set)
* [`dsl_source_rtsp_timeout_get`](#dsl_source_rtsp_timeout_get)
* [`dsl_source_rtsp_timeout_set`](#dsl_source_rtsp_timeout_set)
* [`dsl_source_rtsp_reconnection_params_get`](#dsl_source_rtsp_reconnection_params_get)
* [`dsl_source_rtsp_reconnection_params_set`](#dsl_source_rtsp_reconnection_params_set)
* [`dsl_source_rtsp_connection_data_get`](#dsl_source_rtsp_connection_data_get)
* [`dsl_source_rtsp_connection_stats_clear`](#dsl_source_rtsp_connection_stats_clear)
* [`dsl_source_rtsp_latency_get`](#dsl_source_rtsp_latency_get)
* [`dsl_source_rtsp_latency_set`](#dsl_source_rtsp_latency_set)
* [`dsl_source_rtsp_drop_on_latency_enabled_get`](#dsl_source_rtsp_drop_on_latency_enabled_get)
* [`dsl_source_rtsp_drop_on_latency_enabled_set`](#dsl_source_rtsp_drop_on_latency_enabled_set)
* [`dsl_source_rtsp_tls_validation_flags_get`](#dsl_source_rtsp_tls_validation_flags_get)
* [`dsl_source_rtsp_tls_validation_flags_set`](#dsl_source_rtsp_tls_validation_flags_set)
* [`dsl_source_rtsp_state_change_listener_add`](#dsl_source_rtsp_state_change_listener_add)
* [`dsl_source_rtsp_state_change_listener_remove`](#dsl_source_rtsp_state_change_listener_remove)
* [`dsl_source_rtsp_tap_add`](#dsl_source_rtsp_tap_add)
* [`dsl_source_rtsp_tap_remove`](#dsl_source_rtsp_tap_remove)

**Interpipe Source Methods**
* [`dsl_source_interpipe_listen_to_get`](#dsl_source_interpipe_listen_to_get)
* [`dsl_source_interpipe_listen_to_set`](#dsl_source_interpipe_listen_to_set)
* [`dsl_source_interpipe_accept_settings_get`](#dsl_source_interpipe_accept_settings_get)
* [`dsl_source_interpipe_accept_settings_set`](#dsl_source_interpipe_accept_settings_set)

**Single Image Source Methods**
* [`dsl_source_image_file_path_get`](#dsl_source_image_file_path_get)
* [`dsl_source_image_file_path_set`](#dsl_source_image_file_path_set)

**Multi Image Source Methods**
* [`dsl_source_image_multi_loop_enabled_get`](#dsl_source_image_multi_loop_enabled_get)
* [`dsl_source_image_multi_loop_enabled_set`](#dsl_source_image_multi_loop_enabled_set)
* [`dsl_source_image_multi_indices_get`](#dsl_source_image_multi_indices_get)
* [`dsl_source_image_multi_indices_set`](#dsl_source_image_multi_indices_set)

**Image Stream Methods**
* [`dsl_source_image_stream_timeout_get`](#dsl_source_image_stream_timeout_get)
* [`dsl_source_image_stream_timeout_set`](#dsl_source_image_stream_timeout_get)

**Duplicate Source Methods**
* [`dsl_source_duplicate_original_get`](#dsl_source_duplicate_original_get)
* [`dsl_source_duplicate_original_set`](#dsl_source_duplicate_original_set)

## Return Values
Streaming Source Methods use the following return codes, in addition to the general [Component API Return Values](/docs/api-component.md).
```C
#define DSL_RESULT_SOURCE_RESULT                                    0x00020000
#define DSL_RESULT_SOURCE_NAME_NOT_UNIQUE                           0x00020001
#define DSL_RESULT_SOURCE_NAME_NOT_FOUND                            0x00020002
#define DSL_RESULT_SOURCE_NAME_BAD_FORMAT                           0x00020003
#define DSL_RESULT_SOURCE_NOT_FOUND                                 0x00020004
#define DSL_RESULT_SOURCE_THREW_EXCEPTION                           0x00020005
#define DSL_RESULT_SOURCE_FILE_NOT_FOUND                            0x00020006
#define DSL_RESULT_SOURCE_NOT_IN_USE                                0x00020007
#define DSL_RESULT_SOURCE_NOT_IN_PLAY                               0x00020008
#define DSL_RESULT_SOURCE_NOT_IN_PAUSE                              0x00020009
#define DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE                    0x0002000A
#define DSL_RESULT_SOURCE_CODEC_PARSER_INVALID                      0x0002000B
#define DSL_RESULT_SOURCE_CODEC_PARSER_INVALID                      0x0002000B
#define DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED                       0x0002000C
#define DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED                    0x0002000D
#define DSL_RESULT_SOURCE_TAP_ADD_FAILED                            

#define DSL_RESULT_SOURCE_TAP_REMOVE_FAILED                         0x0002000F
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE                   0x00020010
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_DECODE_SOURCE            0x00020011
#define DSL_RESULT_SOURCE_COMPONENT_IS_NOT_FILE_SOURCE              0x00020012
#define DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED                       0x00020013
#define DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED                    0x00020014
#define DSL_RESULT_SOURCE_SET_FAILED                                0x00020015
#define DSL_RESULT_SOURCE_CSI_NOT_SUPPORTED                         0x00020016
#define DSL_RESULT_SOURCE_HANDLER_ADD_FAILED                        0x00020017
#define DSL_RESULT_SOURCE_HANDLER_REMOVE_FAILED                     0x00020018

```

## DSL State Values
```C
#define DSL_STATE_NULL                                              1
#define DSL_STATE_READY                                             2
#define DSL_STATE_PAUSED                                            3
#define DSL_STATE_PLAYING                                           4
```

## DSL Source Media Types
```C
#define DSL_MEDIA_TYPE_VIDEO_XRAW                                   L"video/x-raw"
#define DSL_MEDIA_TYPE_AUDIO_XRAW                                   L"audio/x-raw"
```

## DSL Video Format Types
```C
#define DSL_VIDEO_FORMAT_I420                                       L"I420"
#define DSL_VIDEO_FORMAT_NV12                                       L"NV12"
#define DSL_VIDEO_FORMAT_RGBA                                       L"RGBA"   
#define DSL_VIDEO_FORMAT_DEFAULT                                    DSL_VIDEO_FORMAT_I420
```

## DSL Stream format Types
```C
#define DSL_STREAM_FORMAT_BYTE                                      2
#define DSL_STREAM_FORMAT_TIME                                      3
```

## NVIDIA Buffer Memory Types
```C
#define DSL_NVBUF_MEM_TYPE_DEFAULT                                  0
#define DSL_NVBUF_MEM_TYPE_PINNED                                   1
#define DSL_NVBUF_MEM_TYPE_DEVICE                                   2
#define DSL_NVBUF_MEM_TYPE_UNIFIED                                  3
```

## RTP Protocols
```C
#define DSL_RTP_TCP                                                 0x04
#define DSL_RTP_ALL                                                 0x07
```

## TLS certificate validation flags
```c
#define DSL_TLS_CERTIFICATE_UNKNOWN_CA                              0x00000001
#define DSL_TLS_CERTIFICATE_BAD_IDENTITY                            0x00000002
#define DSL_TLS_CERTIFICATE_NOT_ACTIVATED                           0x00000004
#define DSL_TLS_CERTIFICATE_EXPIRED                                 0x00000008
#define DSL_TLS_CERTIFICATE_REVOKED                                 0x00000010
#define DSL_TLS_CERTIFICATE_INSECURE                                0x00000020
#define DSL_TLS_CERTIFICATE_GENERIC_ERROR                           0x00000040
#define DSL_TLS_CERTIFICATE_VALIDATE_ALL                            0x0000007f
```

<br>

## Video Source buffer-out-crop Constants
Constants to define how to crop the output buffer for a given Source Component. The constants map to the nvvideoconvert element's `src-crop` and `dest-crop` properties. See the [DeepStream docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvvideoconvert.html#gst-nvvideoconvert) for more information.
```C
#define DSL_VIDEO_CROP_AT_SRC                                       0
#define DSL_VIDEO_CROP_AT_DEST                                      1
```

<br>

## DSL Video Source buffer-out-orientation Constants
```C
#define DSL_VIDEO_ORIENTATION_NONE                                  0       
#define DSL_VIDEO_ORIENTATION_ROTATE_COUNTER_CLOCKWISE_90           1
#define DSL_VIDEO_ORIENTATION_ROTATE_180                            2
#define DSL_VIDEO_ORIENTATION_ROTATE_CLOCKWISE_90                   3
#define DSL_VIDEO_ORIENTATION_FLIP_HORIZONTALLY                     4
#define DSL_VIDEO_ORIENTATION_FLIP_UPPER_RIGHT_TO_LOWER_LEFT        5
#define DSL_VIDEO_ORIENTATION_FLIP_VERTICALLY                       6
#define DSL_VIDEO_ORIENTATION_FLIP_UPPER_LEFT_TO_LOWER_RIGHT        7
```

---

## Types
### dsl_rtsp_connection_data
This DSL Type defines a structure of "connection stats" and "parameters" for a given RTSP Source. The data is queried by calling [dsl_source_rtsp_connection_data_get](#dsl_source_rtsp_connection_data_get).

```C
typedef struct dsl_rtsp_connection_data
{
    boolean is_connected;
    time_t first_connected;
    time_t last_connected;
    time_t last_disconnected;
    uint count;
    boolean is_in_reconnect;
    uint retries;
    uint sleep;
    uint timeout;
}dsl_rtsp_connection_data;
```

**Fields**
* `is_connected` true if the RTSP Source is currently in a connected state, false otherwise
* `first_connected` - epoch time in seconds for the first successful connection, or when the stats were last cleared
* `last_connected`- epoch time in seconds for the last successful connection, or when the stats were last cleared
* `last_disconnected` - epoch time in seconds for the last disconnection, or when the stats were last cleared
* `count` - the number of successful connections from the start of Pipeline play, or from when the stats were last cleared
* `is_in_reconnect` - true if the RTSP Source is currently in a reconnection cycle, false otherwise.
* `retries` - number of re-connection retries for either the current cycle, if `is_in_reconnect` is true, or the last connection if `is_in_reconnect` is false`.
* `sleep` - current setting for the time to sleep between reconnection attempts after failure.
* `is_connect` - true if the RTSP Source is currently in a connected state, false otherwise.
* `timeout` - current setting for the maximum time to wait for an asynchronous state change to complete before resetting the source and then retrying again after the next sleep period.

**Python Example**
```Python
retval, data = dsl_source_rtsp_connection_data_get('rtsp-source')

print('Connection data for source:', 'rtsp-source')
print('  is connected:     ', data.is_connected)
print('  first connected:  ', time.ctime(data.first_connected))
print('  last connected:   ', time.ctime(data.last_connected))
print('  last disconnected:', time.ctime(data.last_disconnected))
print('  total count:      ', data.count)
print('  in is reconnect:  ', data.is_in_reconnect)
print('  retries:          ', data.retries)
print('  sleep time:       ', data.sleep,'seconds')
print('  timeout:          ', data.timeout, 'seconds')
```

<br>

## Client CallBack Typedefs
### *dsl_source_app_need_data_handler_cb*
```C++
typedef void (*dsl_source_app_need_data_handler_cb)(uint length, void* client_data);
```
Callback typedef for the App Source Component. The function is registered with the App Source by calling [dsl_source_app_data_handlers_add](#dsl_source_app_data_handlers_add). Once the Pipeline is playing, the function will be called when the Source needs new data to process.

**Parameters**
* `length` - [in] the amount of bytes needed.  The length is just a hint and when it is set to -1, any number of bytes can be pushed into the App Source.
* `client_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add.

<br>

### *dsl_source_app_enough_data_handler_cb*
```C++
typedef void (*dsl_source_app_enough_data_handler_cb)(void* client_data);
```
Callback typedef for the App Source Component. The function is registered with the App Source by calling [dsl_source_app_data_handlers_add](#dsl_source_app_data_handlers_add). Once the Pipeline is playing, the function will be called when the Source has enough data to process. It is recommended that the application stops calling [dsl_source_app_buffer_push](#dsl_source_app_buffer_push) until [dsl_source_app_need_data_handler_cb](#dsl_source_app_need_data_handler_cb) is called again.

**Parameters**
* `client_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add.

<br>

### *dsl_state_change_listener_cb*
```C++
typedef void (*dsl_state_change_listener_cb)(uint old_state, uint new_state, void* client_data);
```
Callback typedef for a client state-change listener. Functions of this type are added to an RTSP Source by calling [dsl_source_rtsp_state_change_listener_add](#dsl_source_rtsp_state_change_listener_add). Once added, the function will be called on every change of the Source's state until the client removes the listener by calling [dsl_source_rtsp_state_change_listener_remove](#dsl_source_rtsp_state_change_listener_remove).

**Parameters**
* `old_state` - [in] one of [DSL State Values](#dsl-state-values) constants for the old (previous) pipeline state.
* `new_state` - [in] one of [DSL State Values](#dsl-state-values) constants for the new pipeline state.
* `client_data` - [in] opaque pointer to client's user data, passed into the pipeline on callback add.

<br>

## Constructors

### *dsl_source_app_new*
```C
DslReturnType dsl_source_app_new(const wchar_t* name, boolean is_live, 
    const wchar_t* buffer_in_format, uint width, uint height, uint fps_n, uint fps_d);
```
Creates a new, uniquely named App Source component to insert data -- buffers or samples -- into a DSL Pipeline.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `app source`

**Parameters**
* `source` - [in] unique name for the new Source
* `is_live` - [in] set to true to instruct the source to behave like a live source. This includes that it will only push out buffers in the PLAYING state.
* `buffer_in_format` - [in]  one of the [DSL_BUFFER_FORMAT](#dsl-video-format-types) constants.
* `width` - [in] width of the source in pixels
* `height` - [in] height of the source in pixels
* `fps-n` - [in] frames per second fraction numerator
* `fps-d` - [in] frames per second fraction denominator

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_new('my-app-source', True,
    DSL_BUFFER_FORMAT_I420, 1280, 720, 30, 1)
```

<br>

### *dsl_source_csi_new*
```C
DslReturnType dsl_source_csi_new(const wchar_t* source,
    uint width, uint height, uint fps_n, uint fps_d);
```
Creates a new, uniquely named CSI Camera Source component.

**Important:** A unique sensor-id is assigned to each CSI Source on creation, starting with 0. The default setting can be overridden by calling [dsl_source_decode_uri_set](#dsl_source_decode_uri_set). The call will fail if the given sensor-id is not unique. If a source is deleted, the sensor-id will be re-assigned to a new CSI Source if one is created.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `csi source`

**Parameters**
* `name` - [in] unique name for the new Source
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
```C
DslReturnType dsl_source_usb_new(const wchar_t* name,
    uint width, uint height, uint fps_n, uint fps_d);
```
Creates a new, uniquely named USB Camera Source component.

**Important:** A unique device-location is assigned to each USB Source on creation, starting with `/dev/video0`, followed by `/dev/video1`, and so on. The default assignment can be overridden by calling [dsl_source_usb_device_location_set](#dsl_source_usb_device_location_set). The call will fail if the given device-location is not unique. If a source is deleted, the device-location will be re-assigned to a new USB Source if one is created.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `usb source`

**Parameters**
* `name` - [in] unique name for the new Source
* `width` - [in] width of the source in pixels
* `height` - [in] height of the source in pixels
* `fps-n` - [in] frames per second fraction numerator
* `fps-d` - [in] frames per second fraction denominator

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_usb_new('my-usb-source', 1280, 720, 30, 1)
```
<br>

### *dsl_source_uri_new*
```C
DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, 
    boolean is_live, uint skip_frames, uint drop_frame_interval);
```
This service creates a new, uniquely named URI Source component.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `uri source`

**Parameters**
* `name` - [in] unique name for the new Source
* `uri` - [in] fully qualified URI prefixed with `http://`, `https://`,  or `file://`
* `is_live` [in] `true` if the URI is a live source, `false` otherwise. File URI's will use a fixed value of `false`
* `skip_frames` - [in] the type of frames to skip during decoding.
  -   (0): decode_all       - Decode all frames
  -   (1): decode_non_ref   - Decode non-ref frame
  -   (2): decode_key       - decode key frames
* `drop_frame_interval` [in] number of frames to drop between each decoded frame. 0 = decode all frames

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_uri_new('my-uri-source', '../../test/streams/sample_1080p_h264.mp4',
    False, 0)
```

<br>

### *dsl_source_file_new*
```C
DslReturnType dsl_source_file_new(const wchar_t* name,
    const wchar_t* file_path, boolean repeat_enabled);
```
This service creates a new, uniquely named File Source component. The Source implements a URI Source with the following set parameters.
* `is_live = false`
* `intra_decode = false`
* `drop_frame_interval = 0`

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── [`uri source`](dsl_source_uri_new)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `file source`

**Parameters**
* `name` - [in] unique name for the new Source
* `file_path` - [in] absolute or relative path to the file to play
* `repeat_enabled` [in] set to `true` to repeat the file on end-of-stream (EOS), `false` otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_file_new('my-uri-source', './streams/sample_1080p_h264.mp4', false)
```

<br>

### *dsl_source_rtsp_new*
```C
DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri, uint protocol,
    uint skip_frames, uint drop_frame_interval, uint latency, uint timeout);
```

This service creates a new, uniquely named RTSP Source component. The RTSP Source supports transport over TCP or UDP in unicast or multicast mode. By default, the RTSP Source will negotiate a connection in the following order: UDP unicast/UDP multicast/TCP. The order cannot be changed but the allowed protocols can be controlled with the `protocol` parameter.

**Note** The RTSP Source acts like a live source and will therefore only generate data in the `PLAYING` state.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `rtsp source`

**Parameters**
* `name` - [in] unique name for the new Source
* `uri` - [in] fully qualified URI prefixed with `rtsp://`
* `protocol` - [in] one of the [RTP Protocols](#rtp-protocols) constant values.
* `skip_frames` - [in] the type of frames to skip during decoding.
  -   (0): decode_all       - Decode all frames
  -   (1): decode_non_ref   - Decode non-ref frame
  -   (2): decode_key       - decode key frames
* `latency` - [in] source latency setting in milliseconds, equates to the amount of data to buffer. 
* `timeout` - [in] maximum time between successive frame buffers in units of seconds before initiating a "reconnection-cycle". Set to 0 to disable the timeout.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_rtsp_new('dsl_source_uri_new',
    'rtsp://username:password@192.168.0.17:554/rtsp-camera-1', True, 1000, 2)
```

<br>

### *dsl_source_interpipe_new*
```C
DslReturnType dsl_source_interpipe_new(const wchar_t* name,
    const wchar_t* listen_to, boolean is_live,
    boolean accept_eos, boolean accept_events);
```
This service creates a new, uniquely named Interpipe Source component to listen to an Interpipe Sink Component. The Sink to `listen_to` can be updated dynamically while in a playing state. 

**Important!** The Interpipe Services are disabled by default and require additional [install/build steps](/docs/installing-dependencies.md).

Refer to the [Interpipe Services](/docs/overview.md#interpipe-services) overview for more information.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `interpipe source`

**Parameters**
* `name` - [in] unique name for the new Source
* `listen_to` - [in] unique name of the Interpipe Sink to listen to.
* `is_live` - [in] set to true to act as live source, false otherwise.
* `accept_eos` - [in] set to true to accept EOS events from the Interpipe Sink, false otherwise.
* `accept_events` - [in] set to true to accept events (except EOS event) from the Inter-Pipe Sink, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_interpipe_new('my-interpipe-source', 'my-interpipe-sink',
    false, true, true)
```

<br>

### *dsl_source_image_single_new*
```C
DslReturnType dsl_source_image_single_new(const wchar_t* name,
    const wchar_t* file_path);
```
This service creates a new, uniquely named Single-Image Source component. The Image is streamed as a single frame followed by an End of Stream (EOS) event.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── [`image source`](#image-source-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `single-image source`

**Parameters**
* `name` - [in] unique name for the new Source
* `file_path` - [in] absolute or relative path to the image file to play
*
**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_image_single_new('my-image-source', './streams/image4.jpg')
```
<br>

### *dsl_source_image_multi_new*
```C
DslReturnType dsl_source_image_multi_new(const wchar_t* name,
    const wchar_t* file_path, uint fps_n, uint fps_d);
```
This service creates a new, uniquely named Multi Image Source component that decodes multiple images specified by a folder/filename-pattern using the printf style %d.

Example: `./my_images/image.%d04.mjpg`, where the files in "./my_images/" are named `image.0000.mjpg`, `image.0001.mjpg`, `image.0002.mjpg` etc.

The images are streamed one per frame at the specified framerate. A final EOS event occurs once all images have been played.

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── [`image source`](#image-source-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `multi-image source`

**Parameters**
* `name` - [in] unique name for the new Source.
* `file_path` - [in] absolute or relative path to the image files to play specified with the printf style %d.
* `fps-n` - [in] frames per second fraction numerator.
* `fps-d` - [in] frames per second fraction denominator.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_image_multi_new('my-multi-image-source', './my_images/image.%d04.mjpg')
```

<br>

### *dsl_source_image_stream_new*
```C
DslReturnType dsl_source_image_stream_new(const wchar_t* name,
    const wchar_t* file_path, boolean is_live, uint fps_n, uint fps_d, uint timeout);
```
This service creates a new, uniquely named Streaming Image Source component. The Image is overlaid on top of a mock video stream that plays at a specified frame rate. The video source can mock both live and non-live sources allowing the Image to be batched along with other Source components.

**Important!** The Streaming-Image Services are disabled by default and require additional [install/build steps](/docs/installing-dependencies.md).

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── [`image source`](#image-source-methods)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `streaming-image source`

**Parameters**
* `name` - [in] unique name for the new Source
* `file_path` - [in] absolute or relative path to the image file to play
* `is_live` [in] true if the Source is to act as a live source, false otherwise.
* `fps-n` - [in] frames per second fraction numerator
* `fps-d` - [in] frames per second fraction denominator
* `timeout` [in] time to stream the image before generating an end-of-stream (EOS) event, in units of seconds. Set to 0 for no timeout.

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_image_stream_new('my-image-stream-source', './streams/image4.jpg',
    false, 30, 1, 0)
```

<br>

### *dsl_source_duplicate_new*
```C
DslReturnType dsl_source_duplicate_new(const wchar_t* name, const wchar_t* original);
```
This service creates a new, uniquely named Duplicate Source used to duplicate the stream of another named Video Source. Both the Duplicate Source and the Original Source must be added to the same Pipeline. The Duplicate Source will be Tee'd into the Original Source prior to the Original Source's output-buffer video-converter, video-rate-controller,  and caps-filter (output buffer control plugins built into every Video Source). The Duplicate Source, as a Video Source, will have its own output buffer control plugins meaning both sources will have independent control over their buffer-out formatting, dimensions, frame-rate, cropping, and orientation.

The relationship between Duplicate Sources and Original Sources is many to one, i.e. multiple Duplicates Sources can duplicate the same Original Source and a Duplicate Source can be an Original Source for one or more other Duplicate Sources.

**IMPORTANT!** The Original Source must exist prior to calling the Duplicate Source constructor. 

#### Hierarchy
[`component`](/docs/api-component.md)<br>
&emsp;╰── [`source`](#source-methods)<br>
&emsp;&emsp;&emsp;&emsp;╰── [`video source`](#video-sources)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;╰── `duplicate source`

**Parameters**
* `source` - [in] unique name for the new Duplicate Source.
* `original` - [in] unique name of the Original Source to duplicate. 

**Returns**
* `DSL_RESULT_SUCCESS` on successful creation. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_duplicate_new('my-duplicate-source', 'my-rtsp-source')
```
<br>

---

## Destructors
As with all Pipeline components, Sources are deleted by calling [dsl_component_delete](/docs/api-component.md#dsl_component_delete), [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many), or [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)

---

## Source Methods

### *dsl_source_unique_id_get*
```C
DslReturnType dsl_source_unique_id_get(const wchar_t* name, int* unique_id);
```
This service gets the unique-id assigned to the Source component once added to a Pipeline. The unique source-id will be derived from the 
```
unique-id = (unique pipeline-id << DSL_PIPELINE_SOURCE_UNIQUE_ID_OFFSET_IN_BITS) | unique stream-id
```

**Parameters**
* `name` - [in] unique name of the Source to query.
* `unique_id` - [out] unique source id as assigned by the Pipeline. The unique id will be set to -1 when unassigned (i.e. not added to a Pipeline).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, unique_id = dsl_source_unique_id_get('my-source')
```

<br>

### *dsl_source_stream_id_get*
```C
DslReturnType dsl_source_stream_id_get(const wchar_t* name, int* stream_id);
```
This service get the stream-id assigned to the Source component once added to a Pipeline. The 0-based stream-id is assigned to each Source by the Pipeline according to the order they are added. The Source will be connected to a Streammuxer sink-pad with the same pad-id as the stream-id.

IMPORTANT: If a source is dynamically removed (while the Pipeline is playing) and a new Source is added, the stream-id (and Streammuxer sink-pad) will be reused.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `stream_id` - [out] unique stream-id as assigned by the Pipeline. The stream id will be set to -1 when unassigned (i.e. not added to a Pipeline).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, stream_id = dsl_source_stream_id_get('my-source')
```

<br>

### *dsl_source_name_get*
```C
DslReturnType dsl_source_name_get(uint unique_id, const wchar_t** name);
```
This service gets the name of a Source component from a unique source-id.

**Parameters**
* `unique_id` - [in] unique source-id to check for. Must be a valid assigned source-id and not -1.
* `name` - [out] unique name of the Source component if found.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
# get the name of the source from pipeline-id=1 with stream-id=2
retval, name = dsl_source_name_get(0x00010002)
```

<br>

### *dsl_source_media_type_get*
```C
DslReturnType dsl_source_media_type_get(const wchar_t* name,
    const wchar_t** media_type);
```
This service gets the media type for the named Source component. The media-type will be specific to the base source type as follows:
* Video Sources will return `"video/x-raw"`
* Audio Sources will return `"audio/x-raw"`
* Audio/Video Source will return `"video/x-raw;audio/x-raw"`

**Note:** DSL currently implements Video only. Audio is to be supported in a future release.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `media-type` - [out] one of the [DSL_MEDIA_TYPE constants](#dsl-source-media-types).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, media_type = dsl_source_media_type_get('my-source')
```

<br>

### *dsl_source_framerate_get*
```C
DslReturnType dsl_source_frame_rate_get(const wchar_t* name, uint* fps_n, uint* fps_n);
```
This service returns the fractional frames per second as numerator and denominator for a named source. **Note:** Some Sources need to transition to a state of PLAYING before their framerate is known.

**Parameters**
* `name` - [in] unique name of the Source to play.
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
```C
DslReturnType dsl_source_is_live(const wchar_t* name, boolean* is_live);
```
Returns `true` if the Source component's stream is live. CSI, USB, and RTSP Camera sources will always return `True`.

**Parameters**
* `name` - [in] unique name of the Source to query
* `is_live` - [out] `true` if the source is live, false otherwise

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, is_live = dsl_source_is_live('my-uri-source')
```

<br>

### *dsl_source_pause*
```C
DslReturnType dsl_source_pause(const wchar_t* name);
```
This method tries to change the state of a Source component from `DSL_STATE_PLAYING` to `DSL_STATE_PAUSED`.  This service will fail if the Source is not currently in a state of `DSL_STATE_PLAYING`. 

**Parameters**
* `name` - unique name of the Source to pause

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_play('my-uri-source')
```

<br>

### *dsl_source_resume*
```C
DslReturnType dsl_source_resume(const wchar_t* name);
```
This method tries to change the state of a Source component from `DSL_STATE_PAUSED` to `DSL_STATE_PLAYING`. This service will fail if the Source is not currently in a state OF `DSL_STATE_PAUSED`. 

<br>

**Parameters**
* `name` - unique name of the Source to resume.

**Returns**
* `DSL_RESULT_SUCCESS` on successful transition. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_resume('my-source')
```

<br>

### *dsl_source_pph_add*
```C++
DslReturnType dsl_source_pph_add(const wchar_t* name, const wchar_t* handler);
```

This service adds a [Pad Probe Handler](/docs/api-pph.md) -- typically a [New Buffer Timeout PPH](/docs/api-pph.md#dsl_pph_buffer_timeout_new) --- to the src-pad of the named Source Component. 

**Important Note** Adding an [Object Detection Event PPH](/docs/api-pph.md#dsl_pph_ode_new) or an [Non-Maximum Processor PPH](/docs/api-pph.md#dsl_pph_nmp_new) will result in a NOP as there is no batch-metadata attached to the buffers for these PPHs to process. The initial frame level batch-metadata is added to the buffers by the Pipelines's Stream-muxer downstream of the Source. 

**Parameters**
* `name` - [in] unique name of the Source Component to update.
* `handler` - [in] unique name of Pad Probe Handler to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**

```Python
retval = dsl_source_pph_add('my-csi-source-1', 'my-buffer-timeout-pph-1')
```

<br>

### *dsl_source_pph_remove*
```C++
DslReturnType dsl_source_pph_remove(const wchar_t* name, const wchar_t* handler);
```
This service removes a [Pad Probe Handler](/docs/api-pph.md) from the src-pad of the named Source Component. The service will fail if the named handler is not owned by the named source.

**Parameters**
* `name` - [in] unique name of the Source Component to update.
* `handler` - [in] unique name of Pad Probe Handler to remove

**Returns**
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_pph_remove('my-csi-source-1', 'my-buffer-timeout-pph-1')
```

---

## Video Source Methods

### *dsl_source_video_dimensions_get*
```C
DslReturnType dsl_source_video_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service gets the dimensions for a named Video Source component if known. 

**Parameters**
* `source` - [in] unique name of the Source to query.
* `width` - [out] width of the Source in pixels. 0 if unknown.
* `height` - [out] height of the Source in pixels. 0 if unknown.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_source_dimensions_get('my-uri-source')
```

<br>

### *dsl_source_video_buffer_out_format_get*
```C
DslReturnType dsl_source_video_buffer_out_format_get(const wchar_t* name,
    const wchar_t** format);
```
This service gets the current buffer-out-format for the named Video Source component.

**Parameters**
* `source` - [in] unique name of the Source to query.
* `format` - [out] current buffer-out-format. One of the [DSL_VIDEO_FORMAT](#dsl-video-format-types) constant string values. Default = `DSL_VIDEO_FORMAT_DEFAULT`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, format = dsl_source_video_buffer_out_format_get('my-uri-source')
```

<br>

### *dsl_source_video_buffer_out_format_set*
```C
DslReturnType dsl_source_video_buffer_out_format_set(const wchar_t* name,
    const wchar_t* format);
```
This service sets the buffer-out-format for the named Video Source component to use.

**Parameters**
* `source` - [in] unique name of the Source to update.
* `format` - [in] new buffer-out-format. One of the [DSL_VIDEO_FORMAT](#dsl-video-format-types) constant string values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_video_buffer_out_format_set('my-uri-source',
    DSL_VIDEO_FORMAT_RGBA)
```

<br>

### *dsl_source_video_buffer_out_dimensions_get*
```C
DslReturnType dsl_source_video_buffer_out_dimensions_get(const wchar_t* name, 
    uint* width, uint* height);
```
This service gets the scaled buffer-out-dimensions of the named Source component. The default value for both width and height is 0, i.e. no-scaling.

**Parameters**
* `source` - [in] unique name of the Source to query.
* `width` - [out] scaled width of the output buffer in pixels. Default = 0.
* `height` - [out] scaled height of the output buffer in pixels. Default = 0.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, width, height = dsl_source_video_buffer_out_dimensions_get('my-uri-source')
```

<br>

### *dsl_source_video_buffer_out_dimensions_set*
```C
DslReturnType dsl_source_video_buffer_out_dimensions_set(const wchar_t* name, 
    uint width, uint height);
```
This service sets the buffer-out-format for the named Video Source component to use.

**Parameters**
* `source` - [in] unique name of the Source to update.
* `width` - [in] scaled width of the output buffer in pixels.
* `height` - [out] scaled height of the output buffer in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_video_buffer_out_dimensions_set('my-uri-source',
    1280, 720)
```

<br>

### *dsl_source_video_buffer_out_frame_rate_get*
```C
DslReturnType dsl_source_video_buffer_out_frame_rate_get(const wchar_t* name, 
    uint* fps_n, uint* fps_d);
```
This service gets the scaled frame-rate as a fraction for the named Video Source. The default values of 0 for fps_n and fps_d indicate no scaling..

**Parameters**
* `source` - [in] unique name of the Source to query.
* `fps_n` - [out] scaled frames per second numerator. Default = 0.
* `fps_d` - [out] scaled frames per second denominator. Default = 0.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, fps_n, fps_d = dsl_source_video_buffer_out_frame_rate_get('my-uri-source')
```

<br>

### *dsl_source_video_buffer_out_frame_rate_set*
```C
DslReturnType dsl_source_video_buffer_out_frame_rate_set(const wchar_t* name, 
    uint fps_n, uint fps_d);
```
This service sets the scaled frame-rate as a fraction for the named Video Source to use.

**Parameters**
* `source` - [in] unique name of the Source to update.
* `width` - [in] scaled width of the output buffer in pixels.
* `height` - [out] scaled height of the output buffer in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_video_buffer_out_frame_rate_set('my-uri-source', 2, 1)
```

<br>

### *dsl_source_video_buffer_out_crop_rectangle_get*
```C
DslReturnType dsl_source_video_buffer_out_crop_rectangle_get(const wchar_t* name,
    uint when, uint* left, uint* top, uint* width, uint* height);
```
This service gets a buffer-out crop-rectangle for the named Source component. See [buffer-out-crop-rectangles](#buffer-out-crop-rectangles) for an explanation of the `crop_at` parameter. The default is "no-crop" with left, top, width, and height all 0.

**Parameters**
* `source` - [in] unique name of the Source to query.
* `crop_at` - [in] specifies which of the crop rectangles to query, either `DSL_VIDEO_CROP_AT_SRC` or `DSL_VIDEO_CROP_AT_DEST`.
* `left` - [out] left positional coordinate of the rectangle in pixels. Default = 0.
* `top` - [out] top positional coordinate of the rectangle in pixels. Default = 0.
* `width` - [out] width of the rectangle in pixels. Default = 0.
* `height` - [out] height of the rectangle in pixels. Default = 0.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, crop_at, left, top, width, height = dsl_source_video_buffer_out_crop_rectangle_get(
    'my-uri-source')
```

<br>

### *dsl_source_video_buffer_out_crop_rectangle_set*
```C
DslReturnType dsl_source_video_buffer_out_crop_rectangle_set(const wchar_t* name,
    uint when, uint left, uint top, uint width, uint height);
```
This service sets one of the buffer-out crop-rectangles for the named Source component. See [buffer-out-crop-rectangles](#buffer-out-crop-rectangles) for an explanation of the `crop_at` parameter. The default is "no-crop" with left, top, width, and height all 0.

**Parameters**
* `source` - [in] unique name of the Source to update.
* `crop_at` - [in] specifies which of the crop rectangles to update, either `DSL_VIDEO_CROP_AT_SRC` or `DSL_VIDEO_CROP_AT_DEST`.
* `left` - [in] left positional coordinate of the rectangle in pixels.
* `top` - [in] top positional coordinate of the rectangle in pixels.
* `width` - [in] width of the rectangle in pixels.
* `height` - [in] height of the rectangle in pixels.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_video_buffer_out_crop_rectangle_set('my-uri-source',
    DSL_VIDEO_CROP_AT_SRC, 200, 200, 1280, 720)
```

<br>

### *dsl_source_video_buffer_out_orientation_get*
```C
DslReturnType dsl_source_video_buffer_out_orientation_get(const wchar_t* name,
    uint* orientation);
```
This service gets the current buffer-out-orientation for the named Video Source component.

**Parameters**
* `source` - [in] unique name of the Source to query.
* `orientation` - [out] current buffer-out-orientation. One of the [DSL_VIDEO_ORIENTATION](#dsl-video-source-buffer-out-orientation-constants) constant string values. Default = `DSL_VIDEO_ORIENTATION_NONE`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, orientation = dsl_source_video_buffer_out_orientation_get(
    'my-uri-source')
```

<br>

### *dsl_source_video_buffer_out_orientation_set*
```C
DslReturnType dsl_source_video_buffer_out_orientation_set(const wchar_t* name,
    uint orientation);
```
This service sets the buffer-out-orientation for the named Video Source component to use.

**Parameters**
* `source` - [in] unique name of the Source to update.
* `orientation` - [in] new buffer-out-orientation. One of the [DSL_VIDEO_ORIENTATION](#dsl-video-source-buffer-out-orientation-constants) constant string values.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_video_buffer_out_orientation_set('my-uri-source',
    DSL_VIDEO_ORIENTATION_FLIP_VERTICALLY)
```

<br>

### *dsl_source_video_dewarper_add*
```C
DslReturnType dsl_source_video_dewarper_add(const wchar_t* name, const wchar_t* dewarper);
```
This service adds a previously constructed [Dewarper](api-dewarper.md) component to a named Video Source component. A source can have at most one Dewarper and calls to add more will fail. Attempts to add a Dewarper to a Source in a state of `PLAYING` or `PAUSED` will fail.

**Parameters**
* `name` - [in] unique name of the Source to update
* `dewarper` - [in] unique name of the Dewarper to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_video_dewarper_add('my-uri-source', 'my-dewarper')
```

<br>

### *dsl_source_video_dewarper_remove*
```C
DslReturnType dsl_source_video_dewarper_remove(const wchar_t* name);
```
This service removes a [Dewarper](api-dewarper.md) component -- previously added with [dsl_source_video_dewarper_add](#dsl_source_video_dewarper_add) -- from a named Video Source. Calls to remove will fail if the Source is in a state of `PLAYING` or `PAUSED`.

**Parameters**
* `name` - [in] unique name of the Source to update

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_video_dewarper_remove('my-uri-source')
```

<br>

---

## App Source Methods
### *dsl_source_app_data_handlers_add*
```C
DslReturnType dsl_source_app_data_handlers_add(const wchar_t* name, 
    dsl_source_app_need_data_handler_cb need_data_handler, 
    dsl_source_app_enough_data_handler_cb enough_data_handler, 
    void* client_data);
```
Adds data-handler callback functions to a named App Source component.

**Parameters**
* `name` - [in] unique name of the Source to update.
* `need_data_handler` - [in] callback function to be called when new data is needed.
* `enough_data_handler` - [in] callback function to be called when the Source has enough data to process.
* `client_data` - [in]  opaque pointer to client data passed back into the client_handler functions.

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_data_handlers_add('my-app-source',
    my_need_data_handler, my_enough_data_handler, NULL)
```
<br>

### *dsl_source_app_data_handlers_remove*
```C
DslReturnType dsl_source_app_data_handlers_remove(const wchar_t* name);
```
This service removes data-handler callback functions -- previously added with [dsl_source_app_data_handlers_add](#dsl_source_app_data_handlers_add) -- from a named App Source component.

**Parameters**
* `name` - [in] unique name of the Source to update.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_data_handlers_remove('my-app-source')
```

<br>

### *dsl_source_app_buffer_push*
```C
DslReturnType dsl_source_app_buffer_push(const wchar_t* name, void* buffer);
```
This service pushes a new buffer to a uniquely named App Source component for processing.

**Parameters**
* `name` - [in] unique name of the Source to push to.
* `buffer` - [in] buffer to push to the App Source.

**Returns**
* `DSL_RESULT_SUCCESS` on successful push. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_buffer_push('my-app-source', buffer)
```
<br>

### *dsl_source_app_sample_push*
```C
DslReturnType dsl_source_app_sample_push(const wchar_t* name, void* sample);
```
This service pushes a new sample to a uniquely named App Source component for processing.

**Parameters**
* `name` - [in] unique name of the Source to push to.
* `sample` - [in] sample to push to the App Source.

**Returns**
* `DSL_RESULT_SUCCESS` on successful push. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_sample_push('my-app-source', sample)
```

<br>

### *dsl_source_app_eos*
```C
DslReturnType dsl_source_app_eos(const wchar_t* name);
```
This service notifies a uniquely named App Source component that no more buffers are available.

**Parameters**
* `name` - [in] unique name of the Source to end-of-stream.

**Returns**
* `DSL_RESULT_SUCCESS` on successful EOS. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_eos('my-app-source')
```

<br>

### *dsl_source_app_stream_format_get*
```C
DslReturnType dsl_source_app_stream_format_get(const wchar_t* name, 
    uint* stream_format);
```
This service gets the current stream-format setting for the named App Source Component.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `stream_format` - [out] one of the [DSL_STREAM_FORMAT](#dsl-stream-format-types) constants. Default = `DSL_STREAM_FORMAT_BYTE`.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, format = dsl_source_app_stream_format_get('my-app-source')
```
<br>

### *dsl_source_app_stream_format_set*
```C
DslReturnType dsl_source_app_stream_format_set(const wchar_t* name, 
    uint stream_format);
```
This service sets the stream-format setting for the named App Source Component.

**Parameters**
* `name` - [in] unique name of the Source to update.
* `stream_format` - [in] one of the [DSL_STREAM_FORMAT](#dsl-stream-format-types) constants.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_stream_format_set('my-app-source', DSL_STREAM_FORMAT_TIME)
```

<br>

### *dsl_source_app_block_enabled_get*
```C
DslReturnType dsl_source_app_block_enabled_get(const wchar_t* name, 
    boolean* enabled);
```
This service gets the block enabled setting for the named App Source Component. If true, when max-bytes are queued and after the enough-data signal has been emitted, the source will block any further push calls until the amount of queued bytes drops below the max-bytes limit.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `enabled` - [out] current block enabled setting. Default = FALSE.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_source_app_block_enabled_get('my-app-source')
```
<br>

### *dsl_source_app_block_enabled_set*
```C
DslReturnType dsl_source_app_block_enabled_set(const wchar_t* name, 
    boolean enabled);
```
This service sets the block enabled setting for the named App Source Component. If true, when max-bytes are queued and after the enough-data signal has been emitted, the source will block any further push calls until the amount of queued bytes drops below the max-bytes limit.

**Parameters**
* `name` - [in] unique name of the Source to update.
* `enabled` - [in]  new block-enabled setting to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_block_enabled_set('my-app-source', True)
```

<br>

### *dsl_source_app_current_level_bytes_get*
```C
DslReturnType dsl_source_app_current_level_bytes_get(const wchar_t* name,
    uint64_t* level);
```
This service gets the current level of queued data in bytes for the named App Source Component.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `level` - [out] current queue level in units of bytes.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, current_level = dsl_source_app_current_level_bytes_get('my-app-source')
```
<br>

### *dsl_source_app_max_level_bytes_get*
```C
DslReturnType dsl_source_app_max_level_bytes_get(const wchar_t* name,
    uint64_t* level);
```
This service gets the maximum amount of bytes that can be queued for the named App Source Component. After the maximum amount of bytes are queued, the App Source will call the [dsl_source_app_enough_data_handler_cb](#dsl_source_app_enough_data_handler_cb) callback function.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `level` - [out] current max-level in units of bytes. Default = 200000.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, max_level = dsl_source_app_max_level_bytes_get('my-app-source')
```
<br>

### *dsl_source_app_max_level_bytes_set*
```C
DslReturnType dsl_source_app_max_level_bytes_set(const wchar_t* name,
    uint64_t level);
```
This service sets the maximum amount of bytes that can be queued for the named App Source component. After the maximum amount of bytes are queued, the App Source will call the [dsl_source_app_enough_data_handler_cb](#dsl_source_app_enough_data_handler_cb) callback function.

**Parameters**
* `name` - [in] unique name of the Source to update.
* `level` - [in]  new max-level in units of bytes.  Default = 200000.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_app_max_level_bytes_set('my-app-source', 100000)
```

<br>

### *dsl_source_app_do_timestamp_get*
```C
DslReturnType dsl_source_app_do_timestamp_get(const wchar_t* name, 
    boolean* do_timestamp);
```
This service gets the do-timestamp setting for the named App Source component.

**Parameters**
* `source` - [in] unique name of the App Source to query.
* `do_timestamp` - [out]  if TRUE, the source will automatically timestamp outgoing buffers based on the current running_time.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval, do_timestamep = dsl_source_app_do_timestamp_get('my-app-source')
```

<br>

### *dsl_source_app_do_timestamp_set*
```C
DslReturnType dsl_source_app_do_timestamp_set(const wchar_t* name, 
    boolean do_timestamp);
```
This service sets the do-timestamp setting for the named App Source component.

**Parameters**
* `source` - [in] unique name of the App Source to update.
* `do_timestamp` - [in]  set to TRUE to have the source automatically timestamp outgoing buffers based on the current running_time.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure

**Python Example**
```Python
retval = dsl_source_app_do_timestamp_set('my-app-source', True)
```

<br>

## CSI Source Methods
### *dsl_source_csi_sensor_id_get*
```C
DslReturnType dsl_source_csi_sensor_id_get(const wchar_t* name,
    uint* sensor_id);
```
This service gets the sensor-id setting for the named CSI Source. A unique sensor-id is assigned to each CSI Source on creation starting with 0. The default setting can be overridden by calling [dsl_source_decode_uri_set](#dsl_source_decode_uri_set). The call will fail if the given sensor-id is not unique. If a source is deleted, the sensor-id will be re-assigned to a new CSI Source if one is created.

**Parameters**
* `name` - [in] unique name of the Source to query.
* `sensor_id` - [out] unique sensor-id in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, sensor_id = dsl_source_csi_sensor_id_get('my-csi-source')
```
<br>

### *dsl_source_csi_sensor_id_set*
```C
DslReturnType dsl_source_csi_sensor_id_set(const wchar_t* name,
    uint sensor_id);
```
This service sets the sensor-id setting for the named CSI Source to use. A unique sensor-id is assigned to each CSI Source on creation starting with 0. This service will fail if the given sensor-id is not unique. If a source is deleted, the sensor-id will be re-assigned to a new CSI Source if one is created.

**Parameters**
* `name` - [in] unique name of the Source to update.
* `sensor_id` - [in] unique sensor-id for the Source to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_csi_sensor_id_set('my-csi-source', 1)
```

<br>

## USB Source Methods
### *dsl_source_usb_device_location_get*

```C
DslReturnType dsl_source_usb_device_location_get(const wchar_t* name,
    const wchar_t** device_location);
```
This service gets the device-location setting for the named USB Source. A unique device-location is assigned to each USB Source on creation starting with `/dev/video0`, followed by `/dev/video1`, and so on. The default assignment can be overridden by calling [dsl_source_usb_device_location_set](#dsl_source_usb_device_location_set). The call will fail if the given device-location is not unique. If a source is deleted, the device-location will be re-assigned to a new USB Source if one is created.


**Parameters**
* `name` - [in] unique name of the Source to query.
* `device_location` - [out] device location string in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, device_location = dsl_source_usb_device_location_get('my-usb-source')
```
<br>

### *dsl_source_usb_device_location_set*
```C
DslReturnType dsl_source_usb_device_location_set(const wchar_t* name,
    const wchar_t* device_location);
```
This service sets the sensor-id setting for the named CSI Source to use.  A unique device-location is assigned to each USB Source on creation, starting with `/dev/video0`, followed by `/dev/video1`, and so on. This service will fail if the given device-location is not unique. If a source is deleted, the device-location will be re-assigned to a new USB Source if one is created.

**Parameters**
* `name` - [in] unique name of the Source to update.
* `device_location` - [in] unique device-location for the Source to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_usb_device_location_set('my-usb-source', '/dev/video1')
```

<br>

## URI Source Methods
### *dsl_source_uri_uri_get*
```C
DslReturnType dsl_source_uri_uri_get(const wchar_t* name, const wchar_t** uri);
```
This service gets the current URI in use for the named URI source.

**Parameters**
* `name` - [in] unique name of the URI Source to query.
* `uri` - [out] uniform resource identifier (URI) in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, uri = dsl_source_uri_uri_get('my-uri-source')
```
<br>

### *dsl_source_uri_uri_set*
```C
DslReturnType dsl_source_uri_uri_set(const wchar_t* name, const wchar_t* uri);
```
This service sets the URI for the named URI source to use.

**Parameters**
* `name` - [in] unique name of the URI Source to update.
* `uri` - [in] uniform resource identifier (URI) for the Source to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_uri_uri_set('my-uri-source', '../../test/streams/sample_1080p_h264.mp4')
```

<br>

## File Source Methods
### *dsl_source_file_file_path_get*
```C
DslReturnType dsl_source_file_file_path_get(const wchar_t* name, 
    const wchar_t** file_path);
```
This service gets the current file-path in use for the named File Source.

**Parameters**
* `name` - [in] unique name of the File Source to query
* `file_path` - [out] file path setting in use by the File Source

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, file_path = dsl_source_file_file_path_get('my-file-source')
```
<br>

### *dsl_source_file_file_path_set*
```C
DslReturnType dsl_source_file_file_path_set(const wchar_t* name, 
    const wchar_t* file_path);
```
This service sets the file path for the named File Source to use.

**Parameters**
* `name` - [in] unique name of the File Source to update.
* `file_path` - [in] absolute or relative File Path to a new file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_file_file_path_set('my-file-source', './streams/sample_1080p_h264.mp4')
```

<br>

### *dsl_source_file_repeat_enabled_get*
```C
DslReturnType dsl_source_file_repeat_enabled_get(const wchar_t* name, boolean* enabled);
```
This service gets the current repeat-enabled setting in use by the named File source

**Parameters**
* `name` - [in] unique name of the Source to query
* `repeat_enabled` - [out] if true, the File source will repeat the file on end-of-stream (EOS).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, repeat_enabled = dsl_source_file_repeat_enabled_get('my-file-source')
```
<br>

### *dsl_source_file_repeat_enabled_set*
```C
DslReturnType dsl_source_file_repeat_enabled_set(const wchar_t* name, boolean enabled);
```
This service sets the repeat-enabled setting for the named File source to use.

**Parameters**
* `name` - [in] unique name of the Source to update
* `repeat_enabled` - [in] if true, the File source will repeat the file on an end-of-stream (EOS).

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_file_repeat_enabled_set('my-file-source', True)
```

<br>

## RTSP Source Methods
### *dsl_source_rtsp_uri_get*
```C
DslReturnType dsl_source_rtsp_uri_get(const wchar_t* name, const wchar_t** uri);
```
This service gets the current URI in use for the named RTSP source.

**Parameters**
* `name` - [in] unique name of the RTSP Source to query.
* `uri` - [out] uniform resource identifier (URI) in use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, uri = dsl_source_rtsp_uri_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_uri_set*
```C
DslReturnType dsl_source_rtsp_uri_set(const wchar_t* name, const wchar_t* uri);
```
This service sets the URI to for the named RTSP source.

**Parameters**
* `name` - [in] unique name of the URI Source to update.
* `uri` - [in] unique resource identifier (URI) for the Source to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_uri_set('my-rtsp-source', my_rtsp_uir)
```

<br>

### *dsl_source_rtsp_timeout_get*
```C
DslReturnType dsl_source_rtsp_timeout_get(const wchar_t* name, uint* timeout);
```
This service gets the current new-buffer timeout value for the named RTSP Source

**Parameters**
 * `name` - [in] unique name of the Source to query
 * `timeout` - [out] time to wait (in seconds) between successive frames before determining the connection is lost. If set to 0 then timeout is disabled.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, timeout = dsl_source_rtsp_timeout_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_timeout_set*
```C
DslReturnType dsl_source_rtsp_timeout_set(const wchar_t* name, uint timeout);
```
This service sets the new-buffer-timeout value for the named RTSP Source to use. Setting the `timeout` to 0 will disable stream management and terminate any reconnection cycle if in progress.

**Parameters**
 * `name` - [in] unique name of the Source to query
 * `timeout` - [in] time to wait (in seconds) between successive frames before determining the connection is lost. Set to 0 to disable timeout.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_timeout_set('my-rtsp-source', timeout)
```
<br>

### *dsl_source_rtsp_reconnection_params_get*
```C
DslReturnType dsl_source_rtsp_reconnection_params_get(const wchar_t* name, uint* sleep_ms, uint* timeout_ms);
```
This service gets the current reconnection params in use by the named RTSP Source. The parameters are set to `DSL_RTSP_RECONNECT_SLEEP_TIME_MS` and `DSL_RTSP_RECONNECT_TIMEOUT_MS` on Source creation.

**Parameters**
 * `name` - [in] unique name of the Source to query
 * `sleep_ms` - [out] time to sleep between successively checking the status of the asynchronous reconnection
 * `timeout_ms` - [out] time to wait before terminating the current reconnection try and restarting the reconnection cycle again.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, sleep_ms, timeout_ms = dsl_source_rtsp_reconnection_params_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_reconnection_params_set*
```C
DslReturnType dsl_source_rtsp_reconnection_params_get(const wchar_t* name, uint* sleep_ms, uint* timeout_ms);
```
This service sets the reconnection params for the named RTSP Source. The parameters are set to `DSL_RTSP_RECONNECT_SLEEP_TIME_MS` and `DSL_RTSP_RECONNECT_TIMEOUT_MS` on Source creation.

**Note:** Both `sleep_ms` and `time_out` must be greater than 10 ms. `time_out` must be >= `sleep_ms` and should be set as a multiple of. Calling this service during an active "reconnection-cycle" will terminate the current attempt with a new cycle started using the new parameters. The current number of retries will not be reset.

**Parameters**
 * `name` - [in] unique name of the Source to query
 * `sleep_ms` - [out] time to sleep between successively checking the status of the asynchronous reconnection
 * `timeout_ms` - [out] time to wait before terminating the current reconnection try and restarting the reconnection cycle again.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_reconnection_params_get('my-rtsp-source', sleep_ms, timeout_ms)
```
<br>

### *dsl_source_rtsp_connection_data_get*
```C
DslReturnType dsl_source_rtsp_connection_data_get(const wchar_t* name, dsl_rtsp_connection_data* data);
```
This service gets the current connection stats for the named RTSP Source.

**Parameters**
 * `name` - [in] unique name of the Source to query
 * `data` [out] - pointer to a [dsl_rtsp_connection_data](#dsl_rtsp_connection_data) structure.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, connection_data = dsl_source_rtsp_connection_data_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_connection_stats_clear*
```C
DslReturnType dsl_source_rtsp_connection_stats_clear(const wchar_t* name);
```
This service clears the current reconnection stats for the named RTSP Source.

**Note:** the connection `retries` count will not be cleared if `in_reconnect == true`

**Parameters**
 * `name` - [in] unique name of the Source to update
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_connection_stats_clear('my-rtsp-source')
```

<br>

### *dsl_source_rtsp_latency_get*
```C
DslReturnType dsl_source_rtsp_latency_get(const wchar_t* name, uint* latency);
```
This service gets the current latency setting for the named RTSP Source.

**Parameters**
 * `name` - [in] unique name of the Source to query.
 * `latency` - [out] current latency setting = amount of data to buffer in ms.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, latency = dsl_source_rtsp_latency_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_latency_set*
```C
DslReturnType dsl_source_rtsp_latency_set(const wchar_t* name, uint latency);
```
This service sets the latency setting for the named RTSP Source to use.

**Parameters**
 * `name` - [in] unique name of the Source to update
 * `latency` - [in] new latency setting = amount of data to buffer in ms. 
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_latency_set('my-rtsp-source', 2000)
```

<br>

### *dsl_source_rtsp_drop_on_latency_enabled_get*
```C
DslReturnType dsl_source_rtsp_drop_on_latency_enabled_get(const wchar_t* name, 
    boolean* enabled);
```
This service gets the current drop-on-latency setting for the named RTSP Source.

**Parameters**
 * `name` - [in] unique name of the Source to query.
 * `enabled` - [out] If true, tells the jitterbuffer to never exceed the given latency in size.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, enabled = dsl_source_rtsp_drop_on_latency_enabled_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_drop_on_latency_enabled_set*
```C
DslReturnType dsl_source_rtsp_drop_on_latency_enabled_set(const wchar_t* name, 
    boolean enabled);
```
This service sets the drop-on-latency setting for the named RTSP Source to use.

**Parameters**
 * `name` - [in] unique name of the Source to update
 * `enabled` - [in] Set to true to tell the jitterbuffer to never exceed the given latency in size.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_drop_on_latency_enabled_set('my-rtsp-source', True)
```

<br>

### *dsl_source_rtsp_tls_validation_flags_get*
```C
DslReturnType dsl_source_rtsp_tls_validation_flags_get(const wchar_t* name,
    uint* flags);
```
This service gets the current TLS certificate validation flags for the named RTSP Source.

**Parameters**
 * `name` - [in] unique name of the Source to query
 * `flags` - [out] mask of [TLS_certificate validation flags](/docs/api-source.md#tls-certificate-validation-flags). Default = `DSL_TLS_CERTIFICATE_VALIDATE_ALL`.
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, flags = dsl_source_rtsp_tls_validation_flags_get('my-rtsp-source')
```
<br>

### *dsl_source_rtsp_tls_validation_flags_set*
```C
DslReturnType dsl_source_rtsp_tls_validation_flags_set(const wchar_t* name,
    uint flags);
```
This service sets the TLS certificate validation flags for the named RTSP Source to use.

**Parameters**
 * `name` - [in] unique name of the Source to update
 * `flags` - [in] mask of [TLS_certificate validation flags](/docs/api-source.md#tls-certificate-validation-flags) constant values. 
 
**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_tls_validation_flags_set('my-rtsp-source',
    DSL_TLS_CERTIFICATE_UNKNOWN_CA | DSL_TLS_CERTIFICATE_BAD_IDENTITY)
```
<br>

### *dsl_source_rtsp_state_change_listener_add*
```C
DslReturnType dsl_source_rtsp_state_change_listener_add(const wchar_t* pipeline,
    state_change_listener_cb listener, void* user_data);
```
This service adds a callback function of type [dsl_state_change_listener_cb](#dsl_state_change_listener_cb) to a
RTSP Source identified by its unique name. The function will be called on every change-of-state with `old_state`, `new_state`, and the client provided `user_data`. Multiple callback functions can be registered with one Source, and one callback function can be registered with multiple Sources.

**Parameters**
* `name` - [in] unique name of the RTSP Source to update.
* `listener` - [in] state change listener callback function to add.
* `user_data` - [in] opaque pointer to user data returned to the client when listener is called back

**Returns**
* `DSL_RESULT_SUCCESS` on successful add. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
def state_change_listener(old_state, new_state, user_data, user_data):
    print('old_state = ', old_state)
    print('new_state = ', new_state)
   
retval = dsl_source_rtsp_state_change_listener_add('my-rtsp-source', 
 state_change_listener, None)
```

<br>

### *dsl_source_rtsp_state_change_listener_remove*
```C
DslReturnType dsl_source_rtsp_state_change_listener_remove(const wchar_t* name,
    dsl_state_change_listener_cb listener);
```
This service removes a callback function of type [state_change_listener_cb](#state_change_listener_cb) from a
pipeline identified by its unique name.

**Parameters**
* `pipeline` - [in] unique name of the Pipeline to update.
* `listener` - [in] state change listener callback function to remove.

**Returns**  
* `DSL_RESULT_SUCCESS` on successful removal. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_state_change_listener_remove('my-rtsp-source', 
 state_change_listener)
```

<br>

### *dsl_source_rtsp_tap_add*
```C
DslReturnType dsl_source_rtsp_tap_add(const wchar_t* name, const wchar_t* tap);
```
This service adds a named Tap to a named RTSP source. There is currently only one type of Tap which is the [Smart Recording Tap](/docs/api-tap.md#dsl_tap_record_new)

**Parameters**
 * `name` [in] unique name of the Source object to update
 * `tap` [in] unique name of the Tap to add

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_tap_add('my-rtsp-source', 'my-record-tap')
```

<br>

### *dsl_source_rtsp_tap_remove*
```C
DslReturnType dsl_source_rtsp_tap_remove(const wchar_t* name);
```

Removes a Tap component from an RTSP Source component. The call will fail if the RTSP source is without a Tap component.  

**Parameters**
 * `name` [in] name of the Source object to update

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_rtsp_tap_remove('my-rtsp-source')
```

<br>

## Interpipe Source Methods
### *dsl_source_interpipe_listen_to_get*
```C
DslReturnType dsl_source_interpipe_listen_to_get(const wchar_t* name,
    const wchar_t** listen_to);
```
This service gets the name of the [Interpipe Sink](/docs/api-sink.md#dsl_sink_interpipe_new) the named Interpipe Source is currently listening to.

**Parameters**
* `name` - [in] unique name of the Interpipe Source to query
* `listen_to` - [out]  unique name of the Interpipe Sink the Source is listening to.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, listen_to = dsl_source_interpipe_listen_to_get('my-interpipe-source')
```
<br>

### *dsl_source_interpipe_listen_to_set*
```C
DslReturnType dsl_source_interpipe_listen_to_get(const wchar_t* name,
    const wchar_t* listen_to);
```
This service sets the name of the [Interpipe Sink](/docs/api-sink.md#dsl_sink_interpipe_new) for the name Interpipe Source to listen to.

**Parameters**
* `name` - [in] unique name of the Interpipe Source to update.
* `listen_to` - [out]  unique name of the Interpipe Sink listening to.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_interpipe_listen_to_set('my-interpipe-source', 'my-interpipe-sink-2')
```
<br>

### *dsl_source_interpipe_accept_settings_get*
```C
DslReturnType dsl_source_interpipe_accept_settings_get(const wchar_t* name,
    boolean* accept_eos, boolean* accept_events);
```
This service gets the current accept settings in use by the named Interpipe Source.

**Parameters**
* `name` - [in] unique name of the Interpipe Source to query
* `accept_eos` - [out] if true, the Source accepts EOS events from the Interpipe Sink.
* `accept_event` - [out] if true, the Source accepts events (except EOS event) from the Interpipe Sink.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, accept_eos, accept_events = dsl_source_interpipe_accept_settings_get(
    'my-interpipe-source')
```
<br>

### *dsl_source_interpipe_accept_settings_set*
```C
DslReturnType dsl_source_interpipe_accept_settings_get(const wchar_t* name,
    boolean accept_eos, boolean accept_events);
```
This service sets the accept settings for the named Interpipe Source to use.

**Parameters**
* `name` - [in] unique name of the Interpipe Source to update
* `accept_eos` - [in] set to true to accept EOS events from the Inter-Pipe Sink, false otherwise.
* `accept_event` - [in] set to true to accept events (except EOS event) from the Inter-Pipe Sink, false otherwise.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_interpipe_accept_settings_get('my-interpipe-source',
    True, True)
```
<br>

## Image Source Methods
### *dsl_source_image_file_path_get*
```C
DslReturnType dsl_source_image_file_path_get(const wchar_t* name, 
    const wchar_t** file_path);
```
This service gets the current file-path in use for the named Image Source; Single-Image, Multi-Image or Image-Stream.

**Parameters**
* `name` - [in] unique name of the Image Source to query
* `file_path` - [out] file path setting in use by the File Source

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, file_path = dsl_source_image_file_path_get('my-single-image-source')
```
<br>

### *dsl_source_image_file_path_set*
```C
DslReturnType dsl_source_image_file_path_set(const wchar_t* name, 
    const wchar_t* file_path);
```
This service sets the file path to use for the named Image Source; Single-Image, Multi-Image or Image-Stream.

**Parameters**
* `name` - [in] unique name of the Image Source to update.
* `file_path` - [in] absolute or relative File Path to a new file to use.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_image_file_path_set('my-single-image-source', './streams/sample_1080p_h264.mp4')
```

<br>

## Multi Image Source Methods
### *dsl_source_image_multi_loop_enabled_get*
```C
DslReturnType dsl_source_image_multi_loop_enabled_get(const wchar_t* name,
    boolean* enabled);
```
This service gets the current loop-enabled setting for the named Multi-Image source.

**Parameters**
* `name` - [in] unique name of the Source to query
* `enabled` - [out] if true, the Multi-Image source will loop to the `start_index` (default=0) when the last image is played. The Source will stop on the last image if false (default).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, loop_enabled = dsl_source_image_multi_loop_enabled_get('my-multi-image-source')
```
<br>

### *dsl_source_image_multi_loop_enabled_set*
```C
DslReturnType dsl_source_image_multi_loop_enabled_set(const wchar_t* name,
    boolean enabled);
```
This service sets the loop-enabled setting for the named Multi-Image Source to use.

**Parameters**
* `name` - [in] unique name of the Source to update
* `enabled` - [in] if true, the Multi-Image source will loop to the `start_index` (default=0) when the last image is played. The Source will stop on the last image if false (default).

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_image_multi_loop_enabled_set('my-multi-image-source', True)
```

<br>

### *dsl_source_image_multi_indices_get*
```C
DslReturnType dsl_source_image_multi_indices_get(const wchar_t* name,
    int* start_index, int* stop_index);
```
This service gets the current start and stop index settings for the named Multi-Image source.

**Parameters**
* `name` - [in] unique name of the Source to query
* `start_index` - [out] index to start with. When the end of the loop is reached, the current index will be set to the start-index. Default = 0.
* `stop_index` - [out] index to stop on, Default = -1 (no stop).

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, start_index, stop_index = dsl_source_image_multi_indices_get('my-multi-image-source')
```
<br>

### *dsl_source_image_multi_indices_set*
```C
DslReturnType dsl_source_image_multi_indices_set(const wchar_t* name,
    int start_index, int stop_index);
```
This service sets the start and stop index settings for the named Multi-Image Source to use.

**Parameters**
* `name` - [in] unique name of the Source to update
* `start_index` - [in] index to start with. When the end of the loop is reached, the current index will be set to the start-index. Default = 0.
* `stop_index` - [in] index to stop on, Default = -1 (no stop).

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_image_multi_indices_set('my-multi-image-source', 10, -1)
```

<br>

### *dsl_source_image_multi_loop_enabled_set*
```C
DslReturnType dsl_source_image_multi_loop_enabled_set(const wchar_t* name,
    boolean enabled);
```
This service sets the loop-enabled setting for the named Multi-Image Source to use.

**Parameters**
* `name` - [in] unique name of the Source to update
* `enabled` - [in] if true, the Multi-Image source will loop to the `start_index` (default=0) when the last image is played. The Source will stop on the last image if false (default).

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_image_multi_loop_enabled_set('my-multi-image-source', True)
```

<br>

### *dsl_source_image_stream_timeout_get*
```C
DslReturnType dsl_source_image_stream_timeout_get(const wchar_t* name, uint* timeout);
```
This service gets the current timeout setting in use for the named Streaming Image source.

**Parameters**
* `name` - [in] unique name of the Image Source to query.
* `timeout` - [out] current timeout setting in units of seconds. 0 = no timeout.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, timeout = dsl_source_image_stream_timeout_get('my-image-source')
```
<br>

### *dsl_source_image_stream_timeout_set*
```C
DslReturnType dsl_source_image_stream_timeout_set(const wchar_t* name, uint timeout);
```
This service sets the File Path to use by the named Streaming Image source.

**Parameters**
* `name` - [in] unique name of the Image Source to update.
* `timeout` - [in] new timeout setting in units of seconds. 0 = no timeout.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_image_stream_timeout_set('my-image-source', 30)
```

<br>

### *dsl_source_duplicate_original_get*
```C
DslReturnType dsl_source_duplicate_original_get(const wchar_t* name, 
    const wchar_t** original);
```
This service gets the unique name of the Original Source assigned to the named Duplicate Source.

**Parameters**
* `name` - [in] unique name of the Duplicate Source to query.
* `original` - [out] unique name of the current Original Source assigned to named Duplicate Source.

**Returns**
* `DSL_RESULT_SUCCESS` on successful query. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval, original = dsl_source_duplicate_original_get('my-duplicate-source')
```
<br>

### *dsl_source_duplicate_original_set*
```C
DslReturnType dsl_source_duplicate_original_set(const wchar_t* name, 
    const wchar_t* original);
```
This service assigns the Original Source (by unique name) for the named Duplicate Source.

**Parameters**
* `name` - [in] unique name of the Duplicate Source to update
* `original` - [in] unique name of the new Original Source for this Duplicate Source.

**Returns**
* `DSL_RESULT_SUCCESS` on successful update. One of the [Return Values](#return-values) defined above on failure.

**Python Example**
```Python
retval = dsl_source_duplicate_original_set('my-duplicate-source', 'my-usb-source')
```

<br>

---

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Player](/docs/api-player.md)
* **Source**
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Preprocessor](/docs/api-preproc.md)
* [Inference Engine and Server](/docs/api-infer.md)
* [Tracker](/docs/api-tracker.md)
* [Segmentation Visualizer](/docs/api-segvisual.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer, Remxer, and Splitter Tees](/docs/api-tee.md)
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
