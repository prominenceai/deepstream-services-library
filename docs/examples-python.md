## *Warning: this page is extreamly out of date!* 

It is scheduled to be updated to coincide with the firts official Beta comming this fall. Untill then, please refer to the current set of working python examples [available here](/examples/python)

# DSL Python Examples
Note: Many of the examples use the NVIDIA® DeepStream [Python-bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps#python-bindings) (pyds.so),  which can be downloaded from [here](https://developer.nvidia.com/deepstream-download#python_bindings).

**List of Examples:**
* [1csi_live_pgie_demuxer_osd_overlay_rtsp_h264](#1csi_live_pgie_demuxer_osd_overlay_rtsp_h264)
* [1csi_live_pgie_ktl_tiller_redaction_osd_window](#1csi_live_pgie_ktl_tiller_redaction_osd_window)
* [1csi_live_pgie_tiler_osd_window](#1csi_live_pgie_tiler_osd_window)
* [1rtsp_1csi_live_pgie_tiler_osd_window](#1rtsp_1csi_live_pgie_tiler_osd_window)
* [1uri_file_dewarper_pgie_ktl_3sgie_tiler_osd_bmh_window](#1uri_file_dewarper_pgie_ktl_3sgie_tiler_osd_bmh_window)
* [1uri_file_pgie_iou_tiler_osd_bmh_window](#1uri_file_pgie_iou_tiler_osd_bmh_window)
* [1uri_file_pgie_ktl_tiler_osd_bmh_window](#1uri_file_pgie_ktl_tiler_osd_bmh_window)
* [1uri_file_pgie_ktl_tiler_osd_window_h264_mkv](#1uri_file_pgie_ktl_tiler_osd_window_h264_mkv)
* [1uri_file_pgie_ktl_tiler_osd_window_h265_mp4](#1uri_file_pgie_ktl_tiler_osd_window_h265_mp4)
* [1uri_file_pgie_ktl_tiler_osd_window_image_frame_capture](#1uri_file_pgie_ktl_tiler_osd_window_image_frame_capture)
* [1uri_file_pgie_ktl_tiler_window_image_object_capture](#1uri_file_pgie_ktl_tiler_window_image_object_capture)
* [1uri_https_tiler_window_dyn_overlay](#1uri_https_tiler_window_dyn_overlay)
* [2rtsp_splitter_demuxer_pgie_ktl_tiler_osd_window_2_file](#2rtsp_splitter_demuxer_pgie_ktl_tiler_osd_window_2_file)
* [2uri_file_pgie_ktl_3sgie_tiler_osd_bmh_window](#2uri_file_pgie_ktl_3sgie_tiler_osd_bmh_window)
* [4uri_file_pgie_ktl_tiler_osd_overlay](#4uri_file_pgie_ktl_tiler_osd_overlay)
* [4uri_live_pgie_tiler_osd_window](#4uri_live_pgie_tiler_osd_window)
* [dyn_uri_file_pgie_ktl_tiler_osd_window](#dyn_uri_file_pgie_ktl_tiler_osd_window)

### 1csi_live_pgie_demuxer_osd_overlay_rtsp_h264
* 1 Live CSI Camera Source
* Primary GIE using labels in config file
* Demuxer - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
  * `nvidia_osd_sink_pad_buffer_probe` batch-meta-handler (bmh) callback added
* Overlay Sink - render over main display (0)
* RTSP Sink - H.264 RTSP Server

### 1csi_live_pgie_ktl_tiller_redaction_osd_window
* 1 Live CSI Camera Source
* Primary GIE using labels in config file
* KTL Tracker
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
  * Redaction enabled for ClassId = 0
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  
### 1csi_live_pgie_tiler_osd_window
* 1 Live CSI Camera Source
* Primary GIE using labels in config file
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline

### 1rtsp_1csi_live_pgie_tiler_osd_window
* 1 Live RTSP Camera Source
* 1 Live CSI Camera Source
* Primary GIE using labels in config file
* Tiler
* On-Screen-Display
  * Clock enabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline

### 1uri_file_dewarper_pgie_ktl_3sgie_tiler_osd_bmh_window
* 1 URI File Source - playback of 360 degree camera source
* Dewarper using provided config file
* Primary GIE using labels in config file
* KTL Tracker
* 3 Secondary GIEs - all set to infer on the Primary GIE
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
  * `nvidia_osd_sink_pad_buffer_probe` batch-meta-handler (bmh) callback added
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline

### 1uri_file_pgie_iou_tiler_osd_bmh_window
* 1 H.264 URI File Source
* Primary GIE using labels in config file
* IOU Tracker using provided config file
* 3 Secondary GIEs - all set to infer on the Primary GIE
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
  * `nvidia_osd_sink_pad_buffer_probe` batch-meta-handler (bmh) callback added
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline

### 1uri_file_pgie_ktl_tiler_osd_bmh_window
* 1 H.264 URI File Source
* Primary GIE using labels in config file
* KTL Tracker
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
  * `nvidia_osd_sink_pad_buffer_probe` batch-meta-handler (bmh) callback added
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline
  
### 1uri_file_pgie_ktl_tiler_osd_window_h264_mkv
* 1 H.264 URI File Source
* Primary GIE using labels in config file
* KTL Tracker
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* File Sink
  * H.264 encoder
  * MKV media container
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline

### 1uri_file_pgie_ktl_tiler_osd_window_h265_mp4
* 1 H.265 URI File Source
* Primary GIE using labels in config file
* KTL Tracker
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* File Sink
  * H.265 encoder
  * MP4 media container
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline

### 1uri_file_pgie_ktl_tiler_osd_window_image_frame_capture
* 1 H.265 URI File Source
* Primary GIE using labels in config file
* KTL Tracker
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Image Sink
  * Outdir for jpeg image files set to current directory`./`
  * Frame Capture enabled with an interval of every 60th frame
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline
  
### 1uri_file_pgie_ktl_tiler_window_image_object_capture
* 1 H.265 URI File Source
* Primary GIE using labels in config file
* KTL Tracker
* Tiler - demuxer or tiler is required, even with one source
* On-Screen-Display
  * Clock enabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Image Sink
  * Outdir for jpeg image files set to current directory`./`
  * Object Capture enabled for PERSON and VEHICLE classes, both with a capture limit of 50 images.
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline
  
### 1uri_https_tiler_window_dyn_overlay
* 1 https URI source ('https://www.radiantmediaplayer.com/media/bbb-360p.mp4')
* Tiler - demuxer or tiler is required, even with one source
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Dynamic Add/Remove Overlay Sinks
  * Using `xwindow_key_event_handler`
  * Press `'+'` to add a new Overlay Sink - main display
  * Press `'-'` to remove last added
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline

### 2rtsp_splitter_demuxer_pgie_ktl_tiler_osd_window_2_file
* 2 Live RTSP Camera Source
* 1 Splitter with two branches to split the stream after the Stream Muxer
  * Branch 1.
    * 1 Demuxer to demux the batch stream back to 2 streams
    * 2 H.264 URI File Sink, one for each stream
  * Branch 2. 
    * Primary GIE using labels in config file
    * KTL Tracker
    * Tiler
    * On-Screen-Display
      * Clock enabled
      * Default colors
* Default X11 Window Sink
    * `xwindow_delete_event_handler` added to Pipeline
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  
### 2uri_file_pgie_ktl_3sgie_tiler_osd_bmh_window
* 1 H.264 URI File Source
* 1 H.265 URI File Source
* Primary GIE using labels in config file
* KTL Tracker
* 3 Secondary GIEs - all set to infer on the Primary GIE
* Tiler
* On-Screen-Display
  * Clock enabled
  * Default colors
  * `nvidia_osd_sink_pad_buffer_probe` batch-meta-handler (bmh) callback added
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline
* Other Callbacks
  * `eos_event_listener` added to Pipeline
  * `state_change_listener` added to Pipeline

### 2uri_file_pgie_ktl_demuxer_1osd_1overlay_1window
* 1st H.264 URI File Source
  * Overlay Sink - downstream of demuxer
* 2nd H.264 URI File Source
  * On-Screen-Display - downstream of demuxer
  * Default X11 Window Sink - downstream of demuxer
    * `xwindow_delete_event_handler` added to Pipeline
* Primary GIE using labels in config file
* KTL Tracker
* Demuxer
* Other Callbacks
  * `eos_event_listener` added to Pipeline

### 4uri_file_pgie_ktl_tiler_osd_overlay
* 4 H.264 URI File Sources
* Primary GIE using labels in config file
* KTL Tracker
* Tiler
* On-Screen-Display
  * Clock disabled
  * Default colors
* Overlay sink - main window
* Other Callbacks
  * `eos_event_listener` added to Pipeline

### 4uri_live_pgie_tiler_osd_window
* 4 http URI Live Sources - CalTrans traffic cammeras - low resolution
* Primary GIE using labels in config file
* Tiler
* On-Screen-Display
  * Clock disabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline

### dyn_uri_file_pgie_ktl_tiler_osd_window
* Dynamic Add/Remove URI File Sources - initially 1
  * Using `xwindow_key_event_handler`
  * Press `'+'` to add a new URI FIle Source
  * Press `'-'` to remove last added
* Primary GIE using labels in config file
* Tiler
* On-Screen-Display
  * Clock disabled
  * Default colors
* Default X11 Window Sink
  * `xwindow_delete_event_handler` added to Pipeline
  * `xwindow_key_event_handler` added to Pipeline


