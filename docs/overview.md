# DSL Overview
### Overview Contents
* [Introduction](#introduction)
* [Pipeline Components](#pipeline-components)
  * [Streaming Sources](#streaming-sources)
  * [Inference Engines and Servers](#inference-engines-and-servers)
  * [Multi-Object Trackers](#multi-object-trackers)
  * [Multi-Source Tiler](#multi-source-tiler)
  * [On-Screen Display](#on-screen-display)
  * [Rendering and Encoding Sinks](#rendering-and-encoding-sinks)
  * [Tees and Branches](#tees-and-branches)
  * [Pad Probe Handlers](#pad-probe-handlers)
* [Display Types](#display-types)
* [Object Detection Event (ODE) Services](#object-detection-event-ode-services)
  * [ODE Triggers](#ode-triggers)
  * [ODE Actions](#ode-actions)
  * [ODE Areas](#ode-areas)
  * [ODE Line Crossing Analytics](#ode-line-crossing-analytics)
  * [ODE Heat Mapping](#ode-heat-mapping)
* [Smart Recording](#smart-recording)
* [RTSP Stream Connection Management](#rtsp-stream-connection-management)
* [X11 Window Services](#x11-window-services)
* [Player Services](#player-services)
* [SMTP Services](#smtp-services)
* [DSL Initialization](#dsl-initialization)
* [DSL Delete All](#dsl-delete-all)
* [Main Loop Context](#main-loop-context)
* [Service Return Codes](#service-return-codes)
* [Docker Support](#docker-support)
* [API Reference](#api-reference)

## Introduction
The DeepStream Services Library (DSL) is best described as "the NVIDIA DeepStream Reference Applications reimagined as a shared library of DeepStream pipeline services".

[NVIDIA’s DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) -- built on the open source [GStreamer](https://gstreamer.freedesktop.org/) "*an extremely powerful and versatile framework*<sup id="a1">[1](#f1)</sup>" -- enables experienced software developers to "*Seamlessly Develop Complex Stream Processing Pipelines*<sup id="a2">[2](#f2)</sup>". 

For those new to DeepStream, however, GStreamer comes with a learning curve that can be steep or lengthy for some. 

The core function of DSL is to provide a [simple and intuitive API](/docs/api-reference-list.md) for building, playing, and dynamically modifying NVIDIA® DeepStream Pipelines. Modifications made: (1) based on the results of the real-time video analysis, and: (2) by the application user through external input. An example of each:
1. Automatically starting a pre-cached recording session based on the occurrence of specific objects.
2. Interactively switching the view from one rendered Source stream to another on mouse click. 

The general approach to using DSL is to:
1. Create several uniquely named [Components](/docs/api-reference-list.md), each with a specific task to perform. 
2. Define one or more [Client callback functions](/docs/api-pipeline.md#client-callback-typedefs) and/or [Pad Probe Handlers](/docs/api-pph.md)(optional).
4. Add the Components and Callback functions to a new Pipeline.
5. Play the Pipeline and start/join the main execution loop.

Using Python3, for example, the above can be written as:

Create a set of Components, each with a specific function and purpose. 
```Python
# new Camera Sources - setting dimensions and frames-per-second
retval += dsl_source_csi_new('my-source', width=1280, height=720, fps_n=30, fps_d=1)

# create more Source Components as needed
# ...

# new Primary Inference Engine - path to config file and model engine, interval=0 - infer on every frame
retval += dsl_infer_gie_primary_new('my-pgie', path_to_config_file, path_to_model_engine, interval=0)

# new Multi-Source Tiler with dimensions of width and height 
retval += dsl_tiler_new('my-tiler', width=1280, height=720)

# new On-Screen Display for inference visualization - bounding boxes and labels - 
# with both labels and clock enabled
retval += dsl_osd_new('my-osd', text_enabled=True, clock_enabled=True,
    bbox_enabled=True, mask_enabled=False)

# new X11/EGL Window Sink for video rendering - Pipeline will create a new XWindow if one is not provided
retval += dsl_sink_window_new('my-window-sink', width=1280, height=720)

if retval != DSL_RESULT_SUCCESS:
    # one of the components failed to create, handle error
```

Add the components to a new Pipeline.

```Python
# Using a Null terminated list - in any order
retval = dsl_pipeline_new_component_add_many('my-pipeline', components=
    ['my-source', 'my-pgie', 'my-tiler', 'my-osd', 'my-sink', None])
```
Add one or more Client Callback Functions

```Python
# Function to be called on XWindow Delete event
def xwindow_delete_event_handler(client_data):
    # Quit the main loop to shut down and release all resources
    dsl_main_loop_quit()

# add the callback function to the pipeline
retval = dsl_pipeline_xwindow_delete_event_handler_add('my pipeline', xwindow_delete_event_handler, None)
```

Transition the Pipeline to a state of Playing and start/join the main loop

```Python
retval = dsl_pipeline_play('my-pipeline')
if retval != DSL_RESULT_SUCCESS:
    # Pipeline failed to play, handle error
  
 # join the main loop until stopped. 
 dsl_main_loop_run()
 
 # free up all resources
 dsl_delete_all()
 ```

## Pipeline Components
There are seven categories of Components that can be added to a Pipeline, automatically assembled in the order shown below. Many of the categories support multiple types and in the cases of Sources, Secondary Inference Engines, and Sinks, multiple types can be added to a single Pipeline. 

![DSL Components](/Images/dsl-components.png)

## Streaming Sources
Streaming sources are the head component(s) for all Pipelines and all Pipelines must have at least one Source (among other components) before they can transition to a state of Playing. All Pipelines have the ability to multiplex multiple streams -- using their own built-in Stream-Muxer -- as long as all Sources are of the same play-type; live vs. non-live with the ability to Pause. 

There are currently six types of Source components, two live connected Camera Sources:
* Camera Serial Interface (CSI) Source - connected to one of the serial ports on the Jetson SOM
* Universal Serial Bus (USB) Source

Two live decode Sources that.
* Universal Resource Identifier (URI) Source - supports non-live files as well.
* Real-time Streaming Protocol (RTSP) Source

Two non-live Sources 
* File Source that is derived from the URI Decode Source with some of the parameters fixed.
* Image Source that overlays an Image on a mock/fake streaming source at a settable frame rate. The Image Source can mimic a live source allowing it to be batched with other live streaming sources.

All Sources have dimensions, width and height in pixels, and frame-rates expressed as a fractional numerator and denominator.  The URI Source component support multiple codec formats, including H.264, H.265, PNG, and JPEG. A [Dewarper Component](/docs/api-dewarper.md) (not show in the image above) capable of dewarping 360 degree camera streams can be added to both. 

A Pipeline's Stream-Muxer has settable output dimensions with a decoded, batched output stream that is ready to infer on.

A [Record-Tap](#smart-recording) (not show in the image above) can be added to a RTSP Source for cached pre-decode recording, triggered on the occurrence of an [Object Detection Event (ODE)](#object-detection-event-pad-probe-handler).

See the [Source API](/docs/api-source.md) reference section for more information.

## Inference Engines and Servers
NVIDIA's GStreamer Inference Engines (GIEs) and Triton Inference Servers (TISs), using pre-trained models, classify data to “infer” a result, e.g.: person, dog, car? A Pipeline may have at most one Primary Gst Inference Engine (PGIE) or Primary Triton Inference Server (PTIS) -- with a specified set of classification labels to infer-with -- and multiple Secondary Gst Inference Engines (SGIEs) or Secondary Triton Inference Servers (STISs) that can Infer-on the output of either the Primary or other Secondary GIEs/TISs. Although optional, a Primary Inference Engine or Server is required when adding a Multi-Object Tracker, Secondary Inference Engines or Servers, or On-Screen-Display to a Pipeline.

After creation, GIEs and TISs can be updated to use a new model-engine (GIE only), config file, and/or inference interval 

With Primary GIEs and TISs, applications can:
* Add/remove [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers with Metadata for each Frame and Detected-Object found within. 
* Enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Inference Engine and Server API](/docs/api-infer.md) reference section for more information.

DSL supports NVIDIA's [Segmentation Visualizer plugin](https://docs.nvidia.com/metropolis/deepstream/5.0DP/plugin-manual/index.html#page/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.3.11.html#wwpID0E0WT0HA) for viewing segmentation results produced from either a Primary Gst Inference Engine (PGIE) or Primary Triton Inference Server (TIS).

See the [Segmentation Visualizer API](/docs/api-segvisual.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers -- with Metadata for each Frame and Detected-Object.

Tracker components are optional and a Pipeline, or [Branch](#tees-and-branches) can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more details. See NVIDIA's [Low-Level Tracker Library Comparisons and Tradeoffs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.3.02.html#wwpID0E0Q20HA) for additional information.

## Multi-Source Tiler
All Source components connect to the Pipeline's internal Stream-Muxer -- responsible for batching multiple sources and adding the meta-data structures to each frame -- even when there is only one. When using more that one source, the multiplexed stream must either be Tiled **or** Demuxed before reaching an On-Screen Display or Sink component downstream.

Tiler components transform the multiplexed streams into a 2D grid array of tiles, one per Source component. Tilers output a single stream that can connect to a single On-Screen Display (OSD). When using a Tiler, the OSD (optional) and Sinks (minimum one) are added directly to the Pipeline or Branch to operate on the Tiler's single output stream.
```Python
# assumes all components have been created first
retval = dsl_pipeline_component_add_many('my-pipeline', 
    ['src-1', 'src-2', 'pgie', 'tiler', 'osd', 'rtsp-sink', 'window-sink', None])
```
Tilers have dimensions, width and height in pixels, and rows and columns settings that can be updated at any time. The Tiler API provides services to show a single source with [dsl_tiler_source_show_set](/docs/api-timer.md#dsl_tiler_source_show_set) and return to the tiled view with [dsl_tiler_source_show_all](/docs/api-tiler.md#dsl_tiler_source_show_all). The source shown can be controlled manually with operator input, and automatically using [Object Detection Event](#)

Clients of Tiler components can add/remove one or more [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers -- with Metadata for each Frame and Detected-Object within.

See the [Multi-Source Tiler](/docs/api-tiler.md) reference section for additional information.

## On-Screen Display
On-Screen Display (OSD) components highlight detected objects with colored bounding boxes and labels. A Clock with Positional offsets, colors and fonts can be enabled for Display. ODE Actions can be used to add/update Frame and Object metadata for the OSD to display. 

OSDs are optional and a Pipeline (or Branch) can have at most one when using a Tiler or one-per-source when using a Demuxer. See the [On-Screen Display API](/docs/api-osd.md) reference section for more information. 

Clients of On-Screen Display components can add/remove one or more [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers -- with Metadata for each Frame and Detected-Object.

## Rendering and Encoding Sinks
Sinks, as the end components in the Pipeline, are used to render the video stream for visual display or encode the streaming video to a file or network. All Pipelines require at least one Sink Component to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case where the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently six types of Sink Components that can be added.

Clients can add/remove one or more [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers -- with Metadata for each Frame and Detected-Object -- on the input (sink pad) only.

1. Overlay Render Sink - Jetson platform only
2. X11/EGL Window Sink
3. File Sink
4. Record Sink
5. RTSP Sink
6. WebRTC Sink - Requires GStreamer 1.18 or later
7. IoT Message Sink
8. Fake Sink

Overlay and Window Sinks have settable dimensions: width and height in pixels, and X and Y directional offsets that can be updated after creation. 

The File and Record encoder Sinks support three codec formats: H.264, H.265 and MPEG-4, with two media container formats: MP4 and MKV.  See [Smart Recording](#smart-recording) below for more information on using Record Sinks.

RTSP Sinks create RTSP servers - H.264 or H.265 - that are configured when the Pipeline is called to Play. The server is started and attached to the Main Loop context once [dsl_main_loop_run](#dsl-main-loop-functions) is called. Once started, the server can accept connections based on the Sink's unique name and settings provided on creation. The below for example,

```Python
retval = dsl_sink_rtsp_new('my-rtsp-sink', 
    host='my-jetson.local', udp_port=5400, rtsp_port=8554, codec=DSL_CODEC_H265, bitrate=200000, interval=0)
```
would use
```
rtsp://my-jetson.local:8554/my-rtsp-sink
```

The Message Sink converts Object Detection Event (ODE) data into protocol specfic IoT messages and brokers/sends the messages to a remote entity.

See the [Sink API](/docs/api-sink.md) reference section for more information.

<br>

## Tees and Branches
There are two types of Tees that can be added to a Pipeline: Demuxer and Splitter.
1. **Demuxer** - used to demultiplex the single batched output from the Stream-muxer back into separate data streams.  
2. **Splitter** - used to split the stream, batched or otherwise, into multiple duplicate streams. 

Branches connect to the downstream/output pads of the Tee, either as a single component in the case of a Sink or another Tee, or as multiple linked components as in the case of **Branch 1** shown below. 

Important Notes: 
* Single component Branches can be added to a Tee directly, while multi-component Branches must be added to a new Branch component first.
* Branches ***can*** be added and removed from a Tee while a Pipeline is in a state of `Playing`, but the Tee must always have one. A [Fake Sink](/docs/api-sink.md) can be used as a Fake Branch when required.
* Tees are ***not*** required when adding multiple Sinks to a Pipeline or Branch. Multi-sink management is handled by the Pipeline/Branch directly. 

The following example illustrates how a **Pipeline** is assembled with a **Splitter**, **Demuxer**, and **Branch** components. 

![Tees and Branches](/Images/tees-and-branches.png)

#### Building the Pipeline Example above, 

The first step is to create all components for **Branch 1** and assemble - Multi-Source Tiler, On-Screen Display and X11/EGL Window Sink.

![Tees and Branches](/Images/tees-and-branches-branch-1.png)

```Python
# New Tiler, On-Screen Display, and Window Sink
retval = dsl_tiler_new('tiler', width=1920, height=540)
retval = dsl_osd_new('my-osd', text_enabled=True, clock_enabled=True,
    bbox_enabled=True, mask_enabled=False)
retval = dsl_sink_window_new('window-sink', x_offset=0, y_offset=0, width=1920, height=540)

# New Branch component to assemble Branch-1
retval = dsl_branch_new_components_add_many('branch-1', components=['tiler', 'osd', 'window-sink', None])
```

Next, create the Overlay Sinks which become **Branch 3** and **Branch 4** when added to the **Demuxer Tee**, which in turn becomes **Branch 2** when added to the **Splitter Tee** in the next step.

![Tees and Branches](/Images/tees-and-branches-branch-2-3-4.png)

```Python
# New Overlay Sink components to display the non-annotated demuxed video.
retval = dsl_sink_overlay_new('sink-overlay-1', overlay_id=0, display_id=0, depth=0,
   x_offset=20, y_offset=20, width=240, height=135
retval = dsl_sink_overlay_new('sink-overlay-2', overlay_id=0, display_id=0, depth=0,
   x_offset=980, y_offset=20, width=240, height=135

# New Demuxer to to demux into separate streams, one per source.
retval = dsl_tee_demuxer_new_branch_add_many('demuxer', branches=['sink-overlay-1', 'sink-overlay-2', None])
```

Next, create the **Splitter Tee** and add **Branch-1** and the **Demuxer Tee** as **Branch 2**

![Tees and Branches](/Images/tees-and-branches-branch-1-2-3-4.png)


```Python
# New Splitter to split the stream before the On-Screen Display
retval = dsl_tee_splitter_new_branch_add_many('splitter', branches=['branch-1, 'demuxer', None])
```

Last, create the two RTMP Decode Sources, Primary GIE, and Tracker. Then add the components and the Splitter to a new Pipeline

```Python
# For each camera, create a new RTSP Decode Source for the specific RTSP URI
retval = dsl_source_rtsp_new('src-1', 
    url = rtsp_uri_1, 
    protocol = DSL_RTP_ALL, 
    intra_decode = Fale, 
    drop_frame_interval = 0, 
    latency=100)

retval = dsl_source_rtsp_new('src-2', 
    url = rtsp_uri_2, 
    protocol = DSL_RTP_ALL, 
    intra_decode = Fale, 
    drop_frame_interval = 0, 
    latency=100)

retval = dsl_gie_primary_new('pgie', path_to_engine_file, path_to_config_file, interval=0)
retval = dsl_tracker_ktl_new('tracker', max_width=480, max_height=270)

retval = dsl_pipeline_new_components_add_many('pipeline', 
    components=['src-1', 'src-2', 'pgie', 'tracker', 'splitter'])

# ready to play
retval = dsl_pipeline_play('pipeline')

```

### All combined, the example is written as.

```Python
# NOTE: this example assumes that all return values are checked for DSL_RESULT_SUCCESS before proceeding

# New Tiler, On-Screen Display, and Window Sink
retval = dsl_tiler_new('tiler', width=1920, height=540)
retval = dsl_osd_new('my-osd', text_enabled=True, clock_enabled=True,
    bbox_enabled=True, mask_enabled=False)
retval = dsl_sink_window_new('window-sink', x_offset=0, y_offset=0, width=1920, height=540)

# New Branch component to assemble Branch-1
retval = dsl_branch_new_components_add_many('branch-1', components=['tiler', 'osd', 'window-sink', None])

# New Overlay Sink components to display the non-annotated demuxed video.
retval = dsl_sink_overlay_new('sink-overlay-1', overlay_id=0, display_id=0, depth=0,
   x_offset=20, y_offset=20, width=240, height=135
retval = dsl_sink_overlay_new('sink-overlay-2', overlay_id=0, display_id=0, depth=0,
   x_offset=980, y_offset=20, width=240, height=135

# New Demuxer to to demux into separate streams, one per source.
retval = dsl_tee_demuxer_new_branch_add_many('demuxer', branches=['sink-overlay-1', 'sink-overlay-2', None])

# New Splitter to split the stream before the On-Screen Display
retval = dsl_tee_splitter_new_branch_add_many('splitter', branches=['branch-1, 'demuxer', None])

# For each camera, create a new RTSP Decode Source for the specific RTSP URI
retval = dsl_source_rtsp_new('src-1', 
    url = rtsp_uri_1, 
    protocol = DSL_RTP_ALL, 
    intra_decode = Fale, 
    drop_frame_interval = 0, 
    latency=100)

retval = dsl_source_rtsp_new('src-2', 
    url = rtsp_uri_2, 
    protocol = DSL_RTP_ALL, 
    intra_decode = Fale, 
    drop_frame_interval = 0, 
    latency=100)

retval = dsl_gie_primary_new('pgie', path_to_engine_file, path_to_config_file, interval=0)
retval = dsl_tracker_ktl_new('tracker', max_width=480, max_height=270)

retval = dsl_pipeline_new_components_add_many('pipeline', 
    components=['src-1', 'src-2', 'pgie', 'tracker', 'splitter'])

# ready to play
```

See the [Demuxer and Splitter Tee API](/docs/api-tee.md) reference section for more information. 

---

## Pad Probe Handlers
Pipeline components are linked together using directional ["pads"](https://gstreamer.freedesktop.org/documentation/gstreamer/gstpad.html?gi-language=c) with a Source Pad from one component as the producer of data connected to the Sink Pad of the next component as the consumer. Data flowing over the component’s pads can be monitored, inspected and updated using a Pad-Probe with a specific Handler function.

There are three Pad Probe Handlers that can be created and added to either a Sink or Source Pad of most Pipeline components excluding Sources, Taps and Secondary GIE's.
1. Pipeline Meter - measures the throughput for each source in the Pipeline.
2. Object Detection Event Handler - manages a collection of [Triggers](/docs/api-ode-trigger.md) that invoke [Actions](/docs/api-ode-action.md) on the occurrence of specific frame and object metadata. 
3. Custom Handler- allows the client to install a callback with custom behavior. 

See the [Pad Probe Handler API](/docs/api-pph.md) reference section for additional information.

### Pipeline Meter Pad Probe Handler
The [Meter Pad Probe Handler](/docs/api-pph.md#pipeline-meter-pad-probe-handler) measures a Pipeline's throughput for each Source detected in the batched stream. When creating a Meter PPH, the client provides a callback function to be notified with new measurements at a specified interval. The notification includes the average frames-per-second over the last interval and over the current session, which can be stopped with new session started at any time. 

### Object Detection Event Pad Probe Handler
The [Object Detection Event (ODE) Pad Probe Handler](/docs/api-pph.md#object-detection-event-ode-pad-probe-handler) manages an ordered collection of **Triggers**, each with an ordered collection of **Actions** and an optional collection of **Areas**. Together, the Triggers, Areas and Actions provide a full set of [Object Detection Event Services](#object-detection-event-ode-services). 


### Custom Pad Probe Handler
Client applications can create one or more [Custom Pad Probe Handlers](/docs/api-pph.md#custom-pad-probe-handler) with callback functions to be called with every buffer that flows over a component's pad.

Using Python and [NVIDIA's python bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) for example:

```Python
retval = dsl_pph_custom_new('custom-handler', client_handler=handle_buffer, client_data=my_client_data)
```
The callback function can 
```Python
def handle_buffer(buffer, client_data)

    # retrieve the batch metadata from the gst_buffer using NVIDIA's python bindings.
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    
    # cast the opaque client data back to a python object and dereference
    py_client_data = cast(client_data, POINTER(py_object)).contents.value
    
    # process/update the batch_meta as desired. 
    # ...
    
    # return true to continue processing or false to self-remove
    return true
```

Refer to the [ODE Pad Probe Handler API Reference](/docs/api-pph.md) for more information.

---

## Display Types
On-Screen Display Types -- RGBA text and shapes -- can be added to a frame's metadata to be shown by an [On-Screen Display](/docs/api-osd.md) component downstream. 
There are eight base types used when creating other complete types for actual display. 
* RGBA Costom Color
* RGBA Predefined Color
* RGBA Random Color
* RGBA On-Demand Color
* RGBA Custom Color Palette
* RGBA Predefined Color Palette
* RGBA Random Color Palette
* RGBA Font

There are seven types for displaying text and shapes. 
* RGBA Text
* RGBA Line
* RGBA Multi-Line
* RGBA Arrow
* RGBA Rectangle
* RGBA Polygon
* RGBA Circle

And three types for displaying source information specific to each frame. 
* Source Number
* Source Name
* Source Dimensions

<br>

The image below provides examples of the Display Types listed above.

![RGBA Display Types](/Images/display-types.png)

Refer to the [Display Type API Reference](/docs/api-display-type.md) for more information.

---

## Object Detection Event (ODE) Services
DSL Provides an extensive set of ODE Triggers -- to Trigger on specific detection events -- and ODE Actions -- to perform specific action when a detection event occurrs. Triggers use settable criteria to process the Frame and Object metadata produced by the Primary and Secondary GIE's looking for specific detection events. When the criteria for the Trigger is met, the Trigger invokes all Actions in its ordered collection. Each unique Area and Action created can be added to multiple Triggers as shown in the diagram below. The ODE Handler has n Triggers, each Trigger has one shared Area and one unique Area, and one shared Action and one unique Action.

![ODE Services](/Images/ode-services.png)

The Handler can be added to the Pipeline before the On-Screen-Display (OSD) component allowing Actions to update the metadata for display. All Triggers can be enabled and re-enabled at runtime, either by an ODE Action, a client callback function, or directly by the application at any time.

### ODE Triggers
Current **ODE Triggers** supported:
* **Always** - triggers on every frame. Once per-frame always.
* **Absence** - triggers on the absence of objects within a frame. Once per-frame at most.
* **Occurrence** - triggers on each object detected within a frame. Once per-object at most.
* **Instance** - triggers on each new object instance across frames based on a unique tracker id. Once per new tracking id. 
* **Persitence** - triggers on each object instance that persists in view/frame for a specified period of time.
* **Summation** - triggers on the summation of all objects detected within a frame. Once per-frame always.
* **Accumulation** - triggers on the accumulative count of unique instances across frames, Once per-frame always.
* **Intersection** - triggers on the intersection of two objects detected within a frame. Once per-intersecting-pair.
* **Count** - triggers when the count of objects within a frame is within a specified range.. Once per-frame at most.
* **New Low** - triggers when the count of objects within a frame reaches a new low count.
* **New High** trigger when the count of objects within a frame reaches a new high count.
* **Smallest** - triggers on the smallest object by area if one or more objects are detected. Once per-frame at most.
* **Largest** - triggers on the largest object by area if one or more objects are detected. Once per-frame at most.
* **Earliest** - triggers on the object that came into view the earliest (most persistent). Once per-frame at most.
* **Latest** - triggers on the object that came into view the latest (least persistent). Once per-frame at most.
* **Custom** - allows the client to provide a callback function that implements a custom "Check for Occurrence".

Triggers have optional, settable criteria and filters: 
* **Class Id** - filters on a specified GIE Class Id when checking detected objects. Use `DSL_ODE_ANY_CLASS` to disable the filter
* **Source** - filters on a unique Source name. Use `DSL_ODE_ANY_SOURCE` or NULL to disabled the filter
* **Dimensions** - filters on an object's dimensions ensuring both width and height minimums and/or maximums are met. 
* **Confidence** - filters on an object's GIE confidence requiring a minimum value.
* **Inference Component** - filters on inference metadata from a specific inference component.
* **Inference Done** - filtering on the Object's inference-done flag.
* **In-frame Areas** - filters on specific areas (see ODE Areas below) within the frame, with both areas of inclusion and exclusion supported.

Refer to the [ODE Trigger API Reference](/docs/api-ode-trigger.md) for more information.

### ODE Actions
**ODE Actions** handle the occurrence of Object Detection Events each with a specific action under the categories below. 
* **Actions on Buffers** - Capture Frames and Objects to JPEG images and save to file.
* **Actions on Metadata** - Format Object Labels & Bounding Boxes, Fill-Frames and Objects with a color, add Text & Shapes to a Frame.
* **Actions on ODE Data** - Monitor, Print, Log, and Display ODE occurrence data on screen.
* **Actions on Recordings** - Start a new recording session for a Record Tap or Sink 
* **Actions on Pipelines** - Pause Pipeline, Add/Remove Source, Add/Remove Sink, Disable ODE Handler
* **Actions on Triggers** - Disable/Enable/Reset Triggers
* **Actions on Areas** - Add/Remove Areas
* **Actions on Actions** - Disable/Enable Actions

The below screenshot, captured while running the python example [ode_persistence_and_earliest_triggers_custom_labels.py](/examples/python/ode_persistence_and_earliest_triggers_custom_labels.py), shows how ODE Triggers and Actions can be used to update the Frame and Object metadata to display event metrics.

![meta data](/Images/display-action-screenshot.png)

Refer to the [ODE Action API Reference](/docs/api-ode-action.md) for more information.

### ODE Areas
**ODE Areas**, [Lines](/docs/api-display-type.md#dsl_display_type_rgba_line_new) and [Polygons](/docs/api-display-type.md#dsl_display_type_rgba_polygon_new) can be added to any number of Triggers as additional criteria. 

* **Line Areas** - criteria is met when a specific edge of an object's bounding box - left, right, top, bottom - intersects with the Line Area
* **Polygon Areas** - criteria is met when a specific point of an object's bounding box - south, south-west, west, north-west, north, etc - is within the Polygon 

The following image was produced using: 
* Occurrence Trigger filtering on Any Class Id to hide/exclude the Object Text and Bounding Boxes.
* Occurrence Trigger filtering on Person Class Id as criteria, using:
  * Polygon Area of Inclussion as additional criteria,
  * Fill Object Action to fill the object's bounding-box with an opague RGBA color on criteria met

![Polygon Area](/Images/polygon-screenshot.png)

The above is produced with the following example

```python
# example assumes that all return values are checked before proceeding

# Create a Format Label Action to remove the Object Label from view
# Note: the label can be disabled with the OSD API as well. 
retval = dsl_ode_action_format_label_new('remove-label', 
    font=None, has_bg_color=False, bg_color=None)
            
# Create a Format Bounding Box Action to remove the box border from view
retval = dsl_ode_action_format_bbox_new('remove-border', border_width=0,
    border_color=None, has_bg_color=False, bg_color=None)

# Create an Any-Class Occurrence Trigger for our Hide Action
retval = dsl_ode_trigger_occurrence_new('every-occurrence-trigger', source='uri-source-1',
    class_id=DSL_ODE_ANY_CLASS, limit=DSL_ODE_TRIGGER_LIMIT_NONE)
retval = dsl_ode_trigger_action_add_many('every-occurrence-trigger', 
    actions=['remove-label', 'remove-border', None])

# Create the opaque red RGBA Color and "fill-object" Action to fill the bounding box
retval = dsl_display_type_rgba_color_new('opaque-red', red=1.0, green=0.0, blue=0.0, alpha=0.3)
retval = dsl_ode_action_fill_object_new('fill-action', color='opaque-red')

# create a list of X,Y coordinates defining the points of the Polygon.
# Polygons can have a minimum of 3, maximum of 16 points (sides)
coordinates = [dsl_coordinate(365,600), dsl_coordinate(580,620), 
    dsl_coordinate(600, 770), dsl_coordinate(180,750)]

# Create the Polygon display type using the same red RGBA Color
retval = dsl_display_type_rgba_polygon_new('polygon1', 
    coordinates=coordinates, num_coordinates=len(coordinates), border_width=4, color='opaque-red')
    
# New "Area of Inclusion" to use as criteria for ODE Occurrence using the Polygon object
# Test point DSL_BBOX_POINT_SOUTH = center of rectangle bottom edge must be within Pologon.
retval = dsl_ode_area_inclusion_new('polygon-area', polygon='polygon1', 
    show=True, bbox_test_point=DSL_BBOX_POINT_SOUTH)    

# New Occurrence Trigger, filtering on PERSON class_id, and with no limit on the number of occurrences
retval = dsl_ode_trigger_occurrence_new('person-occurrence-trigger', source="East Cam 1",
    class_id=PGIE_CLASS_ID_PERSON, limit=DSL_ODE_TRIGGER_LIMIT_NONE)

# Add the Polygon Area and Fill Object Action to the new Occurrence Trigger
retval = dsl_ode_trigger_area_add('person-occurrence-trigger', area='polygon-area')
retval = dsl_ode_trigger_action_add('person-occurrence-trigger', action='fill-action')
            
# New ODE Handler to handle all ODE Triggers with their Areas and Actions    
retval = dsl_pph_ode_new('ode-handler')
retval = dsl_pph_ode_trigger_add_many('ode-handler', 
    triggers=['any-occurrence-trigger', 'person-occurrence-trigger', None])
    
#  Then add the handler to the sink (input) pad of a Tiler.
retval = dsl_tiler_pph_add('tiler', 'ode-handler', DSL_PAD_SINK)
```

The complete example script under [/examples/python](/examples/python) can be viewed [here](/examples/python/ode_occurrence_polygon_area_inclussion_exclusion.py)

Refer to the [ODE Area API Reference](/docs/api-ode-area.md) for more information.

---

### ODE Line Crossing Analytics
The Python example [ode_line_cross_object_capture_overlay_image.py](/examples/python/ode_line_cross_object_capture_overlay_image.py) demonstrates how an [ODE Cross Trigger](/docs/api-ode-trigger.md#dsl_ode_trigger_cross_new) with an [ODE Line Area](/docs/api-ode-area.md#dsl_ode_area_line_new()) and [ODE Accumulator](/docs/api-ode-accumulator.md) can be used to perform line-crossing analytics. 

**Important Note:** [Multi-Line Areas](/docs/api-ode-area.md#ode_area_line_multi_new) and [Polygon Inclusion Areas](/docs/api-ode-area.md#dsl_ode_area_inclusion_new) can be used as well. 

A Cross Trigger maintains a vector of historical bounding-box coordinates for each object tracked by its unique tracking id. The Trigger, using the bounding box history and the Area's defined Test Point (SOUTH, WEST, etc.), generates an Object Trace - vector of x,y coordinates - to test for line cross with the Area's line. 

There are two methods of testing and displaying the Object Trace:
1. using `ALL` points in the vector to generate the trace to test for line-cross. 
2. using just the `END` points (earlier and latest) to generate the trace to test for line-cross. 

Note that using `ALL` points will add overhead to the processing of each detected object and considerable allocation/deallocation overhead and memory usage if displayed. `End` points are used in the example which is why the traces appear as straight lines. The camera angle and proximity to the objects should be considered when choosing which method to use as well.

An [ODE Accumulator](/docs/api-ode-accumulator.md) with an [ODE Display Action](/docs/api-ode-action.md#dsl_ode_action_display_new) is added to the Cross Trigger to accumulate and display the number of line-crossing occurrences in the IN and OUT directions as shown in the image below.

The example creates an [ODE Print Action](/docs/api-ode-action.md#dsl_ode_action_print_new) and an [ODE Capture Object Action](/docs/api-ode-action.md#dsl_ode_action_capture_object_new) with an [Image Render Player](/docs/api-player.md#dsl_player_render_image_new) to print each line-crossing occurrence to the console and to capture the object to an image file and display the image as an overlay, repectively.  

**Important Note:** A reminder that other actions such as the [ODE File Action](/docs/api-ode-action.md#dsl_ode_action_file_new), [ODE Email Action](/docs/api-ode-action.md#dsl_ode_action_email_new), and the [ODE Add IOT Message Action](/docs/api-ode-action.md#dsl_ode_action_message_meta_add_new) can be leveraged with the ODE Cross Trigger as well.

![](/Images/line-cross-capture-overlay-object-image.png)

The Line Area is created with an [RGBA Line](/docs/api-display-type.md#dsl_display_type_rgba_line_new) with the line's width used as line-cross hysteresis.

```Python
# Create the RGBA Line Display Type with a width of 6 pixels for hysteresis
retval = dsl_display_type_rgba_line_new('line',
    x1=260, y1=680, x2=600, y2=660, width=6, color='opaque-red')

# Create the ODE line area to use as criteria for ODE occurrence
# Use the center point on the bounding box's bottom edge for testing
retval = dsl_ode_area_line_new('line-area', line='line',
    show=True, bbox_test_point=DSL_BBOX_POINT_SOUTH)    
```

The ODE Cross Trigger is created with a `min_frame_count` and `max_trace_points` as criteria for a line-cross occurrence.
```Python
# New Cross Trigger filtering on PERSON class_id to track and trigger on
# objects that fully cross the line. The person must be tracked for a minimum
# of 5 frames prior to crossing the line to trigger an ODE occurrence.
# The trigger can save/use up to a maximum of 200 frames of history to create
# the object's historical trace to test for line-crossing. retval = dsl_ode_trigger_cross_new('person-crossing-line',
retval = dsl_ode_trigger_cross_new('person-crossing-line',
    source = DSL_ODE_ANY_SOURCE,
    class_id = PGIE_CLASS_ID_PERSON,
    limit = DSL_ODE_TRIGGER_LIMIT_NONE,
    min_frame_count = 5,
    max_trace_points = 200,
    test_method = DSL_OBJECT_TRACE_TEST_METHOD_END_POINTS)

# Add the line area to the New Cross Trigger
 retval = dsl_ode_trigger_area_add('person-crossing-line', area='line-area')  

```
Each Tracked Object's historical trace can be added as display metadata for a downstream On-Screen Display to display.
```Python
# New RGBA Random Color to use for Object Trace and BBox    
retval = dsl_display_type_rgba_color_random_new('random-color',
    hue = DSL_COLOR_HUE_RANDOM,
    luminosity = DSL_COLOR_LUMINOSITY_RANDOM,
    alpha = 1.0,
    seed = 0)

# Set the Cross Trigger's view settings to enable display of the Object Trace
retval = dsl_ode_trigger_cross_view_settings_set('person-crossing-line',
    enabled=True, color='random-color', line_width=4)
```
The example creates a new ODE Display Action and adds it to a new ODE Accumulator. It then adds the Accumulator to the Trigger to complete the setup.
```Python
# Create a new Display Action used to display the Accumulated ODE Occurrences.
# Format the display string using the occurrences in and out tokens.
retval = dsl_ode_action_display_new('display-cross-metrics-action',
    format_string =
        "In : %" + str(DSL_METRIC_OBJECT_OCCURRENCES_DIRECTION_IN) +
        ", Out : %" + str(DSL_METRIC_OBJECT_OCCURRENCES_DIRECTION_OUT),  
    offset_x = 1200,
    offset_y = 100,
    font = 'arial-16-yellow',
    has_bg_color = False,
    bg_color = None)
           
# Create an ODE Accumulator to add to the Cross Trigger. The Accumulator
# will work with the Trigger to accumulate the IN and OUT occurrence metrics.
retval = dsl_ode_accumulator_new('cross-accumulator')
       
# Add the Display Action to the Accumulator. The Accumulator will call on
# the Display Action to display the new accumulative metrics after each frame.
retval = dsl_ode_accumulator_action_add('cross-accumulator',
    'display-cross-metrics-action')
       
# Add the Accumulator to the Line Cross Trigger.
retval = dsl_ode_trigger_accumulator_add('person-crossing-line',
    'cross-accumulator')
   
```
See the [complete example](/examples/python/ode_line_cross_object_capture_overlay_image.py) and refer to the [ODE Trigger API Reference](/docs/api-ode-accumulator.md), [ODE Action API Reference](/docs/api-ode-action.md), [ODE Area API Reference](/docs/api-ode-area.md), and [ODE Accumulator API Reference](/docs/api-ode-accumulator.md) sections for more information.

---

### ODE Heat Mapping
[ODE Heat Mappers](/docs/api-ode-heat-mapper.md#dsl_ode_heat_mapper_new) are added to [ODE Triggers](/docs/api-md) to accumulate, map, and display the ODE Occurrences over time. The source frame is partitioned into a configurable number of rows and columns, with each rectangle colored with a specific RGBA color value based on the number of occurrences that were detected within corresponding area within the source frame.

The client application can `get`, `print`, `log`, `file`, and `clear` the metric occurrence data at any time.

The below image was created with the [ode_occurrence_trigger_with_heat_mapper.py](https://github.com/prominenceai/deepstream-services-library/blob/complete_examples/examples/python/ode_occurrence_trigger_with_heat_mapper.py) Python example.

See the [ODE Heat-Mapper API Reference](/docs/api-ode-heat-mapper.md) for more information.

![](/Images/spectral-person-heat-map.png)

--- 

## Smart Recording
As mentioned above, there are two components that provide cached-video-recording:
1. Record Tap - that taps into the pre-decoded stream to record the original video - added to a RTSP Source component directly.
2. Record Sink - that encodes the decoded, inferred-on, and optionally annotated stream - added to a Pipeline or Branch downstream of a Tiler or Demuxer.

Both recording components create a fixed size cache to buffer the last N seconds of the encoded video. Services are provided to start recording at a `start` point within the current cache specified in seconds before the current time. The `duration` to record is specified in units of seconds, though the recording can be stopped at any time.  A client callback is used to notify the application when recording is complete.

"One-time" [ODE Triggers](/docs/api-ode-trigger.md) are defined to trigger on specific occurrences of Object Detection Events, such as a person entering a predefined [ODE Area](/docs/api-ode-area.md). Start-Record [ODE Actions]() are used to start the recording on ODE occurrence. Each "one-time" trigger can be reset in the record-complete callback function added to the Record-Tap or Sink. 

The follow example illustrates a Pipeline with multiple sources, each with a Record Tap and corresponding Occurrence Trigger with a Start-Record Action
![Record Tap](/Images/tap-record.png)

#### Using Python to implement the above.

Each Camera requires: 
* RTSP Source Component - to decode the streaming source to raw video and audio
* Record Tap - with cache to capture pre-event video
* Occurrence Trigger - to trigger ODE occurrence on detection of an object satisfying all criteria
* Custom Action - for notification of ODE occurrence.
* Start Record Action - to start the recording

```Python
## 	
# Defines a class of all component names associated with a single RTSP Source. 
# Objects of this class will be used as "client_data" for all callback notifications.	
# defines a class of all component names associated with a single RTSP Source. 	
# The names are derived from the unique Source name	
##	
class ComponentNames:	
    def __init__(self, source):	
        self.source = source	
        self.instance_trigger = source + '-instance-trigger'
        self.record_tap = source + '-record-tap'	
        self.start_record = source + '-start-record'
        
##
# Client listner function callad at the start and end of a recording session
##
def OnRecordingEvent(session_info_ptr, client_data):

    if client_data == None:
        return None

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value

    session_info = session_info_ptr.contents

    print('session_id: ', session_info.session_id)
    
    # If we're starting a new recording for this source
    if session_info.recording_event == DSL_RECORDING_EVENT_START:
        print('event:      ', 'DSL_RECORDING_EVENT_START')

        # in this example we will call on the Tiler to show the source that started recording.	
        retval = dsl_tiler_source_show_set('tiler', source=components.source, 
            timeout=0, has_precedence=True)	
        if (retval != DSL_RETURN_SUCCESS):
            print('Tiler show single source failed with error: ', dsl_return_value_to_string(retval))
        
    # Else, the recording session has ended for this source
    else:
        print('event:      ', 'DSL_RECORDING_EVENT_END')
        print('filename:   ', session_info.filename)
        print('dirpath:    ', session_info.dirpath)
        print('duration:   ', session_info.duration)
        print('container:  ', session_info.container_type)
        print('width:      ', session_info.width)
        print('height:     ', session_info.height)

        # if we're showing the source that started this recording
        # we can set the tiler back to showing all tiles, otherwise
        # another source has started recording and taken precendence
        retval, current_source, timeout  = dsl_tiler_source_show_get('tiler')
        if reval == DSL_RETURN_SUCCESS and current_source == components.source:
            dsl_tiler_source_show_all('tiler')

        # re-enable the one-shot trigger for the next "New Instance" of a person
        retval = dsl_ode_trigger_reset(components.instance_trigger)	
        if (retval != DSL_RETURN_SUCCESS):
            print('Failed to reset instance trigger with error:', dsl_return_value_to_string(retval))

    return None
```    
The below function creates all "1-per-source" components for a given source-name and RTSP URI.
The new Source component is added to the named Pipeline and the Trigger is added to [ODE Pad Probe Handler](/docs/api-pph.md)

```Python
##
# Function to create all "1-per-source" components, and add them to the Pipeline
# pipeline - unique name of the Pipeline to add the Source components to
# source - unique name for the RTSP Source to create
# uri - unique uri for the new RTSP Source
# ode_handler - Object Detection Event (ODE) handler to add the new Trigger and Actions to
##
def CreatePerSourceComponents(pipeline, source, rtsp_uri, ode_handler):
   
    global duration
    
    # New Component names based on unique source name
    components = ComponentNames(source)
    
    # For each camera, create a new RTSP Source for the specific RTSP URI
    retval = dsl_source_rtsp_new(source, 
        uri = rtsp_uri, 
        protocol = DSL_RTP_ALL, 
        cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 
        intra_decode = False, 
        drop_frame_interval = 0, 
        latency=100)
    if (retval != DSL_RETURN_SUCCESS):
        return retval

    # New record tap created with our common RecordComplete callback function defined above
    retval = dsl_tap_record_new(components.record_tap, 
        outdir = './recordings/', 
        container = DSL_CONTAINER_MKV, 
        client_listener = RecordComplete)
    if (retval != DSL_RETURN_SUCCESS):
        return retval

    # Add the new Tap to the Source directly
    retval = dsl_source_rtsp_tap_add(source, tap=components.record_tap)
    if (retval != DSL_RETURN_SUCCESS):
        return retval

    # Next, create the Person Instance Trigger. We will reset the trigger on DSL_RECORDING_EVENT_END
   	# ... see the OnRecordingEvent() client callback function above
    retval = dsl_ode_trigger_instance_new(components.instance_trigger, 	
        source=source, class_id=PGIE_CLASS_ID_PERSON, limit=1)	
    if (retval != DSL_RETURN_SUCCESS):	
        return retval	

    # Create a new Action to start the record session for this Source, with the component names as client data	
    retval = dsl_ode_action_tap_record_start_new(components.start_record, 	
        record_tap=components.record_tap, start=15, duration=360, client_data=components)	
    if (retval != DSL_RETURN_SUCCESS):	
        return retval	

    # Add the Actions to the trigger for this source. 	
    retval = dsl_ode_trigger_action_add_many(components.instance_trigger, 	
        actions=[components.start_record, None])	
    if (retval != DSL_RETURN_SUCCESS):	
        return retval	

    # Add the new Source with its Record-Tap to the Pipeline	
    retval = dsl_pipeline_component_add(pipeline, source)	
    if (retval != DSL_RETURN_SUCCESS):	
        return retval	
        
    # Add the new Trigger to the ODE Pad Probe Handler
    return dsl_pph_ode_trigger_add(ode_handler, components.occurrence_trigger)

```

The main code to create all other components and assemble the Pipeline can be written as:

```Python

while True:
    # Create the Primary GIE, Tracker, Multi-Source Tiler, On-Screen Display and X11/EGL Window Sink
    retval = dsl_gie_primary_new('pgie', path_to_engine_file, path_to_config_file, 0)
    if (retval != DSL_RETURN_SUCCESS):
        break

    retval = dsl_tracker_ktl_new('tracker', max_width=480, max_height=270)
    if (retval != DSL_RETURN_SUCCESS):
        break

    retval = dsl_tiler_new('tiler', width=1280, height=720)
    if (retval != DSL_RETURN_SUCCESS):
        break

    retval = dsl_osd_new('my-osd', text_enabled=True, clock_enabled=True,
        bbox_enabled=True, mask_enabled=False)
    if (retval != DSL_RETURN_SUCCESS):
        break

    retval = dsl_sink_window_new('window-sink', 0, 0, width=1280, height=720)
    if (retval != DSL_RETURN_SUCCESS):
        break

    retval = dsl_sink_rtsp_new('rtsp-sink', 
        host='my-jetson.local', udp_port=5400, rtsp_port=8554, codec=DSL_CODEC_H265, bitrate=200000, interval=0)
    if (retval != DSL_RETURN_SUCCESS):
        break
    
    # Create a Pipeline and add the new components.
    retval = dsl_pipeline_new_component_add_many('pipeline', 
        components=['pgie', 'tracker', 'tiler', 'osd', 'window-sink', 'rtsp-sink', None]) 
    if (retval != DSL_RETURN_SUCCESS):
        break
   
    # Object Detection Event (ODE) Pad Probe Handler (PPH) to manage our ODE Triggers with their ODE Actions
    retval = dsl_pph_ode_new('ode-handler')
    if (retval != DSL_RETURN_SUCCESS):
        break
 
    # Add the ODE Handler to the Sink (input) pad of the Tiler - before the batched frames are combined/tiled
    retval = dsl_tiler_pph_add('tiler', 'ode-handler', DSL_PAD_SINK)
    if (retval != DSL_RETURN_SUCCESS):
        break

    # For each of our four sources, call the funtion to create the source-specific components.
    retval = CreatePerSourceComponents('pipeline', 'src-0', src_url_0, 'ode-handler')
    if (retval != DSL_RETURN_SUCCESS):
        break
    retval = CreatePerSourceComponents('pipeline', 'src-1', src_url_1, 'ode-handler')
    if (retval != DSL_RETURN_SUCCESS):
        break
    retval = CreatePerSourceComponents('pipeline', 'src-2', src_url_2, 'ode-handler')
    if (retval != DSL_RETURN_SUCCESS):
        break
    retval = CreatePerSourceComponents('pipeline', 'src-3', src_url_3, 'ode-handler')
    if (retval != DSL_RETURN_SUCCESS):
        break
    
    # Pipeline has been successfully created, ok to play
    retval = dsl_pipeline_play('my-pipeline')
    if (retval != DSL_RETURN_SUCCESS):
        break

    # join the main loop until stopped. 
    dsl_main_loop_run()
    break

# Print out the final result
print(dsl_return_value_to_string(retval))

# free up all resources
dsl_delete-all()

```

Please refere to [ode_occurrence_4rtsp_start_record_tap_action.py](/examples/python/ode_occurrence_4rtsp_start_record_tap_action.py) for the complete example.

---
## RTSP Stream Connection Management
RTSP Source Components have "built-in" stream connection management for detecting and resolving stream disconnections.   

When creating a RTSP Source, the client application can specify a `next-buffer-timeout` defined as the maximum time to wait in seconds for each new frame buffer before the Source's Stream Manager -- determining that the connection has been lost -- resets the Source and tries to reconnect.

The Stream manager uses two client settable parameters to control the reconnection behavior. 

1. `sleep` - the time to sleep between failed connection atempts, in units of seconds. 
2. `timeout` - the maximum time to wait for an asynchronous state change to complete before determining that reconnection has failed - also in seconds. 

Note: Setting the reconnection timeout to a value less than the device's socket timeout can result in the Stream failing to connect. Both parameters are set to defaults when the Source is created, defined in `dslapi.h` as:
```C
#define DSL_RTSP_RECONNECTION_SLEEP_S    4
#define DSL_RTSP_RECONNECTION_TIMEOUT_S  30
```

The client can register a `state-change-listener` callback function to be notified on every change-of-state, to monitor the connection process and update the reconnection parameters when needed.

Expanding on the [Smart Recording](#smart-recording) example above,

```Python
##
# Function to be called by all Source components on every change of state
# old_state - the previous state of the source prior to change
# new_state - the new state of source after the state change
# client_data - components object containing the name of the Source
##
def SourceStateChangeListener(old_state, new_state, client_data):

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value

    print('RTSP Source ', components.source, 'change-of-state: previous =',
        dsl_state_value_to_string(old_state), '- new =', dsl_state_value_to_string(new_state))
    
    # A change of state to NULL occurs on every disconnection and after each failed retry.
    # A change of state to PLAYING occurs on every successful connection.
    if (new_state == DSL_STATE_NULL or new_state == DSL_STATE_PLAYING):
    
        # Query the Source for it's current statistics and reconnection parameters
        retval, data = dsl_source_rtsp_connection_data_get(components.source)
        
        print('Connection data for source:', components.source)
        print('  is connected:     ', data.is_connected)
        print('  first connected:  ', time.ctime(data.first_connected))
        print('  last connected:   ', time.ctime(data.last_connected))
        print('  last disconnected:', time.ctime(data.last_disconnected))
        print('  total count:      ', data.count)
        print('  in is reconnect:  ', data.is_in_reconnect)
        print('  retries:          ', data.retries)
        print('  sleep time:       ', data.sleep,'seconds')
        print('  timeout:          ', data.timeout, 'seconds')

        if (new_state == DSL_STATE_PLAYING):
            print("setting the time to sleep between re-connection retries to 4 seconds for quick recovery")
            dsl_source_rtsp_reconnection_params_set(components.source, sleep=4, timeout=30)
            
        # If we're in a reconnection cycle, check if the number of quick recovery attempts has
        # been reached. (20 * 4 =~ 80 seconds), before backing off on the time between retries 
        elif (data.is_in_reconnect and data.retries == 20):
            print("extending the time to sleep between re-connection retries to 20 seconds")
            dsl_source_rtsp_reconnection_params_set(components.source, sleep=20, timeout=30)
```
When creating each RTSP Source component, set the Source's next-buffer-timeout, and then add the common `SourceStateChangeListener` callback to the Source with the `components` object as `client_data` to be returned on change-of-state. 

```Python
##
# Function to create all "1-per-source" components, and add them to the Pipeline
# pipeline - unique name of the Pipeline to add the Source components to
# source - unique name for the RTSP Source to create
# uri - unique uri for the new RTSP Source
# ode_handler - Object Detection Event (ODE) handler to add the new Trigger and Actions to
##
def CreatePerSourceComponents(pipeline, source, rtsp_uri, ode_handler):
   
    # New Component names based on unique source name
    components = ComponentNames(source)
    
    # For each camera, create a new RTSP Source for the specific RTSP URI
    retval = dsl_source_rtsp_new(source, 
        uri = rtsp_uri, 
        protocol = DSL_RTP_ALL, 
        cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 
        intra_decode = False, 
        drop_frame_interval = 0, 
        latency=100,
        timeout=3)
    if (retval != DSL_RETURN_SUCCESS):
        return retval
        
    # Add our state change listener to the new source, with the component names as client data
    retval = dsl_source_rtsp_state_change_listener_add(source, 
        client_listener=source_state_change_listener,
        client_data=components)
    if (retval != DSL_RETURN_SUCCESS):
        return retval
        
    # ---- create the remaining components
    
```
Refer to the [Source API](/docs/api-source.md) documentation for more information. The script [ode_occurrence_4rtsp_start_record_tap_action.py](/examples/python/ode_occurrence_4rtsp_start_record_tap_action.py) provides a complete example.

---

## X11 Window Services
DSL provides X11 Window Services for Pipelines that use a Window Sink. An Application can create an XWindow - using GTK+ for example - and pass the window handle to the Pipeline prior to playing, or let the Pipeline create the XWindow to use by default. 

The client application can register callback functions to handle window events -- `ButtonPress`, `KeyRelease`, and `WindowDelete` -- caused by user interaction. 

Expanding on the [Smart Recording](#smart-recording) example above, with its four Sources and Tiled Display, the following Client callback functions provide examples of how user input can be used to control the application. 

The first callback allows the user to `select` a single source stream within the tiled view based on the positional coordinates of a `ButtonPress`. The selected stream will be shown for a specified time period, or until the window is clicked on again. A timeout value of 0 will disable the timer.

``` Python
## 
# Function to be called on XWindow Button Press event
# button - id of the button pressed, one of Button1..Button5
# x_pos - x positional coordinate relative to the windows top/left corner
# y_pos - y positional coordinate relative to the windows top/left corner
# client_data - unused. 
## 
def XWindowButtonEventHandler(button, x_pos, y_pos, client_data):
    print('button = ', button, ' pressed at x = ', x_pos, ' y = ', y_pos)
    
    # time to show the single source before returning to view all. A timeout value of 0
    # will disable the Tiler's timer and show the single source until called on again.
    global SHOW_SOURCE_TIMEOUT

    if (button == Button1):
        # get the current XWindow dimensions as the User may have resized it. 
        retval, width, height = dsl_pipeline_xwindow_dimensions_get('pipeline')
        
        # call the Tiler to show the source based on the x and y button cooridantes relative
        # to the current window dimensions obtained from the XWindow.
        dsl_tiler_source_show_select('tiler', x_pos, y_pos, width, height, timeout=SHOW_SOURCE_TIMEOUT)
```
The second callback, called on KeyRelease, allows the user to
1. show a single source, or all
2. cycle through all sources on a time interval, 
3. quit the application. 

```Python
## 
# Function to be called on XWindow KeyRelease event
# key_string - the ASCI key string value of the key pressed and released
# client_data
## 
def XWindowKeyReleaseEventHandler(key_string, client_data):
    print('key released = ', key_string)
    
    global SHOW_SOURCE_TIMEOUT
        
    # if one of the unique soure Ids, show source
    elif key_string >= '0' and key_string <= '3':
        retval, source = dsl_source_name_get(int(key_string))
        if retval == DSL_RETURN_SUCCESS:
            dsl_tiler_source_show_set('tiler', source=source, timeout=SHOW_SOURCE_TIMEOUT, has_precedence=True)
            
    # C = cycle All sources
    elif key_string.upper() == 'C':
        dsl_tiler_source_show_cycle('tiler', timeout=SHOW_SOURCE_TIMEOUT)

    # A = show All sources
    elif key_string.upper() == 'A':
        dsl_tiler_source_show_all('tiler')

    # Q or Esc = quit application
    if key_string.upper() == 'Q' or key_string == '':
        dsl_main_loop_quit()
```
The third callback is called when the user closes/deletes the XWindow allowing the application to exit from the main-loop and delete all resources

```Python
# Function to be called on XWindow Delete event
def XWindowDeleteEventHandler(client_data):
    print('delete window event')
    dsl_main_loop_quit()

```
The callback functions are added to the Pipeline after creation. The XWindow, in this example, is set into `full-screen` mode before the Pipeline is played.

```Python
while True:

    # New Pipeline
    retval = dsl_pipeline_new('pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break

    retval = dsl_sink_window_new('window-sink', 0, 0, width=1280, height=720)
    if (retval != DSL_RETURN_SUCCESS):
        break
    # Add the XWindow event handler functions defined above
    retval = dsl_pipeline_xwindow_button_event_handler_add('pipeline', XWindowButtonEventHandler, None)
    if retval != DSL_RETURN_SUCCESS:
        break
    retval = dsl_pipeline_xwindow_key_event_handler_add('pipeline', XWindowKeyReleaseEventHandler, None)
    if retval != DSL_RETURN_SUCCESS:
        break
    retval = dsl_pipeline_xwindow_delete_event_handler_add('pipeline', XWindowDeleteEventHandler, None)
    if retval != DSL_RETURN_SUCCESS:
        break

    # Set the XWindow into 'full-screen' mode for a kiosk look and feel.         
    retval = dsl_pipeline_xwindow_fullscreen_enabled_set('pipeline', enabled=True)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    # Create all other required components and add them to the Pipeline (see some examples above)
    # ...
 
    retval = dsl_pipeline_play('pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break
 
    # Start/Join with main loop until released - blocking call
    dsl_main_loop_run()
    retval = DSL_RETURN_SUCCESS
    break

#print out the final result
print(dsl_return_value_to_string(retval))

# clean up all resources
dsl_delete_all()
```

<br>

---

## Player Services
Players are specialized Pipelines that simplify the processes of:
* testing/confirming camera connections and URIs
* rendering captured images and video recordings.

The following python3 example shows how to add a **Video Render Player** to a **Smart Record Sink" which will automatically playback each video on recording complete. See the [Player API Documentation](/docs/api-player.md) for more information.

```Python
# New Record-Sink that will buffer encoded video while waiting for the ODE trigger/action, 
retval = dsl_sink_record_new('record-sink', outdir="./", codec=DSL_CODEC_H265, container=DSL_CONTAINER_MKV, 
    bitrate=12000000, interval=0, client_listener=recording_event_listener)
if retval != DSL_RETURN_SUCCESS:
    break

# Create the Video Render Player with a NULL file_path to be updated by the Smart Record Sink
retval = dsl_player_render_video_new(
    name = 'video-player',
    file_path = None,
    render_type = DSL_RENDER_TYPE_OVERLAY,
    offset_x = 500, 
    offset_y = 20, 
    zoom = 50,
    repeat_enabled = False)
if retval != DSL_RETURN_SUCCESS:
    break

# Add the Player to the Record Sink. The Sink will add/queue
# the file_path to each video recording created. 
retval = dsl_sink_record_video_player_add('record-sink', 
    player='video-player')
if retval != DSL_RETURN_SUCCESS:
    break
```

See the script [ode_occurrence_object_capture_overlay_image.py](/examples/python/ode_occurrence_object_capture_overlay_image.py) for the complete example.

<br>

---

## SMTP Services
Secure outgoing SMTP email services allow clients to provide server info, credentials and header data (From, To, Cc, Subject, etc.) - settings required for an [ODE Email Action](/docs/api-ode-action.md#dsl_ode_action_email_new) to send email notifications on an [Object Detection Event (ODE) Occurence](#object-detection-event-pad-probe-handler).

Message content is sent out using multipart mime-type. Adding attachments, including captured images, will be supported in a future release.

Refere to the [SMTP Services](/docs/api-smtp.md) for more information.

See the example script [ode_occurrence_uri_send_smtp_mail.py](/examples/python/ode_occurrence_uri_send_smtp_mail.py) for additional reference.

---

## DSL Initialization
The library is automatically initialized on **any** first call to DSL. There is no explicit init or deint service. DSL will initialize GStreamer at this time unless the calling application has already done so. 

<br>

## DSL Delete All
All DSL and GStreammer resources should be deleted on code exit by calling DSL to delete all.
```Python
dsl_delete_all()
```

<br>

## Main Loop Context
After creating all components, adding them to a Pipeline, and setting the Pipeline's state to Playing, the Application must call `dsl_main_loop_run()`. The service creates a mainloop that runs/iterates the default GLib main context to check if anything the Pipeline is watching for has happened. The main loop will be run until another thread -- typically a "client callback function" called from the Pipeline's context -- calls `dsl_main_loop_quit()`

<br>

## Service Return Codes
Most DSL services return values of type `DslReturnType`, return codes of `0` indicating success and `non-0` values indicating failure. All possible return codes are defined as symbolic constants in `DslApi.h` When using Python3, DSL provides a convenience service `dsl_return_value_to_string()` to use as there are no "C" equivalent symbolic constants or enum types in Python.  

**Note:** This convenience service is the preferred method as the return code values are subject to change

`DSL_RESULT_SUCCESS` is defined in both `DslApi.h` and `dsl.py`. The non-zero Return Codes are defined in `DslApi.h` only.

**DslApi.h**
```C
#define DSL_RESULT_SUCCESS 0

typedef uint DslReturnType
```
**Python Script**
```Python
from dsl import *

retVal = dsl_sink_rtsp_new('rtsp-sink', host_uri, 5400, 8554, DSL_CODEC_H264, 4000000, 0)

if dsl_return_value_to_string(retval) eq 'DSL_RESULT_SINK_NAME_NOT_UNIQUE':
    # handle error
```

<br>

## Docker Support
The [deepstream-services-library-docker](https://github.com/prominenceai/deepstream-services-library-docker) repo contain a `Dockerfile`, utility scripts, and instructions to create and run a DSL-DeepStream container, built with the [nvcr.io/nvidia/deepstream-l4t:6.0-triton](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html#id2) base image (Jetson).

## Getting Started
* [Installing Dependencies](/docs/installing-dependencies.md)
* [Building and Importing DSL](/docs/building-dsl.md)

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
* [Demuxer and Splitter Tees](/docs/api-tee)
* [On-Screen Display](/docs/api-osd.md)
* [Sink](docs/api-sink.md)
* [Pad Probe Handler](/docs/api-pph.md)
* [ODE Trigger](/docs/api-ode-trigger.md)
* [ODE Accumulator](/docs/api-ode-accumulator.md)
* [ODE Action ](/docs/api-ode-action.md)
* [ODE Area](/docs/api-ode-area.md)
* [ODE Heat-Mapper](/docs/api-ode-heat-mapper.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](/docs/api-branch.md)
* [Component](/docs/api-component.md)
* [Mailer](/docs/api-mailer.md)
* [WebSocket Server](/docs/api-ws-server.md)
* [Message Broker](/docs/api-msg-broker.md)
* [Info API](/docs/api-info.md)

--- 
* <b id="f1">1</b> Quote from GStreamer documentation [here](https://gstreamer.freedesktop.org/documentation/?gi-language=c). [↩](#a1)
* <b id="f2">2</b> Quote from NVIDIA's Developer Site [here](https://developer.nvidia.com/deepstream-sdk). [↩](#a2)
