# DSL Overview
### Overview Contents
* [Introduction](#introduction)
* [Pipeline Components](#pipeline-components)
  * [Streaming Sources](#streaming-sources)
  * [Primary and Secondary Inference Engines](#primary-and-secondary-inference-engines)
  * [Multi-Object Trackers](#multi-object-trackers)
  * [On-Screen Display](#on-screen-display)
  * [Multi-Source Tiler](#multi-source-tiler)
  * [Rendering and Encoding Sinks](#rendering-and-encoding-sinks)
  * [Tees and Branches](#tees-and-branches)
  * [Pad Probe Handlers](#pad-probe-handlers)
    * [Pipeline Meter](#pipeline-meter-pad-probe-handler)
    * [Object Detection Event Handler](#object-detection-event-pad-probe-handler)
    * [Custom Handler](#custom-pad-probe-handler)
* [Display Types](#display-types)
* [Smart Recording](#smart-recording)
* [RTSP Stream Connection Management](#rtsp-stream-connection-management)
* [X11 Window Services](#x11-window-services)
* [DSL Initialization](#dsl-initialization)
* [DSL Delete All](#dsl-delete-all)
* [Main Loop Context](#main-loop-context)
* [Service Return Codes](#service-return-codes)
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

# new Primary Inference Engine - path to model engine and config file, interval=0 - infer on every frame
retval += dsl_gie_primary_new('my-pgie', path_to_engine_file, path_to_config_file, interval=0)

# new Multi-Source Tiler with dimensions of width and height 
retval += dsl_tiler_new('my-tiler', width=1280, height=720)

# new On-Screen Display for inference visualization - bounding boxes and labels - with clock enabled
retval += dsl_osd_new('my-osd', clock_enabled=True)

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

There are currently four types of Source components, two live connected Camera Sources:
* Camera Serial Interface (CSI) Source
* Universal Serial Bus (USB) Source

And two decode Sources that support both live and non-live streams.
* Universal Resource Identifier (URI) Source
* Real-time Streaming Protocol (RTSP) Source

All Sources have dimensions, width and height in pixels, and frame-rates expressed as a fractional numerator and denominator.  The decode Source components support multiple codec formats, including H.264, H.265, PNG, and JPEG. A [Dewarper Component](/docs/api-dewarper.md) (not show in the image above) capable of dewarping 360 degree camera streams can be added to both. 

A Pipeline's Stream-Muxer has settable output dimensions with a decoded, batched output stream that is ready to infer on.

A [Record-Tap](#smart-recording) (not show in the image above) can be added to a RTSP Source for cached pre-decode recording, triggered on the occurrence of an [Object Detection Event (ODE)](#object-detection-event-pad-probe-handler).

See the [Source API](/docs/api-source.md) reference section for more information.

## Primary and Secondary Inference Engines
NVIDIA's GStreamer Inference Engines (GIEs), using pre-trained models, classify data to “infer” a result, e.g.: person, dog, car? A Pipeline may have at most one Primary Inference Engine (PGIE) -- with a specified set of classification labels to infer-with -- and multiple Secondary Inference Engines (SGIEs) that can Infer-on the output of either the Primary or other Secondary GIEs. Although optional, a Primary Inference Engine is required when adding a Multi-Object Tracker, Secondary Inference Engines, or On-Screen-Display to a Pipeline.

After creation, GIEs can be updated to: 
* Use a new model-engine, config file and/or inference interval 
* Update the Primary or Secondary GIE to infer on (SGIEs only).

With Primary GIEs, applications can:
* Add/remove [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers with Metadata for each Frame and Detected-Object found within. 
* Enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Primary and Secondary GIE API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove [Pad Probe Handlers](#pad-probe-handlers) to process batched stream buffers -- with Metadata for each Frame and Detected-Object.

Tracker components are optional and a Pipeline, or [Branch](#tees-and-branches) can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more details. See NVIDIA's [Low-Level Tracker Library Comparisons and Tradeoffs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.3.02.html#wwpID0E0Q20HA) for additional information.

## Multi-Source Tiler
To support the dynamic addition and removal of Sources and Sinks, all Source components connect to the Pipeline's internal Stream-Muxer, even when there is only one. The multiplexed stream must either be Tiled **or** Demuxed before reaching an On-Screen Display or Sink component downstream.

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

1. Overlay Render Sink
2. X11/EGL Window Sink
3. File Sink
4. Record Sink
5. RTSP Sink
6. Fake Sink

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
retval = dsl_osd_new('osd', clock_enabled=True)
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
    cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 
    intra_decode = Fale, 
    drop_frame_interval = 0, 
    latency=100)

retval = dsl_source_rtsp_new('src-2', 
    url = rtsp_uri_2, 
    protocol = DSL_RTP_ALL, 
    cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 
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
retval = dsl_osd_new('osd', clock_enabled=True)
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
    cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 
    intra_decode = Fale, 
    drop_frame_interval = 0, 
    latency=100)

retval = dsl_source_rtsp_new('src-2', 
    url = rtsp_uri_2, 
    protocol = DSL_RTP_ALL, 
    cudadec_mem_type = DSL_CUDADEC_MEMTYPE_DEVICE, 
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
The [Object Detection Event (ODE) Pad Probe Handler](/docs/api-pph.md#object-detection-event-ode-pad-probe-handler) manages an ordered collection of **Triggers**, each with an ordered collection of **Actions** and an optional collection of **Areas**. Triggers use settable criteria to process the Frame and Object metadata produced by the Primary and Secondary GIE's looking for specific detection events. When the criteria for the Trigger is met, the Trigger invokes all Actions in its ordered collection. Each unique Area and Action created can be added to multiple Triggers as shown in the diagram below. The ODE Handler has n Triggers, each Trigger has one shared Area and one unique Area, and one shared Action and one unique Action.

![ODE Services](/Images/ode-services.png)

The Handler can be added to the Pipeline before the On-Screen-Display (OSD) component allowing Actions to update the metadata for display. All Triggers can be enabled and re-enabled at runtime, either by an ODE Action, a client callback function, or directly by the application at any time.

Current **ODE Triggers** supported:
* **Always** - triggers on every frame. Once per-frame always.
* **Absence** - triggers on the absence of objects within a frame. Once per-frame at most.
* **Occurrence** - triggers on each object detected within a frame. Once per-object at most.
* **Summation** - triggers on the summation of all objects detected within a frame. Once per-frame always.
* **Intersection** - triggers on the intersection of two objects detected within a frame. Once per-intersecting-pair.
* **Minimum** - triggers when the count of detected objects in a frame fails to meet a specified minimum number. Once per-frame at most.
* **Maximum** - triggers when the count of detected objects in a frame exceeds a specified maximum number. Once per-frame at most.
* **Range** - triggers when the count of detected objects falls within a specified lower and upper range. Once per-frame at most.
* **Smallest** - triggers on the smallest object by area if one or more objects are detected. Once per-frame at most.
* **Largest** - triggers on the largest object by area if one or more objects are detected. Once per-frame at most.
* **Custom** - allows the client to provide a callback function that implements a custom "Check for Occurrence".

Triggers have optional, settable criteria and filters: 
* **Class Id** - filters on a specified GIE Class Id when checking detected objects. Use `DSL_ODE_ANY_CLASS` to disable the filter
* **Source** - filters on a unique Source name. Use `DSL_ODE_ANY_SOURCE` or NULL to disabled the filter
* **Dimensions** - filters on an object's dimensions ensuring both width and height minimums and/or maximums are met. 
* **Confidence** - filters on an object's GIE confidence requiring a minimum value.
* **Inference Done** - filtering on the Object's inference-done flag
* **In-frame Areas** - filters on specific areas (see ODE Areas below) within the frame, with both areas of inclusion and exclusion supported.

**ODE Actions** handle the occurrence of Object Detection Events each with a specific action under the categories below. 
* **Actions on Buffers** - Capture Frames and Objects to JPEG images and save to file.
* **Actions on Metadata** - Fill-Frames and Objects with a color, add Text & Shapes to a Frame, Hide Object Text & Borders.
* **Actions on ODE Data** - Print, Log, and Display ODE occurence data on screen.
* **Actions on Recordings** - Start a new recording session for a Record Tap or Sink 
* **Actions on Pipelines** - Pause Pipeline, Add/Remove Source, Add/Remove Sink, Disable ODE Handler
* **Actions on Triggers** - Disable/Enable/Reset Triggers
* **Actions on Areas** - Add/Remove Areas
* **Actions on Actions** - Disable/Enable Actions

Actions acting on Triggers, other Actions and Areas allow for a dynamic sequencing of detection events. For example, a one-time Occurrence Trigger using an Action can enable a one-time Absence Trigger for the same class. The Absence Trigger using an Action can then reset/re-enable the one-time Occurrence Trigger. Combined, they can be used to alert when one or more objects first enters and then exits the frame or Area.

**ODE Areas**, [Lines](/docs/api-display-type.md#dsl_display_type_rgba_line_new) and [Polygons](/docs/api-display-type.md#dsl_display_type_rgba_polygon_new) can be added to any number of Triggers as additional criteria. 

* **Line Areas** - criteria is met when a specific edge of an object's bounding box - left, right, top, bottom - intersects with the Line Area
* **Polygon Areas** - criteria is met when a specific point of an object's bounding box - south, south-west, west, north-west, north, etc - is within the Polygon 

A simple example using python

```python
# example assumes that all return values are checked before proceeding

# Create a new Print Action to print the ODE Frame/Object details to the console
retval = dsl_ode_action_print_new('my-print-action')

# Create a new Capture Frame Action to capture the full frame to a jpeg image and save to the local dir
# The action can be used with multiple triggers for multiple sources.
# Set annotate=True to add bounding box and label to object that triggered the ODE occurrence.
retval = dsl_ode_action_capture_frame_new('capture-action', outdir='./', annotate=True)

# Create a new Occurrence Trigger that will invoke the above Actions on first occurrence of an object with a
# specified Class Id. Set the Trigger limit to one as we are only interested in capturing the first occurrence.
retval = dsl_ode_trigger_occurrence_new('east-cam-1-trigger', source='east-cam-1', class_id=0, limit=1)
retval = dsl_ode_trigger_action_add_many('east-cam-1-trigger', actions=['print-action', 'capture-action', None])

# Create an Area of inclusion using a previously defined [Rectangle Display Type](#display-types) as
# criteria for occurrence and add the Area to the Trigger. A detected object must have at least one pixel of
# overlap before occurrence will be triggered and the Actions invoked.
retval = dsl_ode_area_inclusion_new('east-cam-1-area', 'east-cam-1-polygon', display=True)
retval = dsl_ode_trigger_area_add('east-cam-1-trigger', 'east-cam-1-area')

# Create an ODE handler to manage the Trigger, add the Trigger to the handler
retval = dsl_pph_ode_new('ode-handler)
retval = dsl_pph_ode_trigger_add('ode-handler, 'east-cam-1-trigger')

#  Then add the handler to the sink (input) pad of a Tiler.
retval = dsl_tiler_pph_add('tiler', 'ode-handler', DSL_PAD_SINK)
```

[Issue #259](https://github.com/canammex-tech/deepstream-services-library/issues/259) has been opened to track all open items related to ODE Services.

See the below API Reference sections for more information
* [ODE Pad Probe Handler API Reference](/docs/api-pph.md)
* [ODE Trigger API Reference](/docs/api-ode-trigger.md)
* [ODE Action API Reference](/docs/api-ode-action.md)
* [ODE Area API Reference](/docs/api-ode-area.md)

There are several ODE Python examples provided [here](/examples/python)

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

--- 

## Display Types
On-Screen Display Types, RGBA text and shapes, can be added to a frame's metadata to be shown by an [On-Screen Display](/docs/api-osd.md) component downstream. 
There are two base types used when creating other complete types for actual display. 
* RGBA Color
* RGBA Font

There are four types for displaying text and shapes. 
* RGBA Line
* RGBA Arrow
* RGBA Rectangle
* RGBA Polygon
* RGBA Circle

And three types for displaying source information specific to each frame. 
* Source Number
* Source Name
* Source Dimensions

The [Add Display Meta ODE Action](/docs/api-ode-action.md#dsl_ode_action_display_meta_add_new) adds the data under control of one or more Triggers to render all types of video annotations.

Refer to the [Display Type API](/docs/api-display-type.md)

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
# Defines a class of all component names associated with a single RTSP Source. 
# Objects of this class will be used as "client_data" for all callback notifications.
# The names are derived from the unique Source name. 
class ComponentNames:
    def __init__(self, source):
        self.source = source
        self.record_tap = source + '-record-tap'
        self.occurrence_trigger = source + '-occurrence-trigger'
        self.ode_notify = source + '-ode-notify'
        self.start_record = source + '-start-record'

# Callback function to process all "record-start" notifications
def RecordStarted(event_id, trigger,
    buffer, frame_meta, object_meta, client_data):
    
    global duration
    
    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value
    
    # a good place to enabled an Always Trigger that adds `REC` text to the frame which can
    # be disabled in the RecordComplete callback below. And/or send notifictions to external clients.
    
    # in this example we will call on the Tiler to show the source that started recording.
    dsl_tiler_source_show_set('tiler', source=components.source, timeout=duration, has_precedence=True)

    
# Callback function to process all "record-complete" notifications
def RecordComplete(session_info_ptr, client_data):
    session_info = session_info_ptr.contents

    # cast the C void* client_data back to a py_object pointer and deref
    components = cast(client_data, POINTER(py_object)).contents.value
    
    print('sessionId:     ', session_info.sessionId)
    print('filename:      ', session_info.filename)
    print('dirpath:       ', session_info.dirpath)
    print('duration:      ', session_info.duration)
    print('containerType: ', session_info.containerType)
    print('width:         ', session_info.width)
    print('height:        ', session_info.height)
    
    retval, is_on = dsl_tap_record_is_on_get(components.record_tap)
    print('is_on:         ', is_on)
    
    retval, reset_done = dsl_tap_record_reset_done_get(components.record_tap)
    print('reset_done:    ', reset_done)
    
    # reset the Trigger so that a new session can be started.
    dsl_ode_trigger_reset(components.occurrence_trigger)
    
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

    # Next, create the Person Occurrence Trigger. We will reset the trigger in the recording complete callback
    retval = dsl_ode_trigger_occurrence_new(components.occurrence_trigger, 
        source=source, class_id=PGIE_CLASS_ID_PERSON, limit=1)
    if (retval != DSL_RETURN_SUCCESS):
        return retval

    # New (optional) Custom Action to be notified of ODE Occurrence, and pass component names as client data.
    retval = dsl_ode_action_custom_new(components.ode_notify, 
        client_handler=RecordStarted, client_data=components)
    if (retval != DSL_RETURN_SUCCESS):
        return retval

    # Create a new Action to start the record session for this Source, with the component names as client data
    retval = dsl_ode_action_tap_record_start_new(components.start_record, 
        record_tap=components.record_tap, start=15, duration=duration, client_data=components)
    if (retval != DSL_RETURN_SUCCESS):
        return retval
    
    # Add the Actions to the trigger for this source. 
    retval = dsl_ode_trigger_action_add_many(components.occurrence_trigger, 
        actions=[components.ode_notify, components.start_record, None])
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

    retval = dsl_osd_new('osd', clock_enabled=True)
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

## DSL Version
The version label of the DSL shared library `dsl.so` can be determined by calling `dsl_version_get()`. Version information and release notes can be found on this Repo's Wiki.

**Python Script**
```Python
current_version = dsl_version_get()
```

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

## Getting Started
* [Installing Dependencies](/docs/installing-dependencies.md)
* [Building and Importing DSL](/docs/building-dsl.md)

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Tap](/docs/api-tap.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIEs](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter Tees](/docs/api-tee)
* [Sink](docs/api-sink.md)
* [Pad Probe Handler](docs/api-pph.md)
* [ODE Trigger](docs/api-ode-trigger.md)
* [ODE Action ](docs/api-ode-action.md)
* [ODE Area](docs/api-ode-area.md)
* [Display Type](/docs/api-display-type.md)
* [Branch](docs/api-branch.md)
* [Component](/docs/api-component.md)

--- 
* <b id="f1">1</b> Quote from GStreamer documentation [here](https://gstreamer.freedesktop.org/documentation/?gi-language=c). [↩](#a1)
* <b id="f2">2</b> Quote from NVIDIA's Developer Site [here](https://developer.nvidia.com/deepstream-sdk). [↩](#a2)
