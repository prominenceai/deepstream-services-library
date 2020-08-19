# DSL Overview
### Overview Contents
* [Introduction](#introduction)
* [Pipeline Components](#pipeline-components)
  * [Streaming Sources](#streaming-sources)
  * [Primary and Secondary Inference Engines](#primary-and-secondary-inference-engines)
  * [Multi-Object Trackers](#multi-object-trackers)
  * [Object Detection Event Handler](#object-detection-event-handler)
  * [On-Screen Display](#on-screen-display)
  * [Multi-Source Tiler](#multi-source-tiler)
  * [Rendering and Streaming Sinks](#rendering-and-streaming-sinks)
  * [Pipeline Tees and Branches](#tees-and-branches)
* [DSL Initialization](#dsl-initialization)
* [DSL Delete All](#dsl-delete-all)
* [Main Loop Context](#main-loop-context)
* [Service Return Codes](#service-return-codes)
* [Batch Meta Handler Callback Functions](#batch-meta-handler-callback-functions)
* [X11 Window Support](#x11-window-support)
* [API Reference](#api-reference)

## Introduction
[NVIDIA’s DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) -- built on the open source [GStreamer](https://gstreamer.freedesktop.org/) "*an extremely powerful and versatile framework*<sup id="a1">[1](#f1)</sup>" -- enables experienced software developers to "*Seamlessly Develop Complex Stream Processing Pipelines*<sup id="a2">[2](#f2)</sup>". 

For those new to DeepStream, however, GStreamer comes with a learning curve that can be steep or lengthy for some. 

The core function of DSL is to provide a [simple and intuitive API](/docs/api-reference-list.md) for building, playing, and dynamically modifying NVIDIA® DeepStream Pipelines. Modifications made: (1) based on the results of the real-time video analysis, and: (2) by the application user through external input. An example of each:
1. Automatically starting a pre-cached recording session based on the occurrence of specific objects detected.
2. Interactively switching the view from one rendered Source stream to another. 

The general approach to using DSL is to:
1. Create several uniquely named [Components](/docs/api-reference-list.md) with desired attributes
2. Define and add one or more [Client callback functions](/docs/api-pipeline.md#client-callback-typedefs) and/or [Pad Probe Handlers](/docs/api-pph.md)(optional)
4. Add the Components to a new Pipeline
5. Play the Pipeline and start/join the main execution loop.

Using Python3, for example, the above can be written as:

```Python
# New uniquely named Pipeline. The name will be used to identify
# the Pipeline for subsequent Pipeline service requests.
retval = dsl_pipeline_new('my-pipeline')
```
Create a set of Components, each with a specific function and purpose. 
```Python
# new Camera Sources - setting dimensions and frames-per-second
retval += dsl_source_csi_new('my-source', width=1280, height=720, fps_n=30, fps_d=1)

# create more Source Components as needed

# new Primary Inference Engine - path to model engine and config file, interval 0 = infer on every frame
retval += dsl_gie_primary_new('my-pgie', path_to_engine_file, path_to_config_file, 0)

# new Multi-Source Tiler with dimensions of width and height 
retval += dsl_tiler_new('my-tiler', width=1280, height=720)

# new On-Screen Display for inference visualization - bounding boxes and labels - clocks disabled (False)
retval += dsl_osd_new('my-osd', clock_enabled=False)

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

# add the handler to the pipeline
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
 dsl_delete-all()
 ```

## Pipeline Components
There are seven categories of Components that can be added to a Pipeline, automatically assembled in the order shown below. Many of the categories support multiple types and in the cases of Sources, Secondary Inference Engines, and Sinks, multiple types can be added to a single Pipeline. 

![DSL Components](/Images/dsl-components.png)

## Streaming Sources
Streaming sources are the head component(s) for all Pipelines and all Pipelines must have at least one Source, among other components, before they can transition to a state of Playing. All Pipelines have the ability to multiplex multiple streams -- using their own Stream-Muxer -- as long as all Sources are of the same play-type; live vs. non-live with the ability to Pause. 

There are currently four types of Source components, two live connected Camera Sources:
* Camera Serial Interface (CSI) Source
* Universal Serial Bus (USB) Source

And two decode Sources that support both live and non-live streams.
* Universal Resource Identifier (URI) Source
* Real-time Streaming Protocol (RTSP) Source

All Sources have dimensions, width and height in pixels, and frame-rates expressed as a fractional numerator and denominator.  The decode Source components support multiple codec formats, including H.264, H.265, PNG, and JPEG. A [Dewarper Component](/docs/api-dewarper.md) (not show in the image above) capable of dewarping 360 degree camera streams can be added to both. 

A Pipeline's Stream-Muxer has settable output dimensions with a decoded and `batched` output stream that is ready to infer on.

See the [Source API](/docs/api-source.md) reference section for more information.

## Primary and Secondary Inference Engines
NVIDIA's GStreamer Inference Engines (GIEs), using pre-trained models, classify data to “infer” a result, e.g.: person, dog, car? A Pipeline may have at most one Primary Inference Engine (PGIE) -- with a specified set of classification labels to infer-with -- and multiple Secondary Inference Engines (SGIEs) that can Infer-on the output of either the Primary or other Secondary GIEs. Although optional, a Primary Inference Engine is required when adding a Multi-Object Tracker, Secondary Inference Engines, or On-Screen-Display to a Pipeline.

After creation, GIEs can be updated to:
* Use a new model-engine, config file and/or inference interval, and for Secondary GIEs (only), the Primary or Secondary GIE to infer on.

With Primary GIEs, applications can:
* Add/remove [Pad Probe Handlers] (#pad_probe_handlers) to process batched stream buffers with Metadata for each Frame and Detected-Object found within. 
* Enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Primary and Secondary GIE API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove [Pad Probe Handlers] (#pad_probe_handlers) to process batched stream buffers -- with Metadata for each Frame and Detected-Object found within.

Tracker components are optional and a Pipeline can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more details. See NVIDIA's [Low-Level Tracker Library Comparisons and Tradeoffs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.3.02.html#wwpID0E0Q20HA) for additional information.

## Multi-Source Tiler
To simplify the dynamic addition and removal of Sources and Sinks, all Source components connect to the Pipeline's internal Stream-Muxer, even when there is only one. The multiplexed stream must either be Tiled **or** Demuxed before reaching any Sink component downstream.

Tiler components transform the multiplexed streams into a 2D grid array of tiles, one per Source component. Tilers output a single stream that can connect to a single On-Screen Display (OSD). When using a Tiler, the OSD (optional) and Sinks (minimum one) are added directly to the Pipeline or Branch to operate on the Tiler's single output stream.
```Python
# assumes all components have been created first
retval = dsl_pipeline_component_add_many('my-pipeline', 
    ['src-1', 'src-2', 'pgie', 'tiler', 'osd', 'rtsp-sink`, `window-sink` None])
```
Tilers have dimensions, width and height in pixels, and rows and columns settings that can be updated after creation.

Clients of Tiler components can add/remove `batch-meta-handler` callback functions, [see below](#batch-meta-handler-callback-functions)

See the [Multi-Source Tiler](/docs/api-tiler.md) reference section for additional information.

## On-Screen Display
On-Screen Display (OSD) components highlight detected objects with colored bounding boxes and labels. A Clock with Positional offsets, colors and fonts can be enabled for Display. ODE Actions can be used to add/update Frame and Object metadata for the OSD to display. 

OSDs are optional and a Pipeline can have at most one when using a Tiler or one-per-source when using a Demuxer. See the [On-Screen Display API](/docs/api-osd.md) reference section for more information. 

## Rendering and Encoding Sinks
Sinks, as the end components in the Pipeline, are used to render the Streaming media, stream encoded data as a server or to a file or capture and save frame and object images to file. All Pipelines require at least one Sink Component to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case where the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently six types of Sink Components that can be added.

1. Overlay Render Sink
2. X11/EGL Window Sink
3. Media Container File Sink
4. Image Capture Sink
5. RTSP Server Sink
6. Fake Sink

Overlay and Window Sinks have settable dimensions: width and height in pixels, and X and Y directional offsets that can be updated after creation. 

File Sinks support three codec formats: H.264, H.265 and MPEG-4, with two media container formats: MP4 and MKV.

Image sinks capture and transform full video frames and identified objects into jpeg images and writes them to file

RTSP Sinks create RTSP servers - H.264 or H.265 - that are configured when the Pipeline is called to Play. The server is started and attached to the Main Loop context once [dsl_main_loop_run](#dsl-main-loop-functions) is called. Once started, the server can accept connections based on the Sink's unique name and settings provided on creation. The below for example,

```Python
retval = dsl_sink_rtsp_new('my-rtsp-sink', 5400, 8554, DSL_CODEC_H265, 200000, 0)
```
would use
```
rtsp://my-jetson.local:8554/my-rtsp-sink
```

See the [Sink API](/docs/api-sink.md) reference section for more information.

<br>

## Tees and Branches
There are two types of Tees that can be added to a Pipeline: Demuxers and Splitters.
1. **Demuxer** are used to demultiplex the single batched output from the Stream-muxer back into separate data streams.  
2. **Splitter** split the stream, batched or otherwise, into multiple duplicate streams. 

Branches connect to the downstream/output pads of the Tee, either as a single component, as in the case of a Sink or another Tee, or as multiple linked components, as in the case of **Branch 1** shown below. 

Important Notes: 
* Single component Branches can be added to a Tee directly, while multi-component Branches must be added to a new Branch component first.
* Branches ***can*** be added and removed from a Tee while a Pipeline is in a state of `Playing`, but the Tee must always have one. A [Fake Sink](/docs/api-sink.md) can be used as a Fake Branch when required.
* Tees are ***not*** required when adding multiple Sinks to a Pipeline or Branch. Multi-sink management is handled by the Pipeline/Branch directly. 

The following example illustrates how a **Pipeline** is assembled with a **Splitter**, **Demuxer**, **Tiler**, and **Branch** components. 

![Tees and Branches](/Images/tees-and-branches.png)

#### Building the Pipeline Example above, 

The first step is to create the two RTMP Sources - and the two File Sinks that will be used to stream the original video to file.

![Sources and File Sinks](/Images/sources-and-file-sinks.png)

```Python
# NOTE: this example assumes that all return values are checked for DSL_RESULT_SUCCESS before proceeding

# Create two live RTSP Sources

retval = dsl_source_rtsp_new('src-1', rtsp_uri_1, DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, True, 0, 100)
retval = dsl_source_rtsp_new('src-2', rtsp_uri_2, DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, True, 0, 100)

# Create two File Sinks for Branch 2, one for each source

retval = dsl_sink_file_new('file-sink1', './src-1.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
retval = dsl_sink_file_new('file-sink2', './src-2.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
```

Next, create all components for **Branch 1**

```Python

# Create a Primary GIE, Tracker, Multi-Source Tiler, On-Screen Display and X11/EGL Window Sink

retval = dsl_gie_primary_new('pgie', path_to_engine_file, path_to_config_file, 0)
retval = dsl_tracker_ktl_new('tracker', 480, 270)
retval = dsl_tiler_new('tiler', 1280, 720)
retval = dsl_osd_new('osd', True)
retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
```

**Branch 1**, with its multiple children, requires an explicit Branch component to manage and synchronize the child components when the Pipeline changes states. Create the branch and add the components as shown below.

![branch-1 with PGIE, Tracker, Tiler, OSD, Window](/Images/branch-1-with-pgie-tracker-tiler-osd-window.png)

```Python
# create a branch component for 'branch-1' and add all child components. 

retval = dsl_branch_new_component_add_many('branch-1', ['pgie', 'tiler', tracker', 'osd', 'window-sink', None])
```

**Branch 2**, with its single multi-source **Demuxer Tee** *does not* require an explicit Branch, nor do **Branches 3 and 4** consisting of a single File Sink each. 

Note: adding multiple sinks to a single branch requires a Branch component to contain them.

The relationship of Demuxer-output-Branch to the upstream Source component is set by the order of addition. The first Branch added to the Demuxer is linked from the first upstream Source added to the Pipeline - a one-to-one relationship. 

![branch with demuxer and sinks](/Images/branch-2-3-4.png)

```Python
# create a new Demuxer to de-multiplex the batched source streams and add the 
# two File Sinks as Branches for the Tee.

retval = dsl_tee_demuxer_new_branch_add_many('demuxer1', ['file-sink1', 'file-sink2', None])
```

The **Splitter Tee** is used to split/duplicate the batched stream into multiple branches for separate processing.

![Splitter with Branches 1 and 2](/Images/splitter-branch-1-branch-2.png)

```Python
# Create a new splitter and add 'branch-1` and the 'demexer' as Branch 2

retval = dsl_tee_splitter_new_branch_add_many('splitter', ['branch-1', 'demuxer', None])
```

Complete the assembly by creating the **Pipeline** and adding the two RTSP sources and Splitter

```Python
# finally, add the sources and splitter-tee to the pipeline

retval = dsl_pipeline_new_component_add_many('pipeline', ['src-1', 'src-2', 'splitter',  None])

# ready to play ...

```

### All combined, the example is written as.

```Python
# NOTE: this example assumes that all return values are checked for DSL_RESULT_SUCCESS before proceeding

# Create two live RTSP Sources
retval = dsl_source_rtsp_new('src-1', rtsp_uri_1, DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, True, 0, 100)
retval = dsl_source_rtsp_new('src-2', rtsp_uri_2, DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, True, 0, 100)

# Create two File Sinks, one for each source
retval = dsl_sink_file_new('file-sink1', './src-1.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
retval = dsl_sink_file_new('file-sink2', './src-2.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)

# Create the Primary GIE, Tracker, Multi-Source Tiler, On-Screen Display and X11/EGL File Sink
retval = dsl_gie_primary_new('pgie', path_to_engine_file, path_to_config_file, 0)
retval = dsl_tracker_ktl_new('tracker', 480, 270)
retval = dsl_tiler_new('tiler', 1280, 720)
retval = dsl_osd_new('osd', True)
retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)

# Create a branch component for 'branch-1' and add all child components. 
retval = dsl_branch_new_component_add_many('branch-1', ['pgie', 'tiler', tracker', 'osd', 'window-sink', None])

# create a new Demuxer to de-multiplex the batched source streams and add the 
# two File Sinks as Branches for the Tee.
retval = dsl_tee_new_branch_add_many('demuxer1', ['file-sink1', 'file-sink2', None])

# finally, add the sources and splitter-tee to the pipeline
retval = dsl_pipeline_new_component_add_many('pipeline', ['src-1', 'src-2', 'splitter',  None])
```

See the [Demuxer and Splitter Tee API](/docs/api-tee.md) reference section for more information. 

---

## Pad Probe Handlers
Pipeline components are linked together using directional ["pads"](https://gstreamer.freedesktop.org/documentation/gstreamer/gstpad.html?gi-language=c) with a Source Pad from one component as the producer of data connected to the Sink Pad of the next component as the comsumer. Data flowing over the coponent's pads can be monitored and inspected using a Pad-Probe with a specific Handler function.

There are three Pad Probe Handlers that can be created and added to either a Sink or Source Pad of most Pipeline components excluding Sources, Taps and Secondary GIE's.
1. Pipeline Meter - measures the throughput for each source in the Pipeline.
2. Object Detection Event Handler - manages a collection of [Triggers](/docs/api-ode-trigger.md) that invoke [Actions](/docs/api-ode-action.md) on the occurrence of specific frame and object metadata. 
3. Custom Handler- allows the client to install a callback with custom behavior. 

### Pipeline Meter Pad Probe Handler
The [Meter Pad Probe Handler](/docs/api-pph.md#) measures a Pipeline's throughput for each Source detected in the batched stream. When creating a Meter PPH, the client provides a callback funtion to be notified with new measurements at an interval specified by the client. The notification includes the average frames-per-second over the last interval and over the current session, which can be stoped and new session started at anytime. 

### Object Detection Event Pad Probe Handler
The Object Detection Event (ODE) Pad Probe Handler (PPH) manages an ordered collection of **Triggers**, each with an ordered collection of **Actions** and an optional collection of **Areas**. Triggers use settable criteria to process the Frame and Object metadata produced by the Primary and Secondary GIE's looking for specific detection events. When the criteria for the Trigger is met, the Trigger invokes all Actions in its ordered collection. Each unique Area and Action created can be added to multiple Triggers as shown in the diagram below. The ODE Handler has n Triggers, each Trigger has one shared Area and one unique Area, and one shared Action and one unique Action.

![ODE Services](/Images/ode-services.png)

The Handler is added to the Pipeline before the On-Screen-Display (OSD) component allowing Actions to update the metadata for display. 

There is a growing list of **ODE Triggers** supported:
* **Always** - triggers on every frame. Once per-frame always.
* **Absence** - triggers on the absence of objects within a frame. Once per-frame at most.
* **Occurrence** - triggers on each object detected within a frame. Once per-object at most.
* **Summation** - triggers on the summation of all objects detected within a frame. Once per-frame always.
* **Intersection** - triggers on the intersection of two objects detected within a frame. Once per-intersecting-pair.
* **Minimum** - triggers when the count of detected objects in a frame fails to meet a specified minimum number. Once per-frame at most.
* **Maximum** - triggers when the count of detected objects in a frame exceeds a specified maximum number. Once per-frame at most.
* **Range** - triggers when the count of detected objects falls within a specified lower and upper range. Once per-frame at most.
* **Smallest** - triggers on the smallest object by area if one or more objecst is detected. Once per-frame at most.
* **Largest** - triggers on the largets object by area if one or more objects is detected. Once per-frame at most.
* **Custom** - allows the client to provide a callback function that implements a custom "Check for Occurrence".

Triggers have optional, settable criteria and filters: 
* **Class Id** - filters on a specified GIE Class Id when checking detected objects. Use `DSL_ODE_ANY_CLASS`
* **Source Id** - filters on a unique Source Id, with a default of `DSL_ODE_ANY_SOURCE`
* **Dimensions** - filters on an object's dimensions ensuring both width and height minimums and maximum are met. 
* **Confidence** - filters on an object's GIE confidence requiring a minimum value.
* **Inference Done** - filtering on the Object's inference-done flag
* **In-frame Areas**

Minimum Frames as criteria, expressed as two numbers `n out d` frames, and other forms of detection hysteresis are being considered. 

**ODE Actions** can act on Triggers, on Actions and on Areas allowing for a dynamic sequencing of detection events. For example, a one-time Occurrence Trigger, using an Action, can enable a one-time Absence Trigger for the same class, and the Absence Trigger, using an Action, can reset/re-enable the Occurrence Trigger.

* **Actions on Buffers** - Capture Frames and Objects to JPEG images and save to file.
* **Actions on Metadata** - Fill-Frames and Objects with a color, add Text & Shapes to a Frame, Hide Object Text & Borders.
* **Actions on ODE Data** - Print, Log, and Display ODE occurence data on screen.
* **Actions on Recordings** - Start are a new recording session for a Record Tap or Sink 
* **Actions on Pipelines** - Pause Pipeline, Add/Remove Source, Add/Remove Sink, Disable ODE Handler
* **Actions on Triggers** - Disable/Enable/Reset Triggers
* **Actions on Areas** - Add/Remove Areas
* **Actions on Actions** - Disable/Enable Actions

Planned new actions for upcoming releases include **Start/Stop Record**, **Serialize/Deserialize**, and **Message to cloud**

**ODE Areas**, rectangles with location and dimensions, can be added to any number of Triggers as additional criteria for object occurrence/absence.

A simple example using python

```python
# example assumes that all return values are checked before proceeding

# Create a new Print Action to print the ODE Frame/Object details to the console
retval = dsl_ode_action_print_new('my-print-action')

# Create a new Capture Frame Action to capture the full frame to a jpeg image and save to the local dir
retval = dsl_ode_action_capture_frame_new('my-capture-action', outdir='./')

# Create a new Occurrence Trigger that will invoke the above Actions on first occurrence of an object with a
# specified Class Id. Set the Trigger limit to one as we are only interested in capturing the first occurrence.
retval = dsl_ode_trigger_occurrence_new('my-occurrence-trigger', class_id=0, limit=1)
retval = dsl_ode_trigger_action_add_many('my-occurrence-trigger', actions=['my-print-action', 'my-capture-action', None])

# Create a new Area as criteria for occurrence and add to our Trigger. An Object must have
# at least one pixel of overlap before occurrence will be triggered and the Actions invoked.
retval = dsl_ode_area_new('my-area', left=245, top=0, width=20, height=1028, display=True)
retval = dsl_ode_trigger_area_add('my-occurrence-trigger', 'my-area')

# New ODE handler to add our Trigger to, and then add the handler to the Pipeline.
retval = dsl_ode_handler_new('my-handler)
retval = dsl_ode_handler_trigger_add('my-handler, 'my-occurrence-trigger')
retval = ddsl_pipeline_component_add('my-pipeline', 'my-handler')
```
[Issue #259](https://github.com/canammex-tech/deepstream-services-library/issues/259) has been opened to track all open items related to ODE Services.

See the below API Reference sections for more information
* [ODE Handler API Refernce](/docs/api-ode-handler.md)
* [ODE Trigger API Refernce](/docs/api-ode-trigger.md)
* [ODE Action API Reference](/docs/api-ode-action.md)
* [ODE Area API Reference](/docs/api-ode-area.md)

There are several ODE Python examples provided [here](/examples/python)


## Display Types
On-Screen Display Types, RGBA text and shapes, can be added to a frame's metadata to be shown by an [On-Screen Display](/docs/api-osd.md) component downstream. The [Add Display Meta ODE Action](/docs/api-ode-action.md#dsl_ode_action_display_meta_add) adds the data under control of one or more Triggers to render all types of video adornments.


---


## DSL Initialization
The library is automatically initialized on **any** first call to DSL. There is no explicit init or deint service. DSL will initialize GStreamer at this time, unless the calling application has already done so. 

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

## Batch Meta Handler Callback Functions
All of the `one-at-most` Pipeline Components -- Primary GIEs, Multi-Object Trackers, On-Screen Displays, and Tilers -- support the dynamic addition and removal of `batch-meta-handler` callback functions. Multiple handlers can be added to the component's Input (sink-pad) and Output (src-pad) in any Pipeline state. Batch-meta-handlers allow applications to monitor and block-on data flowing over the component's pads.

Each batch-meta-handler function is called with a buffer of meta-data for each batch processed. A Pipeline's batch-size is set to the current number of Source Components upstream.

Adding a batch-meta-handler to the sink-pad of an On-Screen Display component, for example, is an ideal point in the stream to monitor, process, and make decisions based on all inference and tracker results from  upstream.

When using Python3, NVIDIA's [Python-bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps#python-bindings) (see pypd in the example below) are used to process the buffered batch-meta in the handler callback function. The bindings can be downloaded from [here](https://developer.nvidia.com/deepstream-download#python_bindings)

```Python
# Callback function to handle batch-meta data
def osd_batch_meta_handler_cb(buffer, user_data):

    # Using NVIDIA's pypd.so bindings to process the batch-meta-data
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break    

        # parse frame_meta .....  
        # On first occurence of some object of interest, start streaming to file.
        
            # add the new file sink to immediately start streaming to file.
            dsl_pipeline_component_add('my-pipeline', 'my-file-sink')
            
            # optionally install a new Callback function to remove the sink
            # on some condition observed in the batch-meta data
            
            # return False to self remove this handler function.
            return False 

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return True

while True:

    # Create a new OSD component and add the batch-meta handler function above to the Sink Pad.
    retval = dsl_osd_new('my-osd', False)
    if retval != DSL_RETURN_SUCCESS:
        break
    retval = dsl_osd_batch_meta_handler_add('my-osd', DSL_PAD_SINK, osd_batch_meta_handler_cb, None)
    if retval != DSL_RETURN_SUCCESS:
        break

    # Create a new H.264 File Sink component to be added to the Pipeline by the osd_batch_meta_handler_cb
    retval = dsl_sink_file_new('my-file-sink', './my-video.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
    if retval != DSL_RESULT_SUCCESS:
        break

    # Create all other required components and add them to the Pipeline (see some examples above)
    # ...
    
    retval = dsl_pipeline_play('my-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break
 
    # Start/Join with main loop until released - blocking call
    dsl_main_loop_run()
    retval = DSL_RETURN_SUCCESS
    break

#print out the final result
print(dsl_return_value_to_string(retval))

# clean up all resource
dsl_pipeline_delete_all()
dsl_component_delete_all()

```

### *dsl_batch_meta_handler_cb*
```C++
typedef boolean (*dsl_batch_meta_handler_cb)(void* batch_meta, void* user_data);
```
Callback typedef for a client batch meta handler function. Once added to a Component, the function will be called when the component receives a batch meta buffer from its sink or src pad. Functions of this type are added to a component by calling the `dsl_<component-type>_batch_meta_handler_add` service, and removed with the corresponding `dsl_<component-type>_batch_meta_handler_removed` service.

**Parameters**
* `batch_meta` - [in] pointer to a buffer of batc-meta-data to process
* `user_data` - [in] opaque pointer to client's user data, passed into the component on callback add

<br>

## X11 Window Support
DSL provides X11 Window support for Pipelines that have one or more Window Sinks. An Application can create Windows - using GTK+ for example - and share them with Pipelines prior to playing, or let the Pipeline create a Display and Window to use. 

``` Python
# Function to be called on XWindow ButtonPress event
def xwindow_button_event_handler(xpos, ypos, client_data):
    print('button pressed: xpos = ', xpos, ', ypos = ', ypos)
    
# Function to be called on XWindow KeyRelease event - non-live streams
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == ' ':
        dsl_main_loop_quit()
 
# Function to be called on XWindow Delete event
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_main_loop_quit()

while True:

    # New Pipeline
    retval = dsl_pipeline_new('my-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break

    retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    # Add the XWindow event handler functions defined above
    retval = dsl_pipeline_xwindow_key_event_handler_add(1'my-pipeline', xwindow_key_event_handler, None)
    if retval != DSL_RETURN_SUCCESS:
        break
    retval = dsl_pipeline_xwindow_button_event_handler_add('my-pipeline', xwindow_button_event_handler, None)
    if retval != DSL_RETURN_SUCCESS:
        break
    retval = dsl_pipeline_xwindow_delete_event_handler_add('my pipeline', xwindow_delete_event_handler, None)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    # Create all other required components and add them to the Pipeline (see some examples above)
    # ...
 
    retval = dsl_pipeline_play('my-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break
 
    # Start/Join with main loop until released - blocking call
    dsl_main_loop_run()
    retval = DSL_RETURN_SUCCESS
    break

#print out the final result
print(dsl_return_value_to_string(retval))

# clean up all resources
dsl_pipeline_delete_all()
dsl_component_delete_all()
```

<br>

---

## Getting Started
* [Installing Dependencies](/docs/installing-dependencies.md)
* [Building and Importing DSL](/docs/building-dsl.md)

## API Reference
* [List of all Services](/docs/api-reference-list.md)
* [Pipeline](/docs/api-pipeline.md)
* [Source](/docs/api-source.md)
* [Dewarper](/docs/api-dewarper.md)
* [Primary and Secondary GIEs](/docs/api-gie.md)
* [Tracker](/docs/api-tracker.md)
* [ODE Handler](docs/api-ode-handler.md)
* [ODE Trigger](docs/api-ode-trigger.md)
* [ODE Action ](docs/api-ode-action.md)
* [ODE Area](docs/api-ode-area.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler](/docs/api-tiler.md)
* [Demuxer and Splitter Tees](/docs/api-tee)
* [Sink](docs/api-sink.md)
* [Branch](docs/api-branch.md)
* [Component](/docs/api-component.md)

--- 
* <b id="f1">1</b> Quote from GStreamer documentation [here](https://gstreamer.freedesktop.org/documentation/?gi-language=c). [↩](#a1)
* <b id="f2">2</b> Quote from NVIDIA's Developer Site [here](https://developer.nvidia.com/deepstream-sdk). [↩](#a2)
