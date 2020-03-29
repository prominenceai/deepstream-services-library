# DSL Overview
### Overview Contents
* [Introduction](#introduction)
* [Pipeline Components](#pipeline-components)
* [Streaming Sources](#streaming-sources)
* [Primary and Secondary Inference Engines](#primary-and-secondary-inference-engines)
* [Multi-Object Trackers](#multi-object-trackers)
* [On-Screen Display](#on-screen-display)
* [Multi-Source Tiler](#multi-source-tiler)
* [Rendering and Streaming Sinks](#rendering-and-streaming-sinks)
* [DSL Initialization](#dsl-initialization)
* [Main Loop Context](#main-loop-context)
* [Service Return Codes](#service-return-codes)
* [Batch Meta Handler Callback Functions](#batch-meta-handler-callback-functions)
* [X11 Window Support](#x11-window-support)
* [API Reference](#api-reference)

## Introduction
[NVIDIA’s DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) -- built on the open source [GStreamer](https://gstreamer.freedesktop.org/) "*an extremely powerful and versatile framework*<sup id="a1">[1](#f1)</sup>" -- enables experienced software developers to "*Seamlessly Develop Complex Stream Processing Pipelines*<sup id="a2">[2](#f2)</sup>". 

For those new to DeepStream, however, GStreamer comes with a learning curve that can be steep or lengthy for some. 

The DeepStream Services Library (DSL) was built to enable *"less-experienced"* programmers and hobbyists to develop custom DeepStream applications -- in Python3 or C/C++ -- at a higher level of abstraction. 

The core function of DSL is to provide a [simple and intuitive API](/docs/api-reference-list.md) for building, playing, and dynamically modifying NVIDIA® DeepStream Pipelines. Modifications made: (1) based on the results of the real-time video analysis, and: (2) by the application user through external input. An example of each:
1. Programmatically adding a stream to [File Sink](/docs/api-sinks.md) based on the occurrence of specific detected objects.
2. Interactively resizing stream and window dimensions for viewing control.

The general approach to using DSL is to:
1. Create one or more uniquely named DeepStream [Pipelines](/docs/api-pipeline.md)
2. Create a number of uniquely named [Components](/docs/api-reference-list.md) with desired attributes
3. Define and add one or more [Client callback functions](/docs/api-pipeline.md#client-callback-typedefs) (optional)
4. Add the Components to the Pipeline(s)
5. Play the Pipeline(s) and start/join the main execution loop.

Using Python3, for example, the above can be written as:

```Python
# Import the DSL APIs
from dsl import *

# New uniquely named Pipeline. The name will be used to identify
# the Pipeline for subsequent Pipeline service requests.
retval = dsl_pipeline_new('my-pipeline')
```
Create a set of Components, each with a specific function and purpose. 
```Python
# new Camera Sources - setting dimensions and frames-per-second
retval += dsl_source_csi_new('my-source', 1280, 720, 30, 1)

# new Primary Inference Engine - path to model engine and config file, interval 0 = infer on every frame
retval += dsl_gie_primary_new('my-pgie', path_to_engine_file, path_to_config_file, 0)

# new Multi-Source Tiler with dimensions of width and height 
retval += dsl_tiler_new('my-tiler', 1280, 720)

# new On-Screen Display for inference visualization - bounding boxes and labels - clocks disabled (False)
retval += dsl_osd_new('my-osd', False)

# new X11/EGL Window Sink for video rendering - Pipeline will create a new XWindow if one is not provided
retval += dsl_sink_window_new('my-window-sink', 1280, 720)

if retval != DSL_RESULT_SUCCESS:
    # one of the components failed to create, handle error
```

Add one or more Client Callback Functions

```Python
# Function to be called on XWindow Delete event
def xwindow_delete_event_handler(client_data):
    # Quit the main loop to shutdown and release all resources
    dsl_main_loop_quit()

retval = dsl_pipeline_xwindow_delete_event_handler_add('my pipeline', xwindow_delete_event_handler, None)
```
Add the components to the Pipeline.
```Python
# Using a Null terminated list - in any order
retval = dsl_pipeline_component_add_many('my-pipeline', 
    ['my-source', 'my-pgie', 'my-tiler', 'my-osd', 'my-sink', None])
```

Transition the Pipeline to a state of Playing and start/join the main loop

```Python
 retval = dsl_pipeline_play('my-pipeline')
 if retval != DSL_RESULT_SUCCESS:
  # Pipeline failed to play, handle error
  dsl_main_loop_run()
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
* Use a new model-engine, config file and/or inference interval, and for Secondary GIEs the GIE to infer on.
* To enable/disable output of bounding-box frame and label data to text file in KITTI format for [evaluating object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark).
* To enable/disable output of raw layer information to binary file.

With Primary GIEs, applications can:
* Add/remove `batch-meta-handler` callback functions [see below](#batch-meta-handler-callback-functions)
* Enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Primary and Secondary GIE API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove `batch-meta-handler` callback functions. [see below](#batch-meta-handler-callback-functions)

Tracker components are optional and a Pipeline can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more information.

## Multi-Source Tiler
To simplify the dynamic addition and removal of Sources and Sinks, all Source components connect to the Pipeline's internal Stream-Muxer, even when there is only one. The multiplexed stream must either be Tiled **or** Demuxed before reaching Sink any components downstream.

Tiler components transform the multiplexed streams into a 2D grid array of tiles, one per Source component. Tilers output a single stream that can connect to a single On-Screen Display (OSD). When using a Tiler the OSD (optional) and Sinks (minimum one) are added directly to the Pipeline or Branch to operate on the Tiler's single output stream.
```Python
# assumes all components have been created first
retval = dsl_pipeline_component_add_many('my-pipeline', 
    ['src-1', 'src-2', 'pgie', 'tiler', 'osd', 'rtsp-sink`, `window-sink` None])
```
Tilers have dimensions, width and height in pixels, and rows and columns settings that can be updated after creation.

Clients of Tiler components can add/remove `batch-meta-handler` callback functions, [see below](#batch-meta-handler-callback-functions)

See the [Multi-Source Tiler](/docs/api-tiler.md) reference section for additional information.

## Multi-Source Demuxer
Demuxers demultiplex the multiplexed source streams back into individual output streams. When using a Demuxer, each output stream -- one for each input Source -- must (ultimately) connect to one or more [Sink Components](#rendering-and-streaming-sinks). A Multi-Source Demuxer is one type of Tee component that can manages multiple Branches. See 

## On-Screen Display
On-Screen Display (OSD) components highlight detected objects with colored bounding boxes, labels and clocks. Positional offsets, colors and fonts can all be set and updated. A `batch-meta-handler` callback function, added to the input (sink pad) of the OSD, enables clients to add custom meta data for display [see below](#batch-meta-handler-callback-functions).

OSDs are optional and a Pipeline can have at most one when using a Tiler or one-per-source when using a Demuxer. See the [On-Screen Display API](/docs/api-osd.md) reference section for more information. 

## Rendering and Streaming Sinks
Sinks, as the end components in the Pipeline, are used to either render the Streaming media or to stream encoded data as a server or to a file. All Pipelines require at least one Sink Component in order to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case were the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently five types of Sink Components that can be added.

1. Overlay Render Sink
2. X11/EGL Window Sink
3. Media Container File Sink
4. RTSP Server Sink
5. Fake Sink

Overlay and Window Sinks have settable dimensions: width and height in pixels, and X and Y directional offsets that can be updated after creation. 

File Sinks support three codec formats: H.264, H.265 and MPEG-4, with two media container formats: MP4 and MKV.

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
There are two type of Tee components, Demuxers and Splitters. Demuxer Tees are used to de-multiplex the single batched output from the Stream-muxer back into separate data streams.  Splitter Tees split the stream, batched or otherwise, into multiple duplicate streams. 

Branches connect to the downstream/output pads of the Tee, either as a single component -- as in the case of a Sink or another Tee -- or as multiple linked components -- as in the case of **Branch 1** shown below. Single component Branches can be added to the Tee directly, while multi component Branches must be added to an explicit Branch component first.

The following example illustrates how a **Pipeline** is assembled with a **Splitter**, **Demuxer**, **Tiler**, and **Branch** components. 

![Tees and Branches](/Images/tees-and-branches.png)

To build the Pipeline above, create the two RTMP Sources and the two File Sinks that will be used to stream the raw video feeds to file.

![Sources and File Sinks](/Images/sources-and-file-sinks.png)

```Python
# Create two live RTSP Sources

retval = dsl_source_rtsp_new('src-1', rtsp_uri_1, DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, True, 0)
retval = dsl_source_rtsp_new('src-2', rtsp_uri_2, DSL_RTP_ALL, DSL_CUDADEC_MEMTYPE_DEVICE, True, 0)

# ...and two File Sinks for Branch 2, one for each source

retval = dsl_sink_file_new('file-sink1', './src-1.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
retval = dsl_sink_file_new('file-sink2', './src-2.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)
```

Next, create all components for **Branch  1**

```Python

# Create a Primary GIE, Tracker, Tracker, Multi-Source Tiler, On-Screen Display and X11/EGL File Sink

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

retval = dsl_branch_new('branch-1')
retval = dsl_branch_component_add_many('branch-1', ['pgie', 'tiler', tracker', 'osd', 'window-sink', None])
```

**Branch 2**, with its single multi-source **Demuxer Tee** -- used to de-multiplex the batched source streams back into separate output streams -- *does not* require an explicit Branch, nor do **Branches 3 and 4** with a single File Sink each. 

Note: when adding multiple sinks to single branch, either **3 and 4** below, *would* require the creation of a Branch component to contain them.

The relationship of Source to Demuxer-Branch is set by their order of addition. The first Branch added to the Demuxer is linked downstream from the first Source added to the Pipeline - a one-to-one relationship. 

![branch with demuxer and sinks](/Images/branch-2-3-4.png)

```Python
# create a new Demuxer to de-multiplex the batched source streams and add the 
# two File Sinks as Branches for the Tee.

retval = dsl_tee_demuxer_new('demuxer')
retval = dsl_tee_branch_add_many('demuxer1', ['file-sink1', 'file-sink2', None])
```

The **Splitter Tee** is used to split/duplicate the batched stream into multiple branches for separate processing.

![Splitter with Branches 1 and 2](Images/splitter-branch-1-branch-2.png)

```Python
# Create a new splitter and add 'branch-1` and the 'demexer' as Branch 2

retval = dsl_tee_splitter_new('splitter')
retval = dsl_tee_branch_add_many('splitter', ['branch-1', 'demuxer', None])
```

Complete the assembly by creating the **Pipeline** and adding the two RTSP sources and Splitter

```Python
# finally, add the sources and splitter-tee to the pipeline

retval = dsl_pipeline_new('pipeline')
retval = dsl_pipeline_component_add_many('pipeline', ['src-1', 'src-2', 'splitter',  None])

# ready to play ...
```

A complete example, with dynamic inference driven start/stop control over the streaming to file can found under the [Python Examples](/docs/examples-python.md)


## DSL Initialization
The library is automatically initialized on **any** first call to DSL. There is no explicit init or deint service. DSL will initialize GStreamer at this time, unless the calling application has already done so. 

<br>

## Main Loop Context
After creating a Pipeline(s), creating and adding Components, and setting the Pipeline's state to Playing, the Application must call `dsl_main_loop_run()`. The service creates a mainloop that runs/iterates the default GLib main context to check if anything the Pipeline is watching for has happened. The main loop will be run until another thread -- typically a client Callback function called from the Pipeline's context -- calls `dsl_main_loop_quit()`

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
* [Primary and Secondary GIE](/docs/api-gie)
* [Tracker](/docs/api-tracker.md)
* [On-Screen Display](/docs/api-osd.md)
* [Tiler and Demuxer](/docs/api-tiler.md)
* [Sink](docs/api-sink.md)
* [Component](/docs/api-component.md)

--- 
* <b id="f1">1</b> Quote from GStreamer documentation [here](https://gstreamer.freedesktop.org/documentation/?gi-language=c). [↩](#a1)
* <b id="f2">2</b> Quote from NVIDIA's Developer Site [here](https://developer.nvidia.com/deepstream-sdk). [↩](#a2)
