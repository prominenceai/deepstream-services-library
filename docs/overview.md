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
* [Main Loop Context](#main-loop-context)
* [Service Return Codes](#service-return-codes)
* [Batch Meta Handler Callback Functions](#batch-meta-handler-callback-functions)
* [X11 Window Support](#x11-window-support)
* [API Reference](#api-reference)

## Introduction
NVIDIA’s DeepStream SDK -- built on the open source [GStreamer](https://gstreamer.freedesktop.org/) *"an extremely powerful and versatile framework"* -- enables experienced software developers to *"Seamlessly Develop Complex Stream Processing Pipelines"*. For those new to DeepStream, however, GStreamer comes with a learning curve that can be a little step or lengthy for some. 

The DeepStream Services Library (DSL) was built to enable *"less-experienced"* programmers and hobbyist to develop custom DeepStream applications in Python3 or C/C++ at a much higher level of abstraction - built to encapsulate the complexity that comes with GStreamer's power and flexibility.

The core function of DSL is to provide a [simple and intuitive API](/docs/api-reference-list.md) for building, playing, and dynamically modifying NVIDIA® DeepStream Pipelines; modifications made (1) based on the results of the real-time video analysis (2) by the application User through external input. An example of each:
1. Programmatically adding a stream to [File Sink](/docs/api-sinks.md) based on the occurrence of specific objects detected.
2. Interactively resizing stream and window dimensions for viewing control

The general approach to using DSL is to
1. Create one or more uniquely named DeepStream [Pipelines](/docs/api-pipeline.md)
2. Create a number of uniquely named [Components](/docs/api-reference-list.md) with desired attributes
3. Define and add one or more [Client callback functions](/docs/api-pipeline.md#client-callback-typedefs) (optional)
4. Add the Components to the Pipeline(s)
5. Play the Pipeline(s) and start/join the main execution loop.

Using Python3 for example, the above can be written as

```Python
# Import the DSL APIs
from dsl import *

# New uniquely named Pipeline. The name will be used to identify
# the Pipline for subsequent Pipeline service requests.
retval = dsl_pipeline_new('my-pipeline')
```
Create a set of Components, each with a specific function and purpose. 
```Python
# new Camera Source - setting dimensions and frames-per-second
retval += dsl_source_csi_new('my-source', 1280, 720, 30, 1)

# new Primary Inference Engine - path to model engine and config file, infer-on every frame with interval = 0
retval += dsl_gie_primary_new('my-pgie', path_to_engine_file, path_to_config_file, 0)

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
# Using a Null terminated list
retval = dsl_pipeline_component_add_many('my-pipeline', ['my-source', 'my-pgie', 'my-osd', 'my-sink', None])
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
Streaming sources are the head component(s) for all Pipelines and all Pipelines must have at least one Source, among others components, before they can transition to a state of Playing. All Pipelines have the ability to multiplex multiple streams -- using their own Stream-Muxer -- as long as all Sources are of the same play-type; live vs. non-live with the ability to Pause. 

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
NVIDIA's GStreamer Inference Engines (GIEs), using pre-trained models, classify data to “infer” a result; person, dog, car?. A Pipeline may have at most one Primary Inference Engine (PGIE) -- with a specified set of classification labels to infer-with -- and multiple Secondary Inference Engines (SGIEs) that can Infer-on the output of either the Primary or other Secondary GIEs. Although optional, a Primary Inference Engine is required when adding a Multi-Object Tracker, Secondary Inference Engines, or On-Screen-Display to a Pipeline.

After creation, GIEs can be updated to:
* Use a new model-engine, config file and/or inference interval, and for Secondary GIEs the GIE to infer on.
* To enable/disable output of bounding-box frame and label data to text file in KITTI format for [evaluating object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark).
* To enable/disable output of raw layer information to binary file.

With Primary GIEs, applications can:
* add/remove `batch-meta-handler` callback functions [see below](#batch-meta-handler-callback-functions)
* enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Primary and Secondary GIE API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove `batch-meta-handler` callback functions. [see below](#batch-meta-handler-callback-functions)

Tracker components are optional and a Pipeline can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more information.

## On-Screen Display
On-Screen Display (OSD) components highlight detected objects with colored bounding boxes, labels and clocks. Positional offsets, colors and fonts can all be set and updated. A `batch-meta-handler` callback function, added to the input (sink pad) of the OSD, enables clients to add custom meta data for display [see below](#batch-meta-handler-callback-functions).

OSDs are optional and a Pipeline can have at most one. See the [On-Screen Display API](/docs/api-osd.md) reference section for more information. 

## Multi-Source Tiler
Tiler components transform the multiplexed streams into a 2D grid array of tiles, one per Source component. Tilers have dimensions, width and height in pixels, and rows and columns settings that can be updated after creation.

Clients of Tiler components can add/remove `batch-meta-handler` callback functions, [see below](#batch-meta-handler-callback-functions)

Tiler components are optional and a Pipeline can have at most one. See the [Multi-Source Tiler API](/docs/api-tiler.md) reference section for more information.

## Rendering and Streaming Sinks
Sinks, as the end components in the Pipeline, are used to either render the Streaming media or to stream encoded data as a server or to file. All Pipelines require at least one Sink Component in order to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case were the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently five types of Sink Components that can be added.

1. Overlay Render Sink
2. X11/EGL Window Sink
3. Media Container File Sink
4. RTSP Server Sink
5. Fake Sink

Overlay and Window Sinks have settable dimensions, width and height in pixels, and X and Y directional offsets that can be updated after creation. 

File Sinks support three codec formats, H.264, H.265 and MPEG-4, with two media container formats, MP4 and MKV.

RTSP Sinks create RTSP servers - H.264 or H.265 - that are configured when the Pipeline is called to Play. The server is started and attached to the Main Loop context once [dsl_main_loop_run](#dsl-main-loop-functions) is called. Once started, the server can accept connections based on the Sink's unique name and settings provided on creation. Using the below for example,

```Python
retval = dsl_sink_rtsp_new('my-rtsp-sink', 8050, 554, DSL_CODEC_H265, 200000, 0)
```
would use
```
http://localhost::8050/my-rtsp-sink
```

See the [Sink API](/docs/api-sink.md) reference section for more information.

<br>

## Main Loop Context
After creating a Pipeline(s), creating and adding Components, and setting the Pipeline's state to Playing, the Application must call `dsl_main_loop_run()`. The service creates a mainloop that runs/iterates the default GLib main context to check if anything the Pipeline is watching for has happened. The main loop will be run until another thread -- typically a client Callback function called from the Pipeline's context -- calls `dsl_main_loop_quit()`

<br>

## DSL Version
The version label of the DSL shared library `dsl.so` can be determined by calling `dsl_version_get()`. Version information and release notes can be found on this Respo's Wiki.

**Python Script**
```Python
current_version = dsl_version_get()
```

## Service Return Codes
Most DSL services return values of type `DslReturnType`, return codes with `0` indicating success and `non-0` values indicating failure. All possible return codes are defined as symbolic constants in `DslApi.h` When using Python3, DSL provides a convenience service `dsl_return_value_to_string()` to use there are no "C" equivalent symbolic constants or enum types in Python.  

**Note:** This is the preferred method as the return code values are subject to change

`DSL_RESULT_SUCCESS` is defined in both `DslApi.h` and `dsl.py`. The non-zero Return Codes are defined in `DslApi.h` only.

**DslApi.h**
```C
#define DSL_RESULT_SUCCESS 0

typedef uint DslReturnType
```
**Python Script**
```Python
from dsl import *

retval = dsl_sink_rtsp_new('my-rtsp-sink', 8050, 554, DSL_CODEC_H265, 200000, 0)

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

        # On first occurence of some object of interest, start streaming to file.
        if frame_meta.source_id = some_id_of_interest:
        
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
callback typedef for a client batch meta handler function. Once added to a Component, the function will be called when the component receives a batch meta buffer from its sink or src pad. Functions of this type are added to a component by calling the `dsl_<component-type>_batch_meta_handler_add` service, and removed with the corresponding `dsl_<component-type>_batch_meta_handler_removed` service.

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
* [Tiler](/docs/api-tiler.md)
* [Sink](docs/api-sink.md)
* [Component](/docs/api-component.md)
