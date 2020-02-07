# DSL Overview
The core function of the DeepStream Services Library (DSL) is to provide a simple and intuitive API for building, playing, and dynamically modifying Nvidia DeepStream Pipeiles; modifications made (1) based on the results of the realtime video analysis (2) by the application user through external input. Examples of each:
1. programatically adding a [File Sink](/docs/api-sinks.md) based on the occurence of specific objects detected.
2. interactively resizing stream and window dimensions for viewing control

The general approach to using DSL is to
1. Create one or more uniquely named DeepStream [Pipelines](/docs/api-pipeline.md)
2. Create a number of uniquely named Pipeline [Components](/docs/api-reference-list.md) with desired attributes
3. Define and add one or more [Client callback functions](/docs/api-pipeline.md#client-callback-typedefs) (optional)
4. Add the Components to the Pipeline(s)
5. Play the Pipeline(s) and start/joint the main execution loop.

Using Python for example, the above can be written as

```Python
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

# new On-Screen Display for inference visulization - bounding boxes and labels - clocks disabled (False)
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
 dsl_main_loop_run()
 ```

## Pipeline Components

![DSL Components](/Images/dsl-components.png)

## Streaming Sources
Streaming sources are the head component(s) for all Pipelines and all Pipelines must have at least one Source, among others components, before they can transition to a state of Playing. All Pipelines have the ability to multiplex multiple streams - using their own Stream-Muxer - as long as the Sources are of the same play-type; live vs. non-live with the ability to Pause. 

There are currently four types of Source components, two live connected Camera Sources:
* Camera Serial Interface (CSI) Source
* Universal Serial Bus (USB) Source

And two decode Sources that support both live and non-live streams.
* Universal Resource Identifier (URI) Source
* Real-time Streaming Protocol (RTSP) Source

All Sources have dimensions - width and height in pixels - and frame-rates expressed as a fractional numerator and denominator.  A [Dewarper Componet](/docs/api-dewarper.md) (not show in the image above) capabile of dewarping 360 degree camera streams can be added to the URI and RTSP decode sources. The decode sources supports multiple codec formats, including OpenMAX, H.264, H.265, .png, and .jpeg.

A Pipeline's Stream-Muxer has settable output dimensions with a decoded output stream that is ready to infer on.

See the [Source API](/docs/api-source.md) reference section for more information.

## Primary and Secondary Inference Engines
Nvidia's GStreamer Inference Engines (GIEs) use trained models to classify data to “infer” a result; person, dog, car?. A Pipeline may have at most one Primary Inference Engine (PGIE) - with a specified set of classification labels to infer-with - and multiple Secondary Inference Engines (SGIEs) that can Infer-on the output of either the Primary or other Secondary GIEs. Although optional, a Primary Inference Engine is required when adding a Multi-Object Tracker, Secondary Inference Engines, and/or On-Screen-Displays to a Pipeline. Both Primary and Secondary Inference Engines require a model-engine file and configuration file. 

After creation, GIEs can be updated to:
* use new model-engine and config file
* use a different inference interval, or GIE to infer on 
* to enable/disable output of bounding-box frame and label data to text file in KITTI format for [evaluating object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark)
* to enable/disable raw layer information to binary file.

In additioan, with Primary GIEs, applications can
* add/remove `batch-meta-handler` callback functions [see below](#batch-meta-hanler callback services)
* enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Primary and Secondary GIE API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove `batch-meta-handler` callback functions. [see below](#batch-meta-hanler callback services)

Tracker components are optional and a Pipeline can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more information.

## On-Screen Displays
On-Scrren Display (OSD) components highlight detected objects with colored bounding boxes, labels and clocks. Possitional offsets, colors and fonts can all be set and updated. A `batch-meta-handler` callback function, added to the input (sink pad) of the OSD, enables clients to add custom meta data for display.

OSDs are optional and a Pipeline can have at most one.See the [On-Screen Display API](/docs/api-osd.md) reference section for more information. 

## Multi-Source Tiler
Tiler components 

See the [Multi-Source Tiler API](/docs/api-tiler.md) reference section for more information.

## Pipeline Sinks
Sinks, as the end components in the Pipeline, are used to either render the Streaming media or to stream encoded data as a server or to file. All Pipelines require at least one Sink Component inorder to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case were the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently five types of Sink Components that can be added.

1. Overlay Render Sink
2. X11/EGL Window Sink
3. Media Container File Sink
4. RTSP Server Sink
5. Fake Sink

See the [Sink API](/docs/api-sink.md) reference section for more information.

