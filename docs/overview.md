# DSL Overview
The core function of the DeepStream Services Library (DSL) is to provide a simple and intuitive API for Applications to build and dynamically modify Nvidia DeepStream Pipeiles; modifications made (1) based on the results of the realtime video analysis (2) by the application user through external input. Examples of each:
1. programatically adding a [File Sink](/docs/api-sinks.md) based on the occurence of object detection.
2. interactively resizing stream and window dimensions for viewing control

The general approach to using DSL is to create one or more uniquely named Pipelines as required by the application, using Python for example

```Python
retval = dsl_pipeline_new('my-pipeline')
```
followed by a set of Components, each with a specific function and purpose. 
```Python
# new Camera Source - setting dimensions and frames-per-second
retval += dsl_source_csi_new('my-source', 1280, 720, 30, 1)

# new Primary Inference Engine - path to model engine and config file, infer-on every frame with interval = 0
retval += dsl_gie_primary_new('my-pgie', path_to_engine_file, path_to_config_file, 0)

# new On-Screen Display for inferrence visulization - bounding boxes and labels - no clock (False)
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

retval = dsl_pipeline_xwindow_delete_event_handler_add("my pipeline", xwindow_delete_event_handler, None)
```
Add the components to the Pipeline.
```Python
# Using a Null terminated list
retval = dsl_pipeline_component_add_many('my-pipeline', ['my-source', 'my-pgie', 'my-osd', 'my-sink', None])
```

Transition the Pipeline to a state of Playing and join the main loop

```Python
 retval = dsl_pipeline_play('pipeline')
 dsl_main_loop_run()
 ```

## Pipeline Components

![DSL Components](/Images/dsl-components.png)

## Streaming Sources
Streaming sources are the head component(s) for all Pipelines, and all Pipelines must have at least one Source (among others components) before they can transition to a state of Playing. All Pipelines have the ability to multiplex multiple streams - using their own Stream-Muxer component - as long as the Sources are of the same Play-Type; live vs. non-live with the ability to Pause. 

There are currently four types of Source components, two live connected Camera Source types:
* Camera Serial Interface (CSI) Source
* Universal Serial Bus (USB) Source

And two decode Source types that support both live and non-live streams.
* Universal Resource Identifier (URI) Source
* Real-time Streaming Protocol (RTSP) Source

All Sources have dimensions - width and height in pixels - and frame-rates expressed as a fractional numerator and denominator.  A [Dewarper Componet](/docs/api-dewarper.md) (not show in the image above) capabile of dewarping 360 degree camera streams can be added to the URI and RTSP decode sources.

A Pipeline's Stream-Muxer has output dimensions that can set, with a decoded stream that is ready to Infer-On.

See the [Source API](/docs/api-source.md) reference section for more information.

## Primary and Secondary Inference Engines
Nvidia's GStreamer Inference Engines (GIEs) use trained models to classify data to “infer” a result; person? dog? car?. A Pipeline may have at most one Primary Inference Engine (PGIE) - with a specified set of object labels to infer-on - with multiple Secondary Inference Engines (SGIEs) that can Infer-on the output of either the Primary or other Secondary GIEs. Although optional, a Primary Inference Engine is required when adding a Multi-Object Tracker, Secondary Inference Engines, and/or On-Screen-Displays to a Pipeline. Both Primary and Secondary Inference Engines require a model-engine file and configuration file with setting that can be overridden at runtime. 

See the [Primary API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers

See the [Tracker API](/docs/api-tracker.md) reference section for more information.

## On-Screen Displays

See the [On-Screen Display API](/docs/api-osd.md) reference section for more information.

## Multi-Source Tiler

See the [Multi-Source Tiler API](/docs/api-tiler.md) reference section for more information.

## Pipeline Sinks
Sinks, as the end components in the Pipeline, are used to either render the Streaming media or to stream encoded data as a server or to file. All Pipelines require at least one Sink Component inorder to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case were the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently five types of Sink Components that can be added.

1. Overlay Render Sink
2. X11/EGL Window Sink
3. Media Container File Sink
4. RTSP Server Sink
5. Fake Sink

See the [Sink API](/docs/api-sink.md) reference section for more information.

