# DSL Overview
The core function of the DeepStream Services Library (DSL) is to provide a simple and intuitive API for building, playing, and dynamically modifying Nvidia DeepStream Pipeiles; modifications made (1) based on the results of the realtime video analysis (2) by the application user through external input. An example of each:
1. Programatically adding a [File Sink](/docs/api-sinks.md) based on the occurence of specific objects detected.
2. Interactively resizing stream and window dimensions for viewing control

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
There are seven catagories of Components that can be added to a Pipeline, automatically assembled in the order shown below. Many of the catagories support multiple types and in the cases of Sources, Secondary Inference Engines, and Sinks, multiple types can be added to a single Pipelne. 

![DSL Components](/Images/dsl-components.png)

## Streaming Sources
Streaming sources are the head component(s) for all Pipelines and all Pipelines must have at least one Source, among others components, before they can transition to a state of Playing. All Pipelines have the ability to multiplex multiple streams - using their own Stream-Muxer - as long as all Sources are of the same play-type; live vs. non-live with the ability to Pause. 

There are currently four types of Source components, two live connected Camera Sources:
* Camera Serial Interface (CSI) Source
* Universal Serial Bus (USB) Source

And two decode Sources that support both live and non-live streams.
* Universal Resource Identifier (URI) Source
* Real-time Streaming Protocol (RTSP) Source

All Sources have dimensions - width and height in pixels - and frame-rates expressed as a fractional numerator and denominator.  A [Dewarper Componet](/docs/api-dewarper.md) (not show in the image above) capabile of dewarping 360 degree camera streams can be added to the URI and RTSP decode sources. The decode sources support multiple codec formats, including H.264, H.265, png, and jpeg.

A Pipeline's Stream-Muxer has settable output dimensions with a decoded output stream that is ready to infer on.

See the [Source API](/docs/api-source.md) reference section for more information.

## Primary and Secondary Inference Engines
Nvidia's GStreamer Inference Engines (GIEs), using pre-trained models, classify data to “infer” a result; person, dog, car?. A Pipeline may have at most one Primary Inference Engine (PGIE) - with a specified set of classification labels to infer-with - and multiple Secondary Inference Engines (SGIEs) that can Infer-on the output of either the Primary or other Secondary GIEs. Although optional, a Primary Inference Engine is required when adding a Multi-Object Tracker, Secondary Inference Engines, or On-Screen-Display to a Pipeline.

After creation, GIEs can be updated to:
* Use a new model-engine, config file,  inference interval, and for Secondary GIEs, the GIE to infer on 
* To enable/disable output of bounding-box frame and label data to text file in KITTI format for [evaluating object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark)
* To enable/disable output of raw layer information to binary file.

With Primary GIEs, applications can:
* add/remove `batch-meta-handler` callback functions [see below](#batch-meta-hanler callback services)
* enable/disable raw layer-info output to binary file, one file per layer, per frame.

See the [Primary and Secondary GIE API](/docs/api-gie.md) reference section for more information.

## Multi-Object Trackers
There are two types of streaming Multi-Object Tracker Components.
1. [Kanade–Lucas–Tomasi](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker) (KTL) Feature Tracker
2. [Intersection-Over-Unioun](https://www.researchgate.net/publication/319502501_High-Speed_Tracking-by-Detection_Without_Using_Image_Information_Challenge_winner_IWOT4S) (IOU) High-Frame-Rate Tracker. 

Clients of Tracker components can add/remove `batch-meta-handler` callback functions. [see below](#batch-meta-hanler callback services)

Tracker components are optional and a Pipeline can have at most one. See the [Tracker API](/docs/api-tracker.md) reference section for more information.

## On-Screen Display
On-Scrren Display (OSD) components highlight detected objects with colored bounding boxes, labels and clocks. Possitional offsets, colors and fonts can all be set and updated. A `batch-meta-handler` callback function, added to the input (sink pad) of the OSD, enables clients to add custom meta data for display.

OSDs are optional and a Pipeline can have at most one.See the [On-Screen Display API](/docs/api-osd.md) reference section for more information. 

## Multi-Source Tiler
Tiler components transform the multiplexed streams into a 2D array of tiles, one per Source component. Tilers have dimensions, width and height in pixels, and rows and columns settings that can be updated after creation.

Clients of Tiler components can add/remove `batch-meta-handler` callback functions. [see below](#batch-meta-hanler callback services)

Tiler components are optional and a Pipeline can have at most one. See the [Multi-Source Tiler API](/docs/api-tiler.md) reference section for more information.

## Pipeline Sinks
Sinks, as the end components in the Pipeline, are used to either render the Streaming media or to stream encoded data as a server or to file. All Pipelines require at least one Sink Component inorder to Play. A Fake Sink can be created if the final stream is of no interest and can simply be consumed and dropped. A case were the `batch-meta-data` produced from the components in the Pipeline is the only data of interest. There are currently five types of Sink Components that can be added.

1. Overlay Render Sink
2. X11/EGL Window Sink
3. Media Container File Sink
4. RTSP Server Sink
5. Fake Sink

See the [Sink API](/docs/api-sink.md) reference section for more information.

## Batch Meta Handler Callback Functions
All of the `one-at-most` Pipeline Components - Primary GIEs, Multi-Object Trackers, On-Screen Displays, and Tilers - support the dynamic addition and removal of batch-meta-hanler callback functions. Multiple handlers can be added to the component's Input (sink-pad) and Output (src-pad) streams. Batch-meta-handlers allow applications to monitor and block-on data flowing over the component's pads.

Each batch-meta-handler function is called with a buffer of meta-data for each batch processed. A Pipeline's batch-size is set to the current number of Source Components upstream.

Adding a batch-meta-handler to the sink-pad of an On-Screen Display component, for example, is an ideal point in the stream to monitor, process, and make decisions based on all inference and tracker results based on its location in the stream.

When using Python3, Nvidia's [Python-bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps#python-bindings) are used to process the buffered batch-meta in the handler callback function. The bindings can be downloaded from [here](https://developer.nvidia.com/deepstream-download#python_bindings)

```Python
##
# Callback function to handle batch-meta data
##
def osd_batch_meta_handler_cb(buffer, user_data):

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break    

        # On first occurence of some object of interest, start streaming to file.
        if frame_meta.source_id = some_id_of_interest:
            dsl_pipeline_component_add('my-pipeline', 'my-file-sink')
            
            # return False to self remove.
            return False 

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return True
  
##
# Create a new OSD component and add the batch-meta handler function above to the Sink Pad.
##
retval = dsl_osd_new('my-osd', False)
retval += dsl_osd_batch_meta_handler_add('my-osd', DSL_PAD_SINK, osd_batch_meta_handler_cb, None)

##
# Create a new H.264 File Sink component to be added to the Pipeline by the osd_batch_meta_handler_cb
##
retval += dsl_sink_file_new('my-file-sink', './my-video.mp4', DSL_CODEC_H264, DSL_CONTAINER_MPEG, 200000, 0)

if retval != DSL_RESULT_SUCCESS:
    # Component setup failed

# add the components to the Pipeline and transition to Playing

```

## X11 Window Support
DSL provides X11 Window support for Pipelines that have one or more Window Sinks. An Application can create Windows- using GTK+ for example - and share share them with Pipelines prior to playinging, or let the Pipeline create a sinple Display and Window to use. 

``` Python
# Function to be called on XWindow ButtonPress event
def xwindow_button_event_handler(xpos, ypos, client_data):
    print('button pressed: xpos = ', xpos, ', ypos = ', ypos)
    
## 
# Function to be called on XWindow KeyRelease event - non-live streams
## 
def xwindow_key_event_handler(key_string, client_data):
    print('key released = ', key_string)
    if key_string.upper() == 'P':
        dsl_pipeline_pause('pipeline')
    elif key_string.upper() == 'R':
        dsl_pipeline_play('pipeline')
    elif key_string.upper() == 'Q' or key_string == '':
        dsl_main_loop_quit()
 
## 
# Function to be called on XWindow Delete event
## 
def xwindow_delete_event_handler(client_data):
    print('delete window event')
    dsl_main_loop_quit()

while True:

    ## 
    ## New Pipeline and Window Sink
    ## 
    retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    retval = dsl_pipeline_new('my-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break
    ## 
    ## Add the XWindow event handler functions defined above
    ##
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



