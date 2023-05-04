# UHD Input Source Slicing with Non-Maximum Processing

## ***Eairly work in progress (WIP)***

<img src="/Images/0roi_3840x2160_full_frame.png" alt="0 ROIs" width="960">

## Initial Inference Pipeline
![source slicing inference pipeline diagram](/Images/input-source-slicing-1.png)

### Input Source Component
The example uses a [Streaming Image Source](/docs/api-source.md#dsl_source_image_stream_new) to continuously stream the image at a specified frame rate. This allows us to use a multi-object tracker (MOT) to uniquely identify the objects detected in the image.

```Python
# 4K example image is located under "/deepstream-services-library/test/streams"
image_file = "../../test/streams/4K-image.jpg"

# Create a new streaming image source to stream the UHD file at 10 frames/sec
retval = dsl_source_image_stream_new('image-source', file_path=image_file,
    is_live=False, fps_n=10, fps_d=1, timeout=0)
```

### Primary Gst-Inference Engine Component

```Python
# Filespecs for the Primary GIE
primary_infer_config_file = \
    '/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_infer.txt'

# IMPORTANT! ensure that the model-engine was generated with the config from the Preprocessing example
primary_model_engine_file = \
    '/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b3_gpu0_fp16.engine'

# New Primary GIE using the filespecs above with interval = 0
retval = dsl_infer_gie_primary_new('primary-gie', 
    primary_infer_config_file, primary_model_engine_file, 0)
```

### IOU Multi-Object Tracker Component
An IOU Tracker is used to uniquely identify the objects detected. The yaml configuration file provided with the DeepStream distribution is used for this example.

```Python
# Configuration file for the IOU Tracker
tracker_config_file = \
    '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml'
    
# New IOU Tracker, setting width and height of input frame
retval = dsl_tracker_iou_new('iou-tracker', tracker_config_file, 640, 368)
```

### On-Screen Display Component
An On-Screen Display (OSD) is used for visual verification. The display text and bbound box settings are enabled on creation.
```Python
# New OSD with text and bbox display enabled. 
retval = dsl_osd_new('on-screen-display', 
    text_enabled=True, clock_enabled=False, bbox_enabled=True, mask_enabled=False)
```

### Window Sink Component
A Window Sink is used to render the stream downstream of the On-Screen Display. 
```Python
# New Window Sink, 0 x/y offsets - transpose image to HD dimensions for viewing
retval = dsl_sink_window_new('window-sink', 0, 0, width=1920, height=1080)
```

### Inference Pipeline
A Pipeline with a built-in Streammuxer is created with all Components added. The Streammuxer's deminsions are then set to match the UHD source dimensions. 
```Python
# Add all the components to our pipeline
retval = dsl_pipeline_new_component_add_many('pipeline', components=[
    'image-source', 'preprocessor', 'primary-gie', 'iou-tracker',
    'on-screen-display', 'window-sink', None])
    
# **** IMPORTANT! we must update the default streammuxer dimensions for the 4K image
retval = dsl_pipeline_streammux_dimensions_set('pipeline', 
    width=DSL_STREAMMUX_4K_UHD_WIDTH, height= DSL_STREAMMUX_4K_UHD_HEIGHT)
```

### Frame Capture and Print to Console ODE Acions

```Python
# New Capture Frame ODE Action to capture and save the frame to the current directory
# The image will be captured downstream from the On-Screen Display
retval = dsl_ode_action_capture_frame_new('frame-capture-action',
    outdir = './', annotate = False)

## New File action to write the coordinates, dimensions, and confidence for
## object detected in the image - MOT challenge format.
retval = dsl_ode_action_file_new('file-action',
    file_path='./object_data.txt', mode=DSL_WRITE_MODE_TRUNCATE, 
    format=DSL_EVENT_FILE_FORMAT_MOTC, force_flush=False)
```

### Occurrence ODE Triggers

```Python
# Two new Occurrence Triggers, both filtering on PERSON class_id.
# The first trigger with a limit of one, the second with no-limit. 
retval = dsl_ode_trigger_occurrence_new('person-occurrence-trigger-1',
    source=DSL_ODE_ANY_SOURCE, class_id=2, limit=DSL_ODE_TRIGGER_LIMIT_ONE)
retval = dsl_ode_trigger_occurrence_new('person-occurrence-trigger-2',
    source=DSL_ODE_ANY_SOURCE, class_id=2, limit=DSL_ODE_TRIGGER_LIMIT_NONE)

# Add the frame-capture action to the first occurrence trigger
retval = dsl_ode_trigger_action_add('person-occurrence-trigger-1', 
    action='frame-capture-action')
# Add the file-data action to the second occurrence trigger
retval = dsl_ode_trigger_action_add('person-occurrence-trigger-2', 
    action='file-action')    
```

## Object Detection Event (ODE) Pad Probe Handler (PPH)

```Python
# New ODE Handler to handle the ODE Triggers with their Areas and Actions    
retval = dsl_pph_ode_new('ode-handler')

# Add both occurrence triggers
retval = dsl_pph_ode_trigger_add_many('ode-handler', triggers=[
    'person-occurrence-trigger-1', 'person-occurrence-trigger-2', None])
```

## Three 1280 x 720 ROIs with no overlap
![source slicing inference pipeline diagram](/Images/input-source-slicing-2.png)

### Preprocessor Component

```Python
# Preprocessor config file is located under "/deepstream-services-library/test/configs"
preproc_config_file = \
    '../../test/configs/config_preprocess_4k_input_slicing.txt'
    
# New Preprocessor component using the config filespec defined above.
retval = dsl_preproc_new('preprocessor', preproc_config_file)
```

```Python
# **** IMPORTANT! for best performace we explicity set the GIE's batch-size 
# to the number of ROI's defined in the Preprocessor configuraton file.
retval = dsl_infer_batch_size_set('primary-gie', 3)

# **** IMPORTANT! we must set the input-meta-tensor setting to true when
# using the preprocessor, otherwise the GIE will use its own preprocessor.
retval = dsl_infer_gie_tensor_meta_settings_set('primary-gie',
    input_enabled=True, output_enabled=False);
```

## Three 1328 x 747 ROIs with 72 pixels of overlap
![source slicing inference pipeline diagram](/Images/input-source-slicing-3.png)

## Non-Maximum Processing
![source slicing inference pipeline diagram](/Images/input-source-slicing-4.png)

<table>
  <th>
    <td>Detail of Lower Left Corner</td>
  </th>
  <tr>
    <td>Full image</td>
    <td>1440x1280 ROI No Overlap</td>
    <td>1440x1280 ROI 10% Overlap</td>
  </tr>
  <tr>
    <td>6 objects detected</td>
    <td>8 objects detected</td>
    <td>8 objects detected</td>
  </tr>
  <tr>
    <td><img src="/Images/0roi_3840x2160_full_frame_slice_1.png" alt="0 ROIs Lower Left Corner"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_1.png" alt="0 ROIs Lower Center"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_1.png" alt="0 ROIs Lower Right Corner"></td>
  </tr>
</table>

</br>
<table>
  <th>
    <td>Detail of Lower Center</td>
  </th>
  <tr>
    <td>Full image</td>
    <td>1440x1280 ROI No Overlap</td>
    <td>1440x1280 ROI 10% Overlap</td>
  </tr>
  <tr>
    <td>1 object detected</td>
    <td>1 object detected</td>
    <td>1 object detected</td>
  </tr>
  <tr>
    <td><img src="/Images/0roi_3840x2160_full_frame_slice_2.png" alt="0 ROIs Lower Left Corner"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_2.png" alt="0 ROIs Lower Center"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_2.png" alt="0 ROIs Lower Right Corner"></td>
  </tr>
</table>

</br>
<table>
<table>
  <th>
    <td>Detail of Lower Right Corner</td>
  </th>
  <tr>
    <td>Full image</td>
    <td>1440x1280 ROI No Overlap</td>
    <td>1440x1280 ROI 10% Overlap</td>
  </tr>
  <tr>
    <td>3 objects detected, 2 false positives</td>
    <td>4 objects detected, 3 false positives</td>
    <td>4 objects detected, 3 false positives</td>
  </tr>
  <tr>
    <td><img src="/Images/0roi_3840x2160_full_frame_slice_3.png" alt="0 ROIs Lower Left Corner"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_3.png" alt="0 ROIs Lower Center"></td>
    <td><img src="/Images/3roi_1440x1280_no_overlap_slice_3.png" alt="0 ROIs Lower Right Corner"></td>
  </tr>
</table>

