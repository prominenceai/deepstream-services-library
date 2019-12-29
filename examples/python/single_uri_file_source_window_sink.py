import sys
sys.path.insert(0, "../../")
import time

from dsl import *

DSL_RETURN_SUCCESS = 0

# Filespecs for the Primary GIE
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_fp16.engine'

# Filespecs for the Single Secondary GIE
secondary_infer_config_file = '../../test/configs/config_infer_secondary_carcolor.txt';
secondary_model_engine_file = '../../test/models/Secondary_CarColor/resnet18.caffemodel';

source_uri = "../../test/streams/sample_1080p_h264.mp4"

while True:

    # First new URI File Source
    retval = dsl_source_uri_new('uri-source', source_uri, False, 0, 0, 0)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    # New Primary GIE using the filespecs above with interval = 0
    retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 0)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Secondary GIE using the filespecs above set to infer on the Primary GIE 
    retval = dsl_gie_secondary_new('secondary-gie', secondary_infer_config_file, secondary_model_engine_file, 'primary-gie')
    if retval != DSL_RETURN_SUCCESS:
        break

    # New KTL Tracker, setting width and height of bounding box
    retval = dsl_tracker_ktl_new('ktl-tracker', 480, 272)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Tiled Display, setting width and height, use default cols/rows set by source count
    retval = dsl_display_new('tiled-display', 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New OSD with clock enabled... using default values.
    retval = dsl_osd_new('on-screen-display', False)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
    retval = dsl_sink_window_new('window-sink', 0, 0, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Pipeline to use with the above components
    retval = dsl_pipeline_new('simple-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break

    # Add all the components to our pipeline
    retval = dsl_pipeline_component_add_many('simple-pipeline', 
        ['uri-source', 'primary-gie', 'ktl-tracker', 'tiled-display', 'on-screen-display', 'window-sink', None])
    if retval != DSL_RETURN_SUCCESS:
        break

    # Play the pipeline
    retval = dsl_pipeline_play('simple-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break

    # Once playing, we can dump the pipeline graph to dot file, which can be converted to an image file for viewing/debugging
    dsl_pipeline_dump_to_dot('simple-pipeline', "state-playing")

    # Wait for the User to Interrupt the script with Ctrl-C
    dsl_main_loop_run()
    retval = DSL_RETURN_SUCCESS
    break

print(retval)

dsl_pipeline_delete_all()
dsl_component_delete_all()
