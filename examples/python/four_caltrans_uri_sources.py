import sys
sys.path.insert(0, "../../")
import time

from dsl import *

DSL_RETURN_SUCCESS = 0

# Filespecs for the Primary GIE
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b4_fp16.engine'

source_uri1 = "http://wzmedia.dot.ca.gov/D4/E580_at_Grand_Lakeshore.stream/playlist.m3u8"
source_uri2 = "http://wzmedia.dot.ca.gov/D4/E92_JWO_Foster_City_Bl.stream/playlist.m3u8"
source_uri3 = "http://wzmedia.dot.ca.gov/D4/N17_at_Lark_OFR.stream/playlist.m3u8"
source_uri4 = "http://wzmedia.dot.ca.gov/D4/N1_at_Presidio_Tunnel.stream/playlist.m3u8"

while True:

    # Create 4 new URI Streaming Sources
    retval = dsl_source_uri_new('uri-source1', source_uri1, False, 0, 0, 0)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    retval = dsl_source_uri_new('uri-source2', source_uri2, False, 0, 0, 0)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    retval = dsl_source_uri_new('uri-source3', source_uri3, False, 0, 0, 0)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    retval = dsl_source_uri_new('uri-source4', source_uri4, False, 0, 0, 0)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    # New Primary GIE using the filespecs above with interval = 0
    retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 0)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New KTL Tracker, setting width and height of bounding box
    retval = dsl_tracker_ktl_new('ktl-tracker', 480, 272)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Tiled Display, setting width and height, use default cols/rows set by source count
    retval = dsl_tiler_new('tiler', 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New OSD with clock enabled... using default values.
    retval = dsl_osd_new('on-screen-display', False)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
    retval = dsl_sink_overlay_new('overlay-sink', 100, 100, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Pipeline to use with the above components
    retval = dsl_pipeline_new('simple-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break

    # Add all the components to our pipeline
    retval = dsl_pipeline_component_add_many('simple-pipeline', 
        ['uri-source1', 'uri-source2', 'uri-source3', 'uri-source4', 'primary-gie', 'tiler', 'on-screen-display', 'overlay-sink', None])
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
