import sys
sys.path.insert(0, "../../")
import time

from dsl import *

DSL_RETURN_SUCCESS = 0

# Filespecs for the Primary GIE
inferConfigFile = '../../test/configs/config_infer_primary_nano.txt'
modelEngineFile = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel'

while True:

    # New URI File Source
    retval = dsl_source_uri_new('uri-source', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 2)

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # New Primary GIE using the filespecs above, with interval and Id
    retval = dsl_gie_primary_new('primary-gie', inferConfigFile, modelEngineFile, 4, 1)

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # New Tiled Display, setting width and height, use default cols/rows set by source count
    retval = dsl_display_new('tiled-display', 1280, 720)

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
    retval = dsl_sink_overlay_new('overlay-sink', 0, 0, 1280, 720)

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # New Pipeline to use with the above components
    retval = dsl_pipeline_new('simple-pipeline')

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # Add all the components to our pipeline
#    retval = dsl_pipeline_component_add_many('simple-pipeline', ['uri-source', 'primary-gie', 'tiled-display', 'overlay-sink', None])
    retval = dsl_pipeline_component_add_many('simple-pipeline', ['uri-source' , 'tiled-display', 'overlay-sink', None])

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # Play the pipeline
    retval = dsl_pipeline_play('simple-pipeline')

    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # Once playing, we can dump the pipeline graph to dot file, which can be converted to an image file for viewing/debugging
    dsl_pipeline_dump_to_dot('simple-pipeline', "state-playing")

    dsl_main_loop_run()
    break

dsl_pipeline_delete_all()
dsl_component_delete_all()
