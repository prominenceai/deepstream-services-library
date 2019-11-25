import sys
sys.path.insert(0, "../../")
import time
from dsl import *

# Filespecs for the Primary GIE
inferConfigFile = '../../test/configs/config_infer_primary.txt'
modelEngineFile = '../../test/models/Primary_Detector/resnet10.caffemodel_b1_fp16.engine'

while True:

    # New CSI Live Camera Source
    retval = dsl_source_csi_new('csi-source', 1280, 720, 30, 1)
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
    retval = dsl_pipeline_component_add_many('simple-pipeline', ['csi-source', 'primary-gie', 'tiled-display', 'overlay-sink', None])

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
