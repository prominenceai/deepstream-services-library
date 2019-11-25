
import sys
sys.path.insert(0, "../../")
import time

from dsl import *

DSL_RETURN_SUCCESS = 0

# Filespecs for the Primary GIE
inferConfigFile = '../../test/configs/config_infer_primary_nano.txt'
modelEngineFile = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel'

while True:

    # ***********************************************************
    # Create one source for each Pipeline
    # URI File Source
    
    retval = dsl_source_uri_new('uri-source-1', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 2)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # New CSI Live Camera Source
    retval = dsl_source_csi_new('csi-source-2', 1280, 720, 30, 1)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # ***********************************************************
    # Create one Primary GIE for each Pipeline
    # New Primary GIE using the filespecs above, with interval and Id
    
    retval = dsl_gie_primary_new('primary-gie-1', inferConfigFile, modelEngineFile, 4, 1)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    retval = dsl_gie_primary_new('primary-gie-2', inferConfigFile, modelEngineFile, 4, 1)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # ***********************************************************
    # Create one Tiled Display for each Pipeline
    # New Tiled Display, setting width and height, use default cols/rows set by source count
    
    retval = dsl_display_new('tiled-display-1', 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    retval = dsl_display_new('tiled-display-2', 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # ***********************************************************
    # Create one Sink Overlay for each Pipeline
    # New Overlay Sink, 0 x/y offsets and same dimensions as Tiled Display
    
    retval = dsl_sink_overlay_new('overlay-sink-1', 0, 0, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    retval = dsl_sink_overlay_new('overlay-sink-2', 200, 200, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # ***********************************************************
    # Two new Pipelines to use with the above components
    
    retval = dsl_pipeline_new('pipeline-1')
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    retval = dsl_pipeline_new('pipeline-2')
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break
    
    # ***********************************************************
    # Add all the components - in any order - to each pipeline
    
    retval = dsl_pipeline_component_add_many('pipeline-1', ['uri-source-1' , 'tiled-display-1', 'overlay-sink-1', None])
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break
        
    retval = dsl_pipeline_component_add_many('pipeline-2', ['tiled-display-2', 'overlay-sink-2', 'csi-source-2' , None])
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    # ***********************************************************
    # Play both pipelines
    retval = dsl_pipeline_play('pipeline-1')
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    retval = dsl_pipeline_play('pipeline-2')
    if retval != DSL_RETURN_SUCCESS:
        print(retval)
        break

    dsl_main_loop_run()
    break

dsl_pipeline_delete_all()
dsl_component_delete_all()
