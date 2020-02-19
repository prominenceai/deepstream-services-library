import sys
sys.path.insert(0, "../../")
import time

from dsl import *

# Filespecs for the Primary GIE
inferConfigFile = '../../test/configs/config_infer_primary_nano.txt'
modelEngineFile = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b4_fp16.engine'

# Function to be called on End-of-Stream (EOS) event
def eos_event_listener(client_data):
    print('Pipeline EOS event')
    dsl_main_loop_quit()

def main(args):

    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:

        # Two URI File Sources - using the same file.
        retval = dsl_source_uri_new('uri-source-1', "../../test/streams/sample_1080p_h264.mp4", False, 0, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_uri_new('uri-source-2', "../../test/streams/sample_1080p_h264.mp4", False, 0, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with infer interval
        retval = dsl_gie_primary_new('primary-gie', inferConfigFile, modelEngineFile, 0)
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Demuxer - therefore, OSDs and Sinks are added to Sources, not to the Pipeline.
        retval = dsl_demuxer_new('demuxer')
        if retval != DSL_RETURN_SUCCESS:
            break

        ## 
        # Setup the first Source with its downstream overlay-sink
        ###

        # New Overlay Sink with id, display, depth, x/y offsets and Dimensions
        retval = dsl_sink_overlay_new('overlay-sink', 1, 0, 0, 100, 100, 360, 180)
        if retval != DSL_RETURN_SUCCESS:
            break
            
        retval = dsl_source_sink_add('uri-source-1', 'overlay-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        ## 
        # Setup the Source with its downstream OSD and window-sink
        ###

        retval = dsl_osd_new('on-screen-display', False)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_source_osd_add('uri-source-2', 'on-screen-display')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Window Sink, with x/y offsets and dimensions
        retval = dsl_sink_window_new('window-sink', 0, 0, 720, 360)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        retval = dsl_source_sink_add('uri-source-2', 'window-sink')
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Pipeline to use with the above components
        retval = dsl_pipeline_new('simple-pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the EOS listener callback to the Pipeline
        retval = dsl_pipeline_eos_listener_add('simple-pipeline', eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Add the sources the components to our pipeline
        retval = dsl_pipeline_component_add_many('simple-pipeline', 
            ['uri-source-1', 'uri-source-2', 'primary-gie', 'demuxer', None])
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pipeline_xwindow_dimensions_set('simple-pipeline', 1280, 720)
        if retval != DSL_RETURN_SUCCESS:
            break
        
        # Play the pipeline
        retval = dsl_pipeline_play('simple-pipeline')
        if retval != DSL_RETURN_SUCCESS:
            break

        # Once playing, we can dump the pipeline graph to dot file, which can be converted to an image file for viewing/debugging
        dsl_pipeline_dump_to_dot('simple-pipeline', 'state-playing')

        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    print(retval)

    dsl_pipeline_delete_all()

    dsl_source_sink_remove('uri-source-1', 'overlay-sink')
    dsl_source_sink_remove('uri-source-2', 'window-sink')
    dsl_source_osd_remove('uri-source-2'    )
    dsl_component_delete_all()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
