import sys
sys.path.insert(0, "../../")
import time

from dsl import *

DSL_RETURN_SUCCESS = 0

# Filespecs for the Primary GIE
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_fp16.engine'

source_uri = '../../test/streams/sample_1080p_h264.mp4'

# Function to be called on every change of Pipeline state
def state_change_listener(prev_state, new_state, client_data):
    print('previous state = ', prev_state, ', new state = ', new_state)

# Function to be called on EOS
def eos_listener(client_data):
    dsl_main_loop_quit()

while True:

    # First new URI File Source
    retval = dsl_source_uri_new('uri-source', source_uri, False, 0, 0, 0)
    if retval != DSL_RETURN_SUCCESS:
        break
        
    # New Primary GIE using the filespecs above with interval = 0
    retval = dsl_gie_primary_new('primary-gie', primary_infer_config_file, primary_model_engine_file, 0)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New KTL Tracker, setting max width and height of input frame
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
    retval = dsl_pipeline_new('pipeline-1')
    if retval != DSL_RETURN_SUCCESS:
        break

    # Add all the components to our pipeline
    retval = dsl_pipeline_component_add_many('pipeline-1', 
        ['uri-source', 'primary-gie', 'ktl-tracker', 'tiled-display', 'on-screen-display', 'window-sink', None])
    if retval != DSL_RETURN_SUCCESS:
        break
    
    # Add the listener callback functions defined above
    retval = dsl_pipeline_state_change_listener_add('pipeline-1', state_change_listener, None)
    if retval != DSL_RETURN_SUCCESS:
        break
    retval = dsl_pipeline_eos_listener_add('pipeline-1', eos_listener, None)
    if retval != DSL_RETURN_SUCCESS:
        break

    # Play the pipeline
    retval = dsl_pipeline_play('pipeline-1')
    if retval != DSL_RETURN_SUCCESS:
        break

    # Wait for the User to Interrupt the script with Ctrl-C
    dsl_main_loop_run()
    retval = DSL_RETURN_SUCCESS
    break

print(retval)

dsl_pipeline_delete_all()
dsl_component_delete_all()
