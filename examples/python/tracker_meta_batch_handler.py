import sys
sys.path.insert(0, "../../")
import time
import math
import pyds
from dsl import *

DSL_RETURN_SUCCESS = 0

# Filespecs for the Primary GIE
primary_infer_config_file = '../../test/configs/config_infer_primary_nano.txt'
primary_model_engine_file = '../../test/models/Primary_Detector_Nano/resnet10.caffemodel_b1_fp16.engine'

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

def tracker_batch_meta_handler_cb(buffer, user_data):

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break    

#        '''
        print("Frame Number is ", frame_meta.frame_num)
        print("Source id is ", frame_meta.source_id)
        print("Batch id is ", frame_meta.batch_id)
        print("Source Frame Width ", frame_meta.source_frame_width)
        print("Source Frame Height ", frame_meta.source_frame_height)
        print("Num object meta ", frame_meta.num_obj_meta)
#        '''

while True:

    # First new URI File Source
    retval = dsl_source_uri_new('uri-source', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 0)
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

    retval, width, height = dsl_tracker_max_dimensions_get('ktl-tracker')
    print(width, height)

    retval = dsl_tracker_batch_meta_handler_add('ktl-tracker', tracker_batch_meta_handler_cb, None)
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
    retval = dsl_sink_overlay_new('overlay-sink', 0, 0, 1280, 720)
    if retval != DSL_RETURN_SUCCESS:
        break

    # New Pipeline to use with the above components
    retval = dsl_pipeline_new('simple-pipeline')
    if retval != DSL_RETURN_SUCCESS:
        break

    # Add all the components to our pipeline
    retval = dsl_pipeline_component_add_many('simple-pipeline', 
        ['uri-source', 'primary-gie', 'ktl-tracker', 'tiled-display', 'on-screen-display', 'overlay-sink', None])
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
