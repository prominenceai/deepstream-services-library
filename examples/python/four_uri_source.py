import ctypes
import time
libc = ctypes.CDLL('../../dsl-lib.so')

libc.dsl_source_uri_new('mySource1', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 2)
libc.dsl_source_uri_new('mySource2', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 2)
libc.dsl_source_uri_new('mySource3', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 2)
libc.dsl_source_uri_new('mySource4', "../../test/streams/sample_1080p_h264.mp4", 0, 0, 2)

inferConfigFile = '/test/configs/config_infer_primary_nano.txt'
modelEngineFile = './test/models/Primary_Detector_Nano/resnet10.caffemodel'

libc.dsl_gie_primary_new('myPrimaryGie', inferConfigFile, modelEngineFile, 1, 0)
libc.dsl_display_new('myDisplay', 1280, 720)
libc.dsl_sink_overlay_new('mySink', 0, 0, 0, 0)
libc.dsl_pipeline_new('myPipeline')

libc.dsl_pipeline_component_add('myPipeline', 'mySource1')
libc.dsl_pipeline_component_add('myPipeline', 'mySource2')
libc.dsl_pipeline_component_add('myPipeline', 'mySource3')
libc.dsl_pipeline_component_add('myPipeline', 'mySource4')
libc.dsl_pipeline_component_add('myPipeline', 'myPrimaryGie')
libc.dsl_pipeline_component_add('myPipeline', 'myDisplay')
libc.dsl_pipeline_component_add('myPipeline', 'mySink')

libc.dsl_pipeline_play('myPipeline')
libc.dsl_pipeline_dump_to_dot('myPipeline', "state-playing")

libc.dsl_main_loop_run()
