import ctypes
libc = ctypes.CDLL('../../dsl-lib.so')

libc.dsl_source_uri_new('mySource', "../../test/streams/sample_1080p_h264.mp4", 0, 0)
libc.dsl_display_new('myDisplay', 1280, 720)
libc.dsl_sink_overlay_new('mySink', 0, 0, 0, 0)
libc.dsl_pipeline_new('myPipeline')

libc.dsl_pipeline_component_add('myPipeline', 'mySource')
libc.dsl_pipeline_component_add('myPipeline', 'myDisplay')
libc.dsl_pipeline_component_add('myPipeline', 'mySink')

libc.dsl_pipeline_play('myPipeline')
libc.dsl_pipeline_dump_to_dot('myPipeline', "state-playing")
libc.dsl_main_loop_run()
