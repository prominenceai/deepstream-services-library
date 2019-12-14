from ctypes import *

_dsl = CDLL('dsl-lib.so')

DSL_RETURN_SUCCESS = 0

DSL_PAD_SINK = 0
DSL_PAD_SRC = 1

##
## Callback Typedefs
##
META_BATCH_HANDLER = CFUNCTYPE(c_bool, c_void_p, c_void_p)
DISPLAY_EVENT_HANDLER = CFUNCTYPE(c_bool, c_uint, c_uint, c_void_p)

##
## dsl_source_csi_new()
##
_dsl.dsl_source_csi_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_source_csi_new.restype = c_uint
def dsl_source_csi_new(name, width, height, fps_n, fps_d):
    global _dsl
    result =_dsl.dsl_source_csi_new(name, width, height, fps_n, fps_d)
    return int(result)
#print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))

##
## dsl_source_uri_new()
##
_dsl.dsl_source_uri_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint]
_dsl.dsl_source_uri_new.restype = c_uint
def dsl_source_uri_new(name, uri, cudadec_mem_type, intra_decode, drop_frame_interval):
    global _dsl
    result = _dsl.dsl_source_uri_new(name, uri, cudadec_mem_type, intra_decode, drop_frame_interval)
    return int(result)
#print(dsl_source_uri_new("uri-source", "../../test/streams/sample_1080p_h264.mp4", 0, 0, 0))

##
## dsl_source_is_live()
##
_dsl.dsl_source_is_live.argtypes = [c_wchar_p]
_dsl.dsl_source_is_live.restype = c_bool
def dsl_source_is_live(name):
    global _dsl
    result = _dsl.dsl_source_is_live(name)
    return bool(result)
#print(dsl_source_is_live("uri-source"))

##
## dsl_source_get_num_in_use()
##
_dsl.dsl_source_get_num_in_use.restype = c_uint
def dsl_source_get_num_in_use():
    global _dsl
    result = _dsl.dsl_source_get_num_in_use()
    return int(result)
#print(dsl_source_get_num_in_use())

##
## dsl_source_get_num_in_use_max()
##
_dsl.dsl_source_get_num_in_use_max.restype = c_uint
def dsl_source_get_num_in_use_max():
    global _dsl
    result = _dsl.dsl_source_get_num_in_use_max()
    return int(result)
#print(dsl_source_get_num_in_use_max())

##
## dsl_source_set_num_in_use_max()
##
_dsl.dsl_source_set_num_in_use_max.argtypes = [c_uint]
def dsl_source_set_num_in_use_max(max):
    global _dsl
    result = _dsl.dsl_source_set_num_in_use_max(max)
dsl_source_set_num_in_use_max(20)
#print(dsl_source_get_num_in_use_max())

##
## dsl_gie_primary_new()
##
_dsl.dsl_gie_primary_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_uint]
_dsl.dsl_gie_primary_new.restype = c_uint
def dsl_gie_primary_new(name, infer_config_file, model_engine_file, interval):
    global _dsl
    result = _dsl.dsl_gie_primary_new(name, infer_config_file, model_engine_file, interval)
    return int(result)
#print(dsl_gie_primary_new("primary-gie", "../../test/configs/config_infer_primary_nano.txt", 
#    "../../test/models/Primary_Detector_Nano/resnet10.caffemodel", 0, 0))

##
## dsl_gie_secondary_new()
##
_dsl.dsl_gie_secondary_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_gie_secondary_new.restype = c_uint
def dsl_gie_secondary_new(name, infer_config_file, model_engine_file, infer_on_gie_name):
    global _dsl
    result = _dsl.dsl_gie_secondary_new(name, infer_config_file, model_engine_file, infer_on_gie_name)
    return int(result)

##
## dsl_tracker_ktl_new()
##
_dsl.dsl_tracker_ktl_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_tracker_ktl_new.restype = c_uint
def dsl_tracker_ktl_new(name, width, height):
    global _dsl
    result = _dsl.dsl_tracker_ktl_new(name, width, height)
    return int(result)
#print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))

##
## dsl_tracker_iou_new()
##
_dsl.dsl_tracker_iou_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint]
_dsl.dsl_tracker_iou_new.restype = c_uint
def dsl_tracker_iou_new(name, config_file, width, height):
    global _dsl
    result = _dsl.dsl_tracker_iou_new(name, config_file, width, height)
    return int(result)
#print(dsl_tracker_iou_new("iou-tracker", "./test/configs/iou_config.txt", 300, 150))

##
## dsl_tracker_max_dimensions_get()
##
_dsl.dsl_tracker_max_dimensions_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_tracker_max_dimensions_get.restype = c_uint
def dsl_tracker_max_dimensions_get(name):
    global _dsl
    p_max_width = POINTER(c_uint)
    p_max_height = POINTER(c_uint)
    u_max_width = c_uint(0)
    u_max_height = c_uint(0)
    result = _dsl.dsl_tracker_max_dimensions_get(name, p_max_width(u_max_width), p_max_height(u_max_height))
    return int(result), u_max_width.value, u_max_height.value 

#print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))
#print(dsl_tracker_max_dimensions_get("ktl-tracker",))

##
## dsl_tracker_max_dimensions_set()
##
_dsl.dsl_tracker_max_dimensions_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_tracker_max_dimensions_set.restype = c_uint
def dsl_tracker_max_dimensions_set(name, max_width, max_height):
    global _dsl
    result = _dsl.dsl_tracker_max_dimensions_set(name, max_width, max_height)
    return int(result)

##
## dsl_tracker_batch_meta_handler_add()
##
_dsl.dsl_tracker_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, META_BATCH_HANDLER, c_void_p]
_dsl.dsl_tracker_batch_meta_handler_add.restype = c_uint
def dsl_tracker_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = META_BATCH_HANDLER(handler)
    print(meta_handler)
    result = _dsl.dsl_tracker_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

#def mb_handler(buffer, user_data):
#    print(buffer)
#    print(user_data)
    
#print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))
#print(dsl_tracker_batch_meta_handler_add("ktl-tracker", mb_handler, None))

##
## dsl_tracker_batch_meta_handler_remove()
##
_dsl.dsl_tracker_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_tracker_batch_meta_handler_remove.restype = c_uint
def dsl_tracker_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_tracker_batch_meta_handler_remove(name, pad)
    return int(result)
#print(dsl_tracker_batch_meta_handler_remove("ktl-tracker"))

##
## dsl_osd_new()
##
_dsl.dsl_osd_new.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_osd_new.restype = c_uint
def dsl_osd_new(name, is_clock_enabled):
    global _dsl
    result =_dsl.dsl_osd_new(name, is_clock_enabled)
    return int(result)
#print(dsl_osd_new("on-screen-display", False))

##
## dsl_osd_batch_meta_handler_add()
##
_dsl.dsl_osd_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, META_BATCH_HANDLER, c_void_p]
_dsl.dsl_osd_batch_meta_handler_add.restype = c_uint
def dsl_osd_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = META_BATCH_HANDLER(handler)
    print(meta_handler)
    result = _dsl.dsl_osd_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

#def mb_handler(buffer, user_data):
#    print(buffer)
#    print(user_data)
    
#print(dsl_osd_new("on-screen-display", True))
#print(dsl_osd_batch_meta_handler_add("on-screen-display", mb_handler, None))

##
## dsl_osd_batch_meta_handler_remove()
##
_dsl.dsl_osd_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_osd_batch_meta_handler_remove.restype = c_uint
def dsl_osd_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_osd_batch_meta_handler_remove(name, pad)
    return int(result)
#print(dsl_osd_batch_meta_handler_remove("on-screen-display"))

##
## dsl_display_new()
##
_dsl.dsl_display_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_display_new.restype = c_uint
def dsl_display_new(name, width, height):
    global _dsl
    result =_dsl.dsl_display_new(name, width, height)
    return int(result)
#print(dsl_display_new("tiled-display", 1280, 720))

##
## dsl_display_batch_meta_handler_add()
##
_dsl.dsl_display_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, META_BATCH_HANDLER, c_void_p]
_dsl.dsl_display_batch_meta_handler_add.restype = c_uint
def dsl_display_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = META_BATCH_HANDLER(handler)
    print(meta_handler)
    result = _dsl.dsl_display_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

#def mb_handler(buffer, user_data):
#    print(buffer)
#    print(user_data)
    
#print(dsl_display_new("tiled-display", True))
#print(dsl_display_batch_meta_handler_add("tiled-display", mb_handler, None))

##
## dsl_display_batch_meta_handler_remove()
##
_dsl.dsl_display_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_display_batch_meta_handler_remove.restype = c_uint
def dsl_display_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_display_batch_meta_handler_remove(name, pad)
    return int(result)
#print(dsl_display_batch_meta_handler_remove("tiled-display"))

##
## dsl_sink_overlay_new()
##
_dsl.dsl_sink_overlay_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_overlay_new.restype = c_uint
def dsl_sink_overlay_new(name, offsetX, offsetY, width, height):
    global _dsl
    result =_dsl.dsl_sink_overlay_new(name, offsetX, offsetY, width, height)
    return int(result)
#print(dsl_sink_overlay_new("overlay-sink", 0, 0, 1280, 720))

##
## dsl_component_delete()
##
_dsl.dsl_component_delete.argtypes = [c_wchar_p]
_dsl.dsl_component_delete.restype = c_uint
def dsl_component_delete(name):
    global _dsl
    result =_dsl.dsl_component_delete(name)
    return int(result)
#print(dsl_component_delete("tiled-display"))

##
## dsl_component_delete_many()
##
#_dsl.dsl_component_delete_many.argtypes = [Array]
_dsl.dsl_component_delete_many.restype = c_uint
def dsl_component_delete_many(components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_component_delete_many(arr)
    return int(result)
#print(dsl_component_delete_many(["on-screen-display", "primary-gie", None]))

##
## dsl_component_delete_all()
##
_dsl.dsl_component_delete_all.restype = c_uint
def dsl_component_delete_all():
    global _dsl
    result =_dsl.dsl_component_delete_all()
    return int(result)
#print(dsl_component_delete_all())

##
## dsl_component_list_size()
##
_dsl.dsl_component_list_size.restype = c_uint
def dsl_component_list_size():
    global _dsl
    result =_dsl.dsl_component_list_size()
    return int(result)
#print(dsl_component_list_size())

##
## dsl_pipeline_new()
##
_dsl.dsl_pipeline_new.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_new.restype = c_uint
def dsl_pipeline_new(name):
    global _dsl
    result =_dsl.dsl_pipeline_new(name)
    return int(result)
#print(dsl_pipeline_new("pipeline-1"))

##
## dsl_pipeline_new_many()
##
#_dsl.dsl_pipeline_new_many.argtypes = []
_dsl.dsl_pipeline_new_many.restype = c_uint
def dsl_pipeline_new_many(pipelines):
    global _dsl
    arr = (c_wchar_p * len(pipelines))()
    arr[:] = pipelines
    result =_dsl.dsl_pipeline_new_many(arr)
    return int(result)
#print(dsl_pipeline_new_many(["pipeline-2", "pipeline-3", "pipeline-4", None]))

##
## dsl_pipeline_delete()
##
_dsl.dsl_pipeline_delete.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_delete.restype = c_uint
def dsl_pipeline_delete(name):
    global _dsl
    result =_dsl.dsl_pipeline_delete(name)
    return int(result)

##
## dsl_pipeline_delete_many()
##
#_dsl.dsl_component_delete_many.argtypes = [Array]
_dsl.dsl_pipeline_delete_many.restype = c_uint
def dsl_pipeline_delete_many(pipelines):
    global _dsl
    arr = (c_wchar_p * len(pipelines))()
    arr[:] = pipelines
    result =_dsl.dsl_pipeline_delete_many(arr)
    return int(result)
#print(dsl_pipeline_delete_many(["pipeline-2", "pipeline-3", None]))

##
## dsl_pipeline_delete_all()
##
_dsl.dsl_pipeline_delete_all.restype = c_uint
def dsl_pipeline_delete_all():
    global _dsl
    result =_dsl.dsl_pipeline_delete_all()
    return int(result)
#print(dsl_pipeline_delete_all())

##
## dsl_pipeline_list_size()
##
_dsl.dsl_pipeline_list_size.restype = c_uint
def dsl_pipeline_list_size():
    global _dsl
    result =_dsl.dsl_pipeline_list_size()
    return int(result)
#print(dsl_pipeline_list_size())

##
## dsl_pipeline_component_add()
##
_dsl.dsl_pipeline_component_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_component_add.restype = c_uint
def dsl_pipeline_component_add(pipeline, component):
    global _dsl
    result =_dsl.dsl_pipeline_component_add(pipeline, component)
    return int(result)
#print(dsl_display_new("tiled-display", 1280, 720))
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_component_add("pipeline-1", "tiled-display"))

##
## dsl_pipeline_component_add_many()
##
#_dsl.dsl_pipeline_component_add_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_component_add_many.restype = c_uint
def dsl_pipeline_component_add_many(pipeline, components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_pipeline_component_add_many(pipeline, arr)
    return int(result)
#print(dsl_display_new("tiled-display-2", 1280, 720))
#print(dsl_pipeline_new("pipeline-2"))
#print(dsl_pipeline_component_add_many("pipeline-2", ["tiled-display-2", None]))

##
## dsl_pipeline_pause()
##
_dsl.dsl_pipeline_pause.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_pause.restype = c_uint
def dsl_pipeline_pause(name):
    global _dsl
    result =_dsl.dsl_pipeline_pause(name)
    return int(result)
#print(dsl_pipeline_pause("pipeline-1"))

##
## dsl_pipeline_play()
##
_dsl.dsl_pipeline_play.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_play.restype = c_uint
def dsl_pipeline_play(name):
    global _dsl
    result =_dsl.dsl_pipeline_play(name)
    return int(result)
#print(dsl_pipeline_play("pipeline-1"))

##
## dsl_pipeline_stop()
##
_dsl.dsl_pipeline_stop.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_stop.restype = c_uint
def dsl_pipeline_stop(name):
    global _dsl
    result =_dsl.dsl_pipeline_stop(name)
    return int(result)
#print(dsl_pipeline_stop("pipeline-1"))

##
## dsl_pipeline_dump_to_dot()
##
_dsl.dsl_pipeline_dump_to_dot.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_dump_to_dot.restype = c_uint
def dsl_pipeline_dump_to_dot(pipeline, filename):
    global _dsl
    result =_dsl.dsl_pipeline_dump_to_dot(pipeline, filename)
    return int(result)
#print(dsl_pipeline_dump_to_dot("pipeline-1", "dot-file-name"))

##
## dsl_pipeline_dump_to_dot_with_ts()
##
_dsl.dsl_pipeline_dump_to_dot_with_ts.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_dump_to_dot_with_ts.restype = c_uint
def dsl_pipeline_dump_to_dot_with_ts(pipeline, filename):
    global _dsl
    result =_dsl.dsl_pipeline_dump_to_dot_with_ts(pipeline, filename)
    return int(result)
#print(dsl_pipeline_dump_to_dot_with_ts("pipeline-1", "dot-file-name"))

##
## dsl_main_loop_run()
##
def dsl_main_loop_run():
    global _dsl
    result =_dsl.dsl_main_loop_run()
