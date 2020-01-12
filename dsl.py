from ctypes import *

_dsl = CDLL('dsl-lib.so')

DSL_RETURN_SUCCESS = 0

DSL_PAD_SINK = 0
DSL_PAD_SRC = 1

DSL_RTP_TCP = 4
DSL_RTP_ALL = 7

DSL_CODEC_H264 = 0
DSL_CODEC_H265 = 1
DSL_CODEC_MPEG4 = 2

DSL_MUXER_MPEG4 = 0
DSL_MUXER_MK4 = 1

##
## Pointer Typedefs
##
DSL_UINT_P = POINTER(c_uint)
DSL_BOOL_P = POINTER(c_bool)

##
## Callback Typedefs
##
DSL_META_BATCH_HANDLER = CFUNCTYPE(c_bool, c_void_p, c_void_p)
DSL_STATE_CHANGE_LISTENER = CFUNCTYPE(None, c_uint, c_uint, c_void_p)
DSL_EOS_LISTENER = CFUNCTYPE(None, c_void_p)
DSL_XWINDOW_KEY_EVENT_HANDLER = CFUNCTYPE(None, c_wchar_p, c_void_p)
DSL_XWINDOW_BUTTON_EVENT_HANDLER = CFUNCTYPE(None, c_uint, c_uint, c_void_p)
DSL_XWINDOW_DELETE_EVENT_HANDLER = CFUNCTYPE(None, c_void_p)

callbacks = []
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
_dsl.dsl_source_uri_new.argtypes = [c_wchar_p, c_wchar_p, c_bool, c_uint, c_uint, c_uint]
_dsl.dsl_source_uri_new.restype = c_uint
def dsl_source_uri_new(name, uri, is_live, cudadec_mem_type, intra_decode, drop_frame_interval):
    global _dsl
    result = _dsl.dsl_source_uri_new(name, uri, is_live, cudadec_mem_type, intra_decode, drop_frame_interval)
    return int(result)
#print(dsl_source_uri_new("uri-source", "../../test/streams/sample_1080p_h264.mp4", false, 0, 0, 0))

##
## dsl_source_rtsp_new()
##
_dsl.dsl_source_rtsp_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_source_rtsp_new.restype = c_uint
def dsl_source_rtsp_new(name, uri, protocol, cudadec_mem_type, intra_decode, drop_frame_interval):
    global _dsl
    result = _dsl.dsl_source_rtsp_new(name, uri, protocol, cudadec_mem_type, intra_decode, drop_frame_interval)
    return int(result)
#print(dsl_source_uri_new("rtsp-source", "???????", DSL_RTP_ALL, 0, 0, 0))

##
## dsl_source_dimensions_get()
##
_dsl.dsl_source_dimensions_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_source_dimensions_get.restype = c_uint
def dsl_source_dimensions_get(name):
    global _dsl
    width = c_uint(0)
    height = c_uint(0)
    result = _dsl.dsl_source_dimensions_get(name, DSL_UINT_P(width), DSL_UINT_P(height))
    return int(result), width.value, height.value 
#print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
#print(dsl_source_dimensions_get("csi-source"))

##
## dsl_source_frame_rate_get()
##
_dsl.dsl_source_frame_rate_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_source_frame_rate_get.restype = c_uint
def dsl_source_frame_rate_get(name):
    global _dsl
    fps_n = c_uint(0)
    fps_d = c_uint(0)
    result = _dsl.dsl_source_frame_rate_get(name, DSL_UINT_P(fps_n), DSL_UINT_P(fps_d))
    return int(result), fps_n.value, fps_d.value 
#print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
#print(dsl_source_frame_rate_get("csi-source"))

##
## dsl_source_sink_add()
##
_dsl.dsl_source_sink_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_source_sink_add.restype = c_uint
def dsl_source_sink_add(source, sink):
    global _dsl
    result = _dsl.dsl_source_sink_add(source, sink)
    return int(result)
    
##
## dsl_source_sink_remove()
##
_dsl.dsl_source_sink_remove.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_source_sink_remove.restype = c_uint
def dsl_source_sink_remove(source, sink):
    global _dsl
    result = _dsl.dsl_source_sink_remove(source, sink)
    return int(result)
# *** move to end of file (below sink new) for testing    
#print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
#print(dsl_sink_window_new("overlay-sink", 0, 0, 1280, 720))
#print(dsl_source_sink_add("csi-source", "overlay-sink"))
#print(dsl_source_sink_remove("csi-source", "overlay-sink"))

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
## dsl_gie_primary_batch_meta_handler_add()
##
_dsl.dsl_gie_primary_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_gie_primary_batch_meta_handler_add.restype = c_uint
def dsl_gie_primary_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    result = _dsl.dsl_gie_primary_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

#def mb_handler(buffer, user_data):
#    print(buffer)
#    print(user_data)
    
#print(dsl_gie_primary_new("primary-gie", "../../test/configs/config_infer_primary_nano.txt", 
#    "../../test/models/Primary_Detector_Nano/resnet10.caffemodel", 0, 0))
#print(dsl_gie_primary_batch_meta_handler_add("ktl-tracker", mb_handler, None))

##
## dsl_gie_primary_batch_meta_handler_remove()
##
_dsl.dsl_gie_primary_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_gie_primary_batch_meta_handler_remove.restype = c_uint
def dsl_gie_primary_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_gie_primary_batch_meta_handler_remove(name, pad)
    return int(result)
#print(dsl_gie_primary_batch_meta_handler_remove("ktl-tracker"))


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
    max_width = c_uint(0)
    max_height = c_uint(0)
    result = _dsl.dsl_tracker_max_dimensions_get(name, DSL_UINT_P(max_width), DSL_UINT_P(max_height))
    return int(result), max_width.value, max_height.value 

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
_dsl.dsl_tracker_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_tracker_batch_meta_handler_add.restype = c_uint
def dsl_tracker_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
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
_dsl.dsl_osd_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_osd_batch_meta_handler_add.restype = c_uint
def dsl_osd_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
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
## dsl_tiler_new()
##
_dsl.dsl_tiler_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_tiler_new.restype = c_uint
def dsl_tiler_new(name, width, height):
    global _dsl
    result =_dsl.dsl_tiler_new(name, width, height)
    return int(result)
#print(dsl_tiler_new("tiler", 1280, 720))

##
## dsl_tiler_batch_meta_handler_add()
##
_dsl.dsl_tiler_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_tiler_batch_meta_handler_add.restype = c_uint
def dsl_tiler_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    result = _dsl.dsl_tiler_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

#def mb_handler(buffer, user_data):
#    print(buffer)
#    print(user_data)
    
#print(dsl_tiler_new("tiler", True))
#print(dsl_tiler_batch_meta_handler_add("tiler", mb_handler, None))

##
## dsl_tiler_batch_meta_handler_remove()
##
_dsl.dsl_tiler_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_tiler_batch_meta_handler_remove.restype = c_uint
def dsl_tiler_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_tiler_batch_meta_handler_remove(name, pad)
    return int(result)
#print(dsl_tiler_batch_meta_handler_remove("tiler"))

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
## dsl_sink_window_new()
##
_dsl.dsl_sink_window_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_window_new.restype = c_uint
def dsl_sink_window_new(name, offsetX, offsetY, width, height):
    global _dsl
    result =_dsl.dsl_sink_window_new(name, offsetX, offsetY, width, height)
    return int(result)
#print(dsl_sink_window_new("overlay-sink", 0, 0, 1280, 720))

##
## dsl_sink_file_new()
##
_dsl.dsl_sink_file_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_file_new.restype = c_uint
def dsl_sink_file_new(name, filepath, codec, mutex, bitrate, interval):
    global _dsl
    result =_dsl.dsl_sink_file_new(name, filepath, codec, mutex, bitrate, interval)
    return int(result)
#print(dsl_sink_file_new("file-sink", "./output.mp4", DSL_CODEC_H265, DSL_MUXER_MPEG4, 2000000, 1))

##
## dsl_component_delete()
##
_dsl.dsl_component_delete.argtypes = [c_wchar_p]
_dsl.dsl_component_delete.restype = c_uint
def dsl_component_delete(name):
    global _dsl
    result =_dsl.dsl_component_delete(name)
    return int(result)
#print(dsl_component_delete("tiler"))

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
#print(dsl_tiler_new("tiler", 1280, 720))
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_component_add("pipeline-1", "tiler"))

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
#print(dsl_tiler_new("tiler-2", 1280, 720))
#print(dsl_pipeline_new("pipeline-2"))
#print(dsl_pipeline_component_add_many("pipeline-2", ["tiler-2", None]))

##
## dsl_pipeline_streammux_batch_properties_get()
##
_dsl.dsl_pipeline_streammux_batch_properties_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_pipeline_streammux_batch_properties_get.restype = c_uint
def dsl_pipeline_streammux_batch_properties_get(name):
    global _dsl
    batch_size = c_uint(0)
    batch_timeout = c_uint(0)
    result = _dsl.dsl_pipeline_streammux_batch_properties_get(name, DSL_UINT_P(batch_size), DSL_UINT_P(batch_timeout))
    return int(result), batch_size.value, batch_timeout.value 

#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_streammux_batch_properties_get("pipeline-1"))

##
## dsl_pipeline_streammux_dimensions_get()
##
_dsl.dsl_pipeline_streammux_dimensions_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_pipeline_streammux_dimensions_get.restype = c_uint
def dsl_pipeline_streammux_dimensions_get(name):
    global _dsl
    width = c_uint(0)
    height = c_uint(0)
    result = _dsl.dsl_pipeline_streammux_dimensions_get(name, DSL_UINT_P(width), DSL_UINT_P(height))
    return int(result), width.value, height.value 

#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_streammux_dimensions_get("pipeline-1"))

##
## dsl_pipeline_streammux_dimensions_set()
##
_dsl.dsl_pipeline_streammux_dimensions_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_pipeline_streammux_dimensions_set.restype = c_uint
def dsl_pipeline_streammux_dimensions_set(name, width, height):
    global _dsl
    result = _dsl.dsl_pipeline_streammux_dimensions_set(name, width, height)
    return int(result)

##
## dsl_pipeline_streammux_padding_get()
##
_dsl.dsl_pipeline_streammux_padding_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_pipeline_streammux_padding_get.restype = c_uint
def dsl_pipeline_streammux_padding_get(name):
    global _dsl
    enabled = c_bool(0)
    result = _dsl.dsl_pipeline_streammux_padding_get(name, DSL_BOOL_P(enabled))
    return int(result), enabled.value

#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_streammux_padding_get("pipeline-1"))

##
## dsl_pipeline_streammux_padding_set()
##
_dsl.dsl_pipeline_streammux_padding_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_pipeline_streammux_padding_set.restype = c_uint
def dsl_pipeline_streammux_padding_set(name, enabled):
    global _dsl
    result = _dsl.dsl_pipeline_streammux_dimensions_set(name, enabled)
    return int(result)

##
## dsl_pipeline_xwindow_dimensions_get()
##
_dsl.dsl_pipeline_xwindow_dimensions_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_pipeline_xwindow_dimensions_get.restype = c_uint
def dsl_pipeline_xwindow_dimensions_get(name):
    global _dsl
    width = c_uint(0)
    height = c_uint(0)
    result = _dsl.dsl_pipeline_xwindow_dimensions_get(name, DSL_UINT_P(width), DSL_UINT_P(height))
    return int(result), width.value, height.value 
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_xwindow_dimensions_get("pipeline-1"))

##
## dsl_pipeline_xwindow_dimensions_set()
##
_dsl.dsl_pipeline_xwindow_dimensions_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_pipeline_xwindow_dimensions_set.restype = c_uint
def dsl_pipeline_xwindow_dimensions_set(name, width, height):
    global _dsl
    result = _dsl.dsl_pipeline_xwindow_dimensions_set(name, width, height)
    return int(result)

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
## dsl_pipeline_state_change_listener_add()
##
_dsl.dsl_pipeline_state_change_listener_add.argtypes = [c_wchar_p, DSL_STATE_CHANGE_LISTENER, c_void_p]
_dsl.dsl_pipeline_state_change_listener_add.restype = c_uint
def dsl_pipeline_state_change_listener_add(name, listener, user_data):
    global _dsl
    client_listener = DSL_STATE_CHANGE_LISTENER(listener)
    print(client_listener)
    callbacks.append(client_listener)
    result = _dsl.dsl_pipeline_state_change_listener_add(name, client_listener, user_data)
    return int(result)
    
##
## dsl_pipeline_state_change_listener_remove()
##
_dsl.dsl_pipeline_state_change_listener_remove.argtypes = [c_wchar_p, DSL_STATE_CHANGE_LISTENER]
_dsl.dsl_pipeline_state_change_listener_remove.restype = c_uint
def dsl_pipeline_state_change_listener_remove(name, listener):
    global _dsl
    client_listener = DSL_STATE_CHANGE_LISTENER(listener)
    print(client_listener)
    result = _dsl.dsl_pipeline_state_change_listener_remove(name, client_listener)
    return int(result)
#def listener(prev_state, new_state, user_data):
#    print(prev_state)
#    print(new_state)
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_state_change_listener_add("pipeline-1", listener, None))
#print(dsl_pipeline_state_change_listener_remove("pipeline-1", listener))

##
## dsl_pipeline_eos_listener_add()
##
_dsl.dsl_pipeline_eos_listener_add.argtypes = [c_wchar_p, DSL_EOS_LISTENER, c_void_p]
_dsl.dsl_pipeline_eos_listener_add.restype = c_uint
def dsl_pipeline_eos_listener_add(name, listener, user_data):
    global _dsl
    client_listener = DSL_EOS_LISTENER(listener)
    print(client_listener)
    callbacks.append(client_listener)
    result = _dsl.dsl_pipeline_eos_listener_add(name, client_listener, user_data)
    return int(result)
    
##
## dsl_pipeline_state_change_listener_remove()
##
_dsl.dsl_pipeline_eos_listener_remove.argtypes = [c_wchar_p, DSL_EOS_LISTENER]
_dsl.dsl_pipeline_eos_listener_remove.restype = c_uint
def dsl_pipeline_eos_listener_remove(name, listener):
    global _dsl
    client_listener = DSL_EOS_LISTENER(listener)
    print(client_listener)
    result = _dsl.dsl_pipeline_eos_listener_remove(name, client_listener)
    return int(result)
#def listener(user_data):
#    print(user_data)
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_eos_listener_add("pipeline-1", listener, None))
#print(dsl_pipeline_eos_listener_remove("pipeline-1", listener))

##
## dsl_pipeline_xwindow_key_event_handler_add()
##
_dsl.dsl_pipeline_xwindow_key_event_handler_add.argtypes = [c_wchar_p, DSL_XWINDOW_KEY_EVENT_HANDLER, c_void_p]
_dsl.dsl_pipeline_xwindow_key_event_handler_add.restype = c_uint
def dsl_pipeline_xwindow_key_event_handler_add(name, handler, user_data):
    global _dsl
    print(handler)
    client_handler = DSL_XWINDOW_KEY_EVENT_HANDLER(handler)
    print(client_handler)
    callbacks.append(client_handler)
    result = _dsl.dsl_pipeline_xwindow_key_event_handler_add(name, client_handler, user_data)
    return int(result)

##
## dsl_pipeline_xwindow_key_event_handler_remove()
##
_dsl.dsl_pipeline_xwindow_key_event_handler_remove.argtypes = [c_wchar_p, DSL_XWINDOW_KEY_EVENT_HANDLER]
_dsl.dsl_pipeline_xwindow_key_event_handler_remove.restype = c_uint
def dsl_pipeline_xwindow_key_event_handler_remove(name, handler):
    global _dsl
    client_handler = DSL_XWINDOW_KEY_EVENT_HANDLER(handler)
    print(client_handler)
    result = _dsl.dsl_pipeline_xwindow_key_event_handler_remove(name, client_handler)
    return int(result)
#def handler(prev_state, new_state, user_data):
#    print(prev_state)
#    print(new_state)
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_xwindow_key_event_handler_add("pipeline-1", handler, None))
#print(dsl_pipeline_xwindow_key_event_handler_remove("pipeline-1", handler))

##
## dsl_pipeline_xwindow_button_event_handler_add()
##
_dsl.dsl_pipeline_xwindow_button_event_handler_add.argtypes = [c_wchar_p, DSL_XWINDOW_BUTTON_EVENT_HANDLER, c_void_p]
_dsl.dsl_pipeline_xwindow_button_event_handler_add.restype = c_uint
def dsl_pipeline_xwindow_button_event_handler_add(name, handler, user_data):
    global _dsl
    print(handler)
    client_handler = DSL_XWINDOW_BUTTON_EVENT_HANDLER(handler)
    print(client_handler)
    callbacks.append(client_handler)
    result = _dsl.dsl_pipeline_xwindow_button_event_handler_add(name, client_handler, user_data)
    return int(result)

##
## dsl_pipeline_xwindow_button_event_handler_remove()
##
_dsl.dsl_pipeline_xwindow_button_event_handler_remove.argtypes = [c_wchar_p, DSL_XWINDOW_BUTTON_EVENT_HANDLER]
_dsl.dsl_pipeline_xwindow_button_event_handler_remove.restype = c_uint
def dsl_pipeline_xwindow_button_event_handler_remove(name, handler):
    global _dsl
    client_handler = DSL_XWINDOW_BUTTON_EVENT_HANDLER(handler)
    print(client_handler)
    result = _dsl.dsl_pipeline_xwindow_button_event_handler_remove(name, client_handler)
    return int(result)
#def handler(prev_state, new_state, user_data):
#    print(prev_state)
#    print(new_state)
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_xwindow_button_event_handler_add("pipeline-1", handler, None))
#print(dsl_pipeline_xwindow_button_event_handler_remove("pipeline-1", handler))

##
## dsl_pipeline_xwindow_delete_event_handler_add()
##
_dsl.dsl_pipeline_xwindow_delete_event_handler_add.argtypes = [c_wchar_p, DSL_XWINDOW_DELETE_EVENT_HANDLER, c_void_p]
_dsl.dsl_pipeline_xwindow_delete_event_handler_add.restype = c_uint
def dsl_pipeline_xwindow_delete_event_handler_add(name, handler, user_data):
    global _dsl
    print(handler)
    client_handler = DSL_XWINDOW_DELETE_EVENT_HANDLER(handler)
    print(client_handler)
    callbacks.append(client_handler)
    result = _dsl.dsl_pipeline_xwindow_delete_event_handler_add(name, client_handler, user_data)
    return int(result)

##
## dsl_pipeline_xwindow_delete_event_handler_remove()
##
_dsl.dsl_pipeline_xwindow_delete_event_handler_remove.argtypes = [c_wchar_p, DSL_XWINDOW_BUTTON_EVENT_HANDLER]
_dsl.dsl_pipeline_xwindow_delete_event_handler_remove.restype = c_uint
def dsl_pipeline_xwindow_delete_event_handler_remove(name, handler):
    global _dsl
    client_handler = DSL_XWINDOW_DELETE_EVENT_HANDLER(handler)
    print(client_handler)
    result = _dsl.dsl_pipeline_xwindow_delete_event_handler_remove(name, client_handler)
    return int(result)
#def handler(prev_state, new_state, user_data):
#    print(prev_state)
#    print(new_state)
#print(dsl_pipeline_new("pipeline-1"))
#print(dsl_pipeline_xwindow_delete_event_handler_add("pipeline-1", handler, None))
#print(dsl_pipeline_xwindow_delete_event_handler_remove("pipeline-1", handler))

##
## dsl_main_loop_run()
##
def dsl_main_loop_run():
    global _dsl
    _dsl.dsl_main_loop_run()

##
## dsl_main_loop_quit()
##
def dsl_main_loop_quit():
    global _dsl
    _dsl.dsl_main_loop_quit()
