from ctypes import *

_dsl = CDLL('./dsl-lib.so')

DSL_RETURN_SUCCESS = 0

DSL_PAD_SINK = 0
DSL_PAD_SRC = 1

DSL_RTP_TCP = 4
DSL_RTP_ALL = 7

DSL_CUDADEC_MEMTYPE_DEVICE = 0
DSL_CUDADEC_MEMTYPE_PINNED = 1
DSL_CUDADEC_MEMTYPE_UNIFIED = 2

DSL_SOURCE_CODEC_PARSER_H264 = 0
DSL_SOURCE_CODEC_PARSER_H265 = 1

DSL_CODEC_H264 = 0
DSL_CODEC_H265 = 1
DSL_CODEC_MPEG4 = 2

DSL_CONTAINER_MP4 = 0
DSL_CONTAINER_MKV = 1

DSL_STATE_NULL = 1
DSL_STATE_READY = 2
DSL_STATE_PAUSED = 3
DSL_STATE_PLAYING = 4
DSL_STATE_IN_TRANSITION = 5

DSL_CAPTURE_TYPE_OBJECT = 0
DSL_CAPTURE_TYPE_FRAME = 1

DSL_ODE_TRIGGER_LIMIT_NONE = 0
DSL_ODE_TRIGGER_LIMIT_ONE = 1

DSL_ODE_PRE_OCCURRENCE_CHECK = 0
DSL_ODE_POST_OCCURRENCE_CHECK = 1

# Any Source/Class == INT32_MAX
DSL_ODE_ANY_SOURCE = int('7FFFFFFF',16)
DSL_ODE_ANY_CLASS = int('7FFFFFFF',16)

##
## Pointer Typedefs
##
DSL_UINT_P = POINTER(c_uint)
DSL_BOOL_P = POINTER(c_bool)
DSL_WCHAR_PP = POINTER(c_wchar_p)
DSL_DOUBLE_P = POINTER(c_double)

##
## Callback Typedefs
##
DSL_META_BATCH_HANDLER = CFUNCTYPE(c_bool, c_void_p, c_void_p)
DSL_STATE_CHANGE_LISTENER = CFUNCTYPE(None, c_uint, c_uint, c_void_p)
DSL_EOS_LISTENER = CFUNCTYPE(None, c_void_p)
DSL_XWINDOW_KEY_EVENT_HANDLER = CFUNCTYPE(None, c_wchar_p, c_void_p)
DSL_XWINDOW_BUTTON_EVENT_HANDLER = CFUNCTYPE(None, c_uint, c_uint, c_void_p)
DSL_XWINDOW_DELETE_EVENT_HANDLER = CFUNCTYPE(None, c_void_p)
DSL_ODE_HANDLE_OCCURRENCE = CFUNCTYPE(None, c_uint, c_wchar_p, c_void_p, c_void_p, c_void_p, c_void_p)
DSL_ODE_CHECK_FOR_OCCURRENCE = CFUNCTYPE(c_bool, c_void_p, c_void_p, c_void_p, c_void_p)
DSL_SINK_RECORD_CLIENT_LISTNER = CFUNCTYPE(c_void_p, c_void_p, c_void_p)

##
## TODO: CTYPES callback management needs to be completed before any of
## the callback remove wrapper functions will work correctly.
## The below is a simple solution for supporting add functions only.
##
callbacks = []

##
## dsl_display_type_rgba_color_new()
##
_dsl.dsl_display_type_rgba_color_new.argtypes = [c_wchar_p, c_double, c_double, c_double, c_double]
_dsl.dsl_display_type_rgba_color_new.restype = c_uint
def dsl_display_type_rgba_color_new(name, red, green, blue, alpha):
    global _dsl
    result =_dsl.dsl_display_type_rgba_color_new(name, red, green, blue, alpha)
    return int(result)

##
## dsl_display_type_rgba_font_new()
##
_dsl.dsl_display_type_rgba_font_new.argtypes = [c_wchar_p, c_uint, c_wchar_p]
_dsl.dsl_display_type_rgba_font_new.restype = c_uint
def dsl_display_type_rgba_font_new(name, size, color):
    global _dsl
    result =_dsl.dsl_display_type_rgba_font_new(name, size, color)
    return int(result)

##
## dsl_display_type_rgba_text_new()
##
_dsl.dsl_display_type_rgba_text_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_wchar_p, c_bool, c_wchar_p]
_dsl.dsl_display_type_rgba_text_new.restype = c_uint
def dsl_display_type_rgba_text_new(name, text, x_offset, y_offset, font, has_bg_color, bg_color):
    global _dsl
    result =_dsl.dsl_display_type_rgba_text_new(name, text, x_offset, y_offset, font, has_bg_color, bg_color)
    return int(result)

##
## dsl_display_type_rgba_line_new()
##
_dsl.dsl_display_type_rgba_line_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_wchar_p]
_dsl.dsl_display_type_rgba_line_new.restype = c_uint
def dsl_display_type_rgba_line_new(name, x1, y1, x2, y2, width, color):
    global _dsl
    result =_dsl.dsl_display_type_rgba_line_new(name, x1, y1, x2, y2, width, color)
    return int(result)

##
## dsl_display_type_rgba_rectangle_new()
##
_dsl.dsl_display_type_rgba_rectangle_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_wchar_p, c_bool, c_wchar_p]
_dsl.dsl_display_type_rgba_rectangle_new.restype = c_uint
def dsl_display_type_rgba_rectangle_new(name, left, top, width, height, border_width, color, has_bg_color, bg_color):
    global _dsl
    result =_dsl.dsl_display_type_rgba_rectangle_new(name, left, top, width, height, border_width, color, has_bg_color, bg_color)
    return int(result)

##
## dsl_display_type_rgba_circle_new()
##
_dsl.dsl_display_type_rgba_circle_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_wchar_p, c_bool, c_wchar_p]
_dsl.dsl_display_type_rgba_circle_new.restype = c_uint
def dsl_display_type_rgba_circle_new(name, x_center, y_center, radius, color, has_bg_color, bg_color):
    global _dsl
    result =_dsl.dsl_display_type_rgba_circle_new(name, x_center, y_center, radius, color, has_bg_color, bg_color)
    return int(result)

##
## dsl_display_type_delete()
##
_dsl.dsl_display_type_delete.argtypes = [c_wchar_p]
_dsl.dsl_display_type_delete.restype = c_uint
def dsl_display_type_delete(name):
    global _dsl
    result =_dsl.dsl_display_type_delete(name)
    return int(result)

##
## dsl_display_type_delete_many()
##
#_dsl.dsl_display_type_delete_many.argtypes = [??]
_dsl.dsl_display_type_delete_many.restype = c_uint
def dsl_display_type_delete_many(names):
    global _dsl
    arr = (c_wchar_p * len(names))()
    arr[:] = names
    result =_dsl.dsl_display_type_delete_many(arr)
    return int(result)

##
## dsl_display_type_delete_all()
##
_dsl.dsl_display_type_delete_all.argtypes = []
_dsl.dsl_display_type_delete_all.restype = c_uint
def dsl_display_type_delete_all():
    global _dsl
    result =_dsl.dsl_display_type_delete_all()
    return int(result)

##
## dsl_display_type_list_size()
##
_dsl.dsl_display_type_list_size.restype = c_uint
def dsl_display_type_list_size():
    global _dsl
    result =_dsl.dsl_display_type_list_size()
    return int(result)

##
## dsl_ode_action_callback_new()
##
_dsl.dsl_ode_action_callback_new.argtypes = [c_wchar_p, DSL_ODE_HANDLE_OCCURRENCE, c_void_p]
_dsl.dsl_ode_action_callback_new.restype = c_uint
def dsl_ode_action_callback_new(name, client_handler, client_data):
    global _dsl
    handler_cb = DSL_ODE_HANDLE_OCCURRENCE(client_handler)
    callbacks.append(handler_cb)
    result = _dsl.dsl_ode_action_callback_new(name, handler_cb, client_data)
    return int(result)
    
##
## dsl_ode_action_capture_frame_new()
##
_dsl.dsl_ode_action_capture_frame_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_capture_frame_new.restype = c_uint
def dsl_ode_action_capture_frame_new(name, outdir):
    global _dsl
    result =_dsl.dsl_ode_action_capture_frame_new(name, outdir)
    return int(result)

##
## dsl_ode_action_capture_object_new()
##
_dsl.dsl_ode_action_capture_object_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_capture_object_new.restype = c_uint
def dsl_ode_action_capture_object_new(name, outdir):
    global _dsl
    result =_dsl.dsl_ode_action_capture_object_new(name, outdir)
    return int(result)

##
## dsl_ode_action_display_new()
##
_dsl.dsl_ode_action_display_new.argtypes = [c_wchar_p, c_uint, c_uint, c_bool]
_dsl.dsl_ode_action_display_new.restype = c_uint
def dsl_ode_action_display_new(name, offsetX, offsetY, offsetY_with_classId):
    global _dsl
    result =_dsl.dsl_ode_action_display_new(name, offsetX, offsetY, offsetY_with_classId)
    return int(result)

##
## dsl_ode_action_fill_area_new()
##
_dsl.dsl_ode_action_fill_area_new.argtypes = [c_wchar_p, c_wchar_p, c_double, c_double, c_double, c_double]
_dsl.dsl_ode_action_fill_area_new.restype = c_uint
def dsl_ode_action_fill_area_new(name, area, red, green, blue, alpha):
    global _dsl
    result =_dsl.dsl_ode_action_fill_area_new(name, area, red, green, blue, alpha)
    return int(result)

##
## dsl_ode_action_fill_frame_new()
##
_dsl.dsl_ode_action_fill_frame_new.argtypes = [c_wchar_p, c_double, c_double, c_double, c_double]
_dsl.dsl_ode_action_fill_frame_new.restype = c_uint
def dsl_ode_action_fill_frame_new(name, red, green, blue, alpha):
    global _dsl
    result =_dsl.dsl_ode_action_fill_frame_new(name, red, green, blue, alpha)
    return int(result)

##
## dsl_ode_action_fill_object_new()
##
_dsl.dsl_ode_action_fill_object_new.argtypes = [c_wchar_p, c_double, c_double, c_double, c_double]
_dsl.dsl_ode_action_fill_object_new.restype = c_uint
def dsl_ode_action_fill_object_new(name, red, green, blue, alpha):
    global _dsl
    result =_dsl.dsl_ode_action_fill_object_new(name, red, green, blue, alpha)
    return int(result)

##
## dsl_ode_action_handler_disable_new()
##
_dsl.dsl_ode_action_handler_disable_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_handler_disable_new.restype = c_uint
def dsl_ode_action_handler_disable_new(name, handler):
    global _dsl
    result =_dsl.dsl_ode_action_handler_disable_new(name, handler)
    return int(result)

##
## dsl_ode_action_hide_new()
##
_dsl.dsl_ode_action_hide_new.argtypes = [c_wchar_p, c_bool, c_bool]
_dsl.dsl_ode_action_hide_new.restype = c_uint
def dsl_ode_action_hide_new(name, text, border):
    global _dsl
    result =_dsl.dsl_ode_action_hide_new(name, text, border)
    return int(result)

##
## dsl_ode_action_log_new()
##
_dsl.dsl_ode_action_log_new.argtypes = [c_wchar_p]
_dsl.dsl_ode_action_log_new.restype = c_uint
def dsl_ode_action_log_new(name):
    global _dsl
    result =_dsl.dsl_ode_action_log_new(name)
    return int(result)

##
## dsl_ode_action_overlay_frame_new()
##
_dsl.dsl_ode_action_overlay_frame_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_overlay_frame_new.restype = c_uint
def dsl_ode_action_overlay_frame_new(name, display_type):
    global _dsl
    result =_dsl.dsl_ode_action_overlay_frame_new(name, display_type)
    return int(result)

##
## dsl_ode_action_print_new()
##
_dsl.dsl_ode_action_print_new.argtypes = [c_wchar_p]
_dsl.dsl_ode_action_print_new.restype = c_uint
def dsl_ode_action_print_new(name):
    global _dsl
    result =_dsl.dsl_ode_action_print_new(name)
    return int(result)

##
## dsl_ode_action_pause_new()
##
_dsl.dsl_ode_action_pause_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_pause_new.restype = c_uint
def dsl_ode_action_pause_new(name, pipeline):
    global _dsl
    result =_dsl.dsl_ode_action_pause_new(name, pipeline)
    return int(result)

##
## dsl_ode_action_redact_new()
##
_dsl.dsl_ode_action_redact_new.argtypes = [c_wchar_p]
_dsl.dsl_ode_action_redact_new.restype = c_uint
def dsl_ode_action_redact_new(name):
    global _dsl
    result =_dsl.dsl_ode_action_redact_new(name)
    return int(result)

##
## dsl_ode_action_sink_add_new()
##
_dsl.dsl_ode_action_sink_add_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_sink_add_new.restype = c_uint
def dsl_ode_action_sink_add_new(name, pipeline, sink):
    global _dsl
    result =_dsl.dsl_ode_action_sink_add_new(name, pipeline, sink)
    return int(result)

##
## dsl_ode_action_sink_remove_new()
##
_dsl.dsl_ode_action_sink_remove_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_sink_remove_new.restype = c_uint
def dsl_ode_action_sink_remove_new(name, pipeline, sink):   
    global _dsl
    result =_dsl.dsl_ode_action_sink_remove_new(name, pipeline, sink)
    return int(result)

##
## dsl_ode_action_sink_record_start_new()
##
_dsl.dsl_ode_action_sink_record_start_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_void_p]
_dsl.dsl_ode_action_sink_record_start_new.restype = c_uint
def dsl_ode_action_sink_record_start_new(name, record_sink, start, duration, client_data):
    global _dsl
    result =_dsl.dsl_ode_action_sink_record_start_new(name, record_sink, start, duration, client_data)
    return int(result)


##
## dsl_ode_action_source_add_new()
##
_dsl.dsl_ode_action_source_add_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_source_add_new.restype = c_uint
def dsl_ode_action_source_add_new(name, pipeline, source):
    global _dsl
    result =_dsl.dsl_ode_action_source_add_new(name, pipeline, source)
    return int(result)

##
## dsl_ode_action_source_remove_new()
##
_dsl.dsl_ode_action_source_remove_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_source_remove_new.restype = c_uint
def dsl_ode_action_source_remove_new(name, pipeline, source):
    global _dsl
    result =_dsl.dsl_ode_action_source_remove_new(name, pipeline, source)
    return int(result)

##
## dsl_ode_action_action_disable_new()
##
_dsl.dsl_ode_action_action_disable_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_action_disable_new.restype = c_uint
def dsl_ode_action_action_disable_new(name, action):
    global _dsl
    result =_dsl.dsl_ode_action_action_disable_new(name, action)
    return int(result)

##
## dsl_ode_action_action_enable()
##
_dsl.dsl_ode_action_action_enable_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_action_enable_new.restype = c_uint
def dsl_ode_action_action_enable_new(name, action):
    global _dsl
    result =_dsl.dsl_ode_action_action_enable_new(name, action)
    return int(result)

##
## dsl_ode_action_area_add_new()
##
_dsl.dsl_ode_action_area_add_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_area_add_new.restype = c_uint
def dsl_ode_action_area_add_new(name, trigger, area):
    global _dsl
    result =_dsl.dsl_ode_action_area_add_new(name, trigger, area)
    return int(result)

##
## dsl_ode_action_area_remove_new()
##
_dsl.dsl_ode_action_area_remove_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_area_remove_new.restype = c_uint
def dsl_ode_action_area_remove_new(name, trigger, area):
    global _dsl
    result =_dsl.dsl_ode_action_area_remove_new(name, trigger, area)
    return int(result)

##
## dsl_ode_action_trigger_reset_new()
##
_dsl.dsl_ode_action_trigger_reset_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_trigger_reset_new.restype = c_uint
def dsl_ode_action_trigger_reset_new(name, trigger):
    global _dsl
    result =_dsl.dsl_ode_action_trigger_reset_new(name, trigger)
    return int(result)

##
## dsl_ode_action_trigger_disable_new()
##
_dsl.dsl_ode_action_trigger_disable_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_trigger_disable_new.restype = c_uint
def dsl_ode_action_trigger_disable_new(name, trigger):
    global _dsl
    result =_dsl.dsl_ode_action_trigger_disable_new(name, trigger)
    return int(result)

##
## dsl_ode_action_trigger_enable_new()
##
_dsl.dsl_ode_action_trigger_enable_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_action_trigger_enable_new.restype = c_uint
def dsl_ode_action_trigger_enable_new(name, trigger):
    global _dsl
    result =_dsl.dsl_ode_action_trigger_enable_new(name, trigger)
    return int(result)

##
## dsl_ode_action_delete()
##
_dsl.dsl_ode_action_delete.argtypes = [c_wchar_p]
_dsl.dsl_ode_action_delete.restype = c_uint
def dsl_ode_action_delete(name):
    global _dsl
    result =_dsl.dsl_ode_action_delete(name)
    return int(result)

##
## dsl_ode_action_delete_many()
##
#_dsl.dsl_ode_action_delete_many.argtypes = [??]
_dsl.dsl_ode_action_delete_many.restype = c_uint
def dsl_ode_action_delete_many(names):
    global _dsl
    arr = (c_wchar_p * len(names))()
    arr[:] = names
    result =_dsl.dsl_ode_action_delete_many(arr)
    return int(result)

##
## dsl_ode_action_delete_all()
##
_dsl.dsl_ode_action_delete_all.argtypes = []
_dsl.dsl_ode_action_delete_all.restype = c_uint
def dsl_ode_action_delete_all():
    global _dsl
    result =_dsl.dsl_ode_action_delete_all()
    return int(result)

##
## dsl_ode_action_list_size()
##
_dsl.dsl_ode_action_list_size.restype = c_uint
def dsl_ode_action_list_size():
    global _dsl
    result =_dsl.dsl_ode_action_list_size()
    return int(result)

##
## dsl_ode_area_new()
##
_dsl.dsl_ode_area_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint, c_bool]
_dsl.dsl_ode_area_new.restype = c_uint
def dsl_ode_area_new(name, left, top, width, height, display):
    global _dsl
    result =_dsl.dsl_ode_area_new(name, left, top, width, height, display)
    return int(result)

##
## dsl_ode_area_get()
##
_dsl.dsl_ode_area_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_ode_area_get.restype = c_uint
def dsl_ode_area_get(name):
    global _dsl
    left = c_uint(0)
    top = c_uint(0)
    width = c_uint(0)
    height = c_uint(0)
    display = c_bool(0)
    result = _dsl.dsl_ode_area_get(name, DSL_UINT_P(left), 
        DSL_UINT_P(top), DSL_UINT_P(width), DSL_UINT_P(height), DSL_BOOL_P(display))
    return int(result), left.value, top.value, width.value, height.value, display.value 

##
## dsl_ode_area_set()
##
_dsl.dsl_ode_area_set.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint, c_bool]
_dsl.dsl_ode_area_set.restype = c_uint
def dsl_ode_area_set(name, left, top, width, height, display):
    global _dsl
    result =_dsl.dsl_ode_area_set(name, left, top, width, height, display)
    return int(result)

##
## dsl_ode_area_color_get()
##
_dsl.dsl_ode_area_color_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_ode_area_color_get.restype = c_uint
def dsl_ode_area_color_get(name):
    global _dsl
    red = c_double(0)
    green = c_double(0)
    blue = c_double(0)
    alpha = c_double(0)
    result = _dsl.dsl_ode_area_color_get(name, 
        DSL_DOUBLE_P(red), DSL_DOUBLE_P(green), DSL_DOUBLE_P(blue), DSL_DOUBLE_P(alpha))
    return int(result), red.value, green.value, blue.value, alpha.value 

##
## dsl_ode_area_color_set()
##
_dsl.dsl_ode_area_color_set.argtypes = [c_wchar_p, c_double, c_double, c_double, c_double]
_dsl.dsl_ode_area_color_set.restype = c_uint
def dsl_ode_area_color_set(name, red, green, blue, alpha):
    global _dsl
    result =_dsl.dsl_ode_area_color_set(name, red, green, blue, alpha)
    return int(result)

##
## dsl_ode_area_delete()
##
_dsl.dsl_ode_area_delete.argtypes = [c_wchar_p]
_dsl.dsl_ode_area_delete.restype = c_uint
def dsl_ode_area_delete(name):
    global _dsl
    result =_dsl.dsl_ode_area_delete(name)
    return int(result)

##
## dsl_ode_area_delete_many()
##
#_dsl.dsl_ode_area_delete_many.argtypes = [??]
_dsl.dsl_ode_area_delete_many.restype = c_uint
def dsl_ode_area_delete_many(names):
    global _dsl
    arr = (c_wchar_p * len(names))()
    arr[:] = names
    result =_dsl.dsl_ode_area_delete_many(arr)
    return int(result)

##
## dsl_ode_area_delete_all()
##
_dsl.dsl_ode_area_delete_all.argtypes = []
_dsl.dsl_ode_area_delete_all.restype = c_uint
def dsl_ode_area_delete_all():
    global _dsl
    result =_dsl.dsl_ode_area_delete_all()
    return int(result)

##
## dsl_ode_area_list_size()
##
_dsl.dsl_ode_area_list_size.restype = c_uint
def dsl_ode_area_list_size():
    global _dsl
    result =_dsl.dsl_ode_area_list_size()
    return int(result)

##
## dsl_ode_trigger_always_new()
##
_dsl.dsl_ode_trigger_always_new.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_ode_trigger_always_new.restype = c_uint
def dsl_ode_trigger_always_new(name, when):
    global _dsl
    result =_dsl.dsl_ode_trigger_always_new(name, when)
    return int(result)

##
## dsl_ode_trigger_absence_new()
##
_dsl.dsl_ode_trigger_absence_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_ode_trigger_absence_new.restype = c_uint
def dsl_ode_trigger_absence_new(name, class_id, limit):
    global _dsl
    result =_dsl.dsl_ode_trigger_absence_new(name, class_id, limit)
    return int(result)

##
## dsl_ode_trigger_custom_new()
##
_dsl.dsl_ode_trigger_custom_new.argtypes = [c_wchar_p, c_uint, c_uint, DSL_ODE_CHECK_FOR_OCCURRENCE, c_void_p]
_dsl.dsl_ode_trigger_custom_new.restype = c_uint
def dsl_ode_trigger_custom_new(name, class_id, limit, client_checker, client_data):
    global _dsl
    checker_cb = DSL_ODE_CHECK_FOR_OCCURRENCE(client_checker)
    callbacks.append(checker_cb)
    result = _dsl.dsl_ode_trigger_custom_new(name, class_id, limit, checker_cb, client_data)
    return int(result)

##
## dsl_ode_trigger_intersection_new()
##
_dsl.dsl_ode_trigger_intersection_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_ode_trigger_intersection_new.restype = c_uint
def dsl_ode_trigger_intersection_new(name, class_id, limit):
    global _dsl
    result =_dsl.dsl_ode_trigger_intersection_new(name, class_id, limit)
    return int(result)

##
## dsl_ode_trigger_maximum_new()
##
_dsl.dsl_ode_trigger_maximum_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint]
_dsl.dsl_ode_trigger_maximum_new.restype = c_uint
def dsl_ode_trigger_maximum_new(name, class_id, limit, maximum):
    global _dsl
    result =_dsl.dsl_ode_trigger_maximum_new(name, class_id, limit, maximum)
    return int(result)

##
## dsl_ode_trigger_minimum_new()
##
_dsl.dsl_ode_trigger_minimum_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint]
_dsl.dsl_ode_trigger_minimum_new.restype = c_uint
def dsl_ode_trigger_minimum_new(name, class_id, limit, minimum):
    global _dsl
    result =_dsl.dsl_ode_trigger_minimum_new(name, class_id, limit, minimum)
    return int(result)

##
## dsl_ode_trigger_occurrence_new()
##
_dsl.dsl_ode_trigger_occurrence_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_ode_trigger_occurrence_new.restype = c_uint
def dsl_ode_trigger_occurrence_new(name, class_id, limit):
    global _dsl
    result =_dsl.dsl_ode_trigger_occurrence_new(name, class_id, limit)
    return int(result)

##
## dsl_ode_trigger_summation_new()
##
_dsl.dsl_ode_trigger_summation_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_ode_trigger_summation_new.restype = c_uint
def dsl_ode_trigger_summation_new(name, class_id, limit):
    global _dsl
    result =_dsl.dsl_ode_trigger_summation_new(name, class_id, limit)
    return int(result)

##
## dsl_ode_trigger_range_new()
##
_dsl.dsl_ode_trigger_range_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_ode_trigger_range_new.restype = c_uint
def dsl_ode_trigger_range_new(name, class_id, limit, lower, upper):
    global _dsl
    result =_dsl.dsl_ode_trigger_range_new(name, class_id, limit, lower, upper)
    return int(result)

##
## dsl_ode_trigger_reset()
##
_dsl.dsl_ode_trigger_reset.argtypes = [c_wchar_p]
_dsl.dsl_ode_trigger_reset.restype = c_uint
def dsl_ode_trigger_reset(name):
    global _dsl
    result =_dsl.dsl_ode_trigger_reset(name)
    return int(result)

##
## dsl_ode_trigger_enabled_get()
##
_dsl.dsl_ode_trigger_enabled_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_ode_trigger_enabled_get.restype = c_uint
def dsl_ode_trigger_enabled_get(name):
    global _dsl
    enabled = c_bool(0)
    result =_dsl.dsl_ode_trigger_enabled_get(name, DSL_BOOL_P(enabled))
    return int(result), enabled.value

##
## dsl_ode_trigger_enabled_set()
##
_dsl.dsl_ode_trigger_enabled_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_ode_trigger_enabled_set.restype = c_uint
def dsl_ode_trigger_enabled_set(name, enabled):
    global _dsl
    result =_dsl.dsl_ode_trigger_enabled_set(name, enabled)
    return int(result)

##
## dsl_ode_trigger_source_id_get()
##
_dsl.dsl_ode_trigger_source_id_get.argtypes = [c_wchar_p, POINTER(c_uint)]
_dsl.dsl_ode_trigger_source_id_get.restype = c_uint
def dsl_ode_trigger_source_id_get(name):
    global _dsl
    source_id = c_uint(0)
    result =_dsl.dsl_ode_trigger_source_id_get(name, DSL_UINT_P(source_id))
    return int(result), source_id.value

##
## dsl_ode_trigger_source_id_set()
##
_dsl.dsl_ode_trigger_source_id_set.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_ode_trigger_source_id_set.restype = c_uint
def dsl_ode_trigger_source_id_set(name, source_id):
    global _dsl
    result =_dsl.dsl_ode_trigger_source_id_set(name, source_id)
    return int(result)

##
## dsl_ode_trigger_class_id_get()
##
_dsl.dsl_ode_trigger_class_id_get.argtypes = [c_wchar_p, POINTER(c_uint)]
_dsl.dsl_ode_trigger_class_id_get.restype = c_uint
def dsl_ode_trigger_class_id_get(name):
    global _dsl
    class_id = c_uint(0)
    result =_dsl.dsl_ode_trigger_class_id_get(name, DSL_UINT_P(class_id))
    return int(result), class_id.value

##
## dsl_ode_trigger_class_id_set()
##
_dsl.dsl_ode_trigger_class_id_set.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_ode_trigger_class_id_set.restype = c_uint
def dsl_ode_trigger_class_id_set(name, class_id):
    global _dsl
    result =_dsl.dsl_ode_trigger_class_id_set(name, class_id)
    return int(result)

##
## dsl_ode_trigger_confidence_min_get()
##
_dsl.dsl_ode_trigger_confidence_min_get.argtypes = [c_wchar_p, POINTER(c_double)]
_dsl.dsl_ode_trigger_confidence_min_get.restype = c_uint
def dsl_ode_trigger_confidence_min_get(name):
    global _dsl
    min_confidence = c_doube(0)
    result =_dsl.dsl_ode_trigger_confidence_min_get(name, DSL_DOUBLE_P(min_confidence))
    return int(result), min_confidence.value

##
## dsl_ode_trigger_confidence_min_set()
##
_dsl.dsl_ode_trigger_confidence_min_set.argtypes = [c_wchar_p, c_double]
_dsl.dsl_ode_trigger_confidence_min_set.restype = c_uint
def dsl_ode_trigger_confidence_min_set(name, min_confidence):
    global _dsl
    result =_dsl.dsl_ode_trigger_confidence_min_set(name, min_confidence)
    return int(result)

##
## dsl_ode_trigger_dimensions_min_get()
##
_dsl.dsl_ode_trigger_dimensions_min_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_ode_trigger_dimensions_min_get.restype = c_uint
def dsl_ode_trigger_dimensions_min_get(name):
    global _dsl
    min_width = c_uint(0)
    min_height = c_uint(0)
    result = _dsl.dsl_ode_trigger_dimensions_min_get(name, DSL_UINT_P(min_width), DSL_UINT_P(min_height))
    return int(result), min_width.value, min_height.value 

##
## dsl_ode_trigger_dimensions_min_set()
##
_dsl.dsl_ode_trigger_dimensions_min_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_ode_trigger_dimensions_min_set.restype = c_uint
def dsl_ode_trigger_dimensions_min_set(name, min_width, min_height):
    global _dsl
    result = _dsl.dsl_ode_trigger_dimensions_min_set(name, min_width, min_height)
    return int(result)

##
## dsl_ode_trigger_dimensions_max_get()
##
_dsl.dsl_ode_trigger_dimensions_max_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_ode_trigger_dimensions_max_get.restype = c_uint
def dsl_ode_trigger_dimensions_max_get(name):
    global _dsl
    max_width = c_uint(0)
    max_height = c_uint(0)
    result = _dsl.dsl_ode_trigger_dimensions_max_get(name, DSL_UINT_P(max_width), DSL_UINT_P(max_height))
    return int(result), max_width.value, max_height.value 

##
## dsl_ode_trigger_dimensions_max_set()
##
_dsl.dsl_ode_trigger_dimensions_max_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_ode_trigger_dimensions_max_set.restype = c_uint
def dsl_ode_trigger_dimensions_max_set(name, max_width, max_height):
    global _dsl
    result = _dsl.dsl_ode_trigger_dimensions_max_set(name, max_width, max_height)
    return int(result)

##
## dsl_ode_trigger_infer_done_only_get()
##
_dsl.dsl_ode_trigger_infer_done_only_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_ode_trigger_infer_done_only_get.restype = c_uint
def dsl_ode_trigger_infer_done_only_get(name):
    global _dsl
    infer_done_only = c_bool(0)
    result =_dsl.dsl_ode_trigger_infer_done_only_get(name, DSL_BOOL_P(infer_done_only))
    return int(result), infer_done_only.value

##
## dsl_ode_trigger_infer_done_only_set()
##
_dsl.dsl_ode_trigger_infer_done_only_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_ode_trigger_infer_done_only_set.restype = c_uint
def dsl_ode_trigger_infer_done_only_set(name, infer_done_only):
    global _dsl
    result =_dsl.dsl_ode_trigger_infer_done_only_set(name, infer_done_only)
    return int(result)

##
## dsl_ode_trigger_action_add()
##
_dsl.dsl_ode_trigger_action_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_trigger_action_add.restype = c_uint
def dsl_ode_trigger_action_add(name, action):
    global _dsl
    result =_dsl.dsl_ode_trigger_action_add(name, action)
    return int(result)

##
## dsl_ode_trigger_action_add_many()
##
#_dsl.dsl_ode_trigger_action_add_many.argtypes = [??]
_dsl.dsl_ode_trigger_action_add_many.restype = c_uint
def dsl_ode_trigger_action_add_many(name, actions):
    global _dsl
    arr = (c_wchar_p * len(actions))()
    arr[:] = actions
    result =_dsl.dsl_ode_trigger_action_add_many(name, arr)
    return int(result)

##
## dsl_ode_trigger_action_remove()
##
_dsl.dsl_ode_trigger_action_remove.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_trigger_action_remove.restype = c_uint
def dsl_ode_trigger_action_remove(name, action):
    global _dsl
    result =_dsl.dsl_ode_trigger_action_remove(name, action)
    return int(result)

##
## dsl_ode_trigger_action_remove_many()
##
#_dsl.dsl_ode_trigger_action_remove_many.argtypes = [??]
_dsl.dsl_ode_trigger_action_remove_many.restype = c_uint
def dsl_ode_trigger_action_remove_many(name, actions):
    global _dsl
    arr = (c_wchar_p * len(actions))()
    arr[:] = actions
    result =_dsl.dsl_ode_trigger_action_remove_many(name, arr)
    return int(result)

##
## dsl_ode_trigger_action_remove_all()
##
_dsl.dsl_ode_trigger_action_remove_all.argtypes = [c_wchar_p]
_dsl.dsl_ode_trigger_action_remove_all.restype = c_uint
def dsl_ode_trigger_action_remove_all(name):
    global _dsl
    result =_dsl.dsl_ode_trigger_action_remove_all(name)
    return int(result)

##
## dsl_ode_trigger_area_add()
##
_dsl.dsl_ode_trigger_area_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_trigger_area_add.restype = c_uint
def dsl_ode_trigger_area_add(name, area):
    global _dsl
    result =_dsl.dsl_ode_trigger_area_add(name, area)
    return int(result)

##
## dsl_ode_trigger_area_add_many()
##
#_dsl.dsl_ode_trigger_area_add_many.argtypes = [??]
_dsl.dsl_ode_trigger_area_add_many.restype = c_uint
def dsl_ode_trigger_area_add_many(name, areas):
    global _dsl
    arr = (c_wchar_p * len(areas))()
    arr[:] = areas
    result =_dsl.dsl_ode_trigger_area_add_many(name, arr)
    return int(result)

##
## dsl_ode_trigger_delete()
##
_dsl.dsl_ode_trigger_delete.argtypes = [c_wchar_p]
_dsl.dsl_ode_trigger_delete.restype = c_uint
def dsl_ode_trigger_delete(name):
    global _dsl
    result =_dsl.dsl_ode_trigger_delete(name)
    return int(result)

##
## dsl_ode_trigger_delete_many()
##
#_dsl.dsl_ode_trigger_delete_many.argtypes = [??]
_dsl.dsl_ode_trigger_delete_many.restype = c_uint
def dsl_ode_trigger_delete_many(names):
    global _dsl
    arr = (c_wchar_p * len(names))()
    arr[:] = names
    result =_dsl.dsl_ode_trigger_delete_many(arr)
    return int(result)

##
## dsl_ode_trigger_delete_all()
##
_dsl.dsl_ode_trigger_delete_all.argtypes = []
_dsl.dsl_ode_trigger_delete_all.restype = c_uint
def dsl_ode_trigger_delete_all():
    global _dsl
    result =_dsl.dsl_ode_trigger_delete_all()
    return int(result)

##
## dsl_ode_trigger_list_size()
##
_dsl.dsl_ode_trigger_list_size.restype = c_uint
def dsl_ode_trigger_list_size():
    global _dsl
    result =_dsl.dsl_ode_trigger_list_size()
    return int(result)

##
## dsl_ode_handler_new()
##
_dsl.dsl_ode_handler_new.argtypes = [c_wchar_p]
_dsl.dsl_ode_handler_new.restype = c_uint
def dsl_ode_handler_new(name):
    global _dsl
    result =_dsl.dsl_ode_handler_new(name)
    return int(result)

##
## dsl_ode_handler_trigger_add()
##
_dsl.dsl_ode_handler_trigger_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_ode_handler_trigger_add.restype = c_uint
def dsl_ode_handler_trigger_add(name, trigger):
    global _dsl
    result =_dsl.dsl_ode_handler_trigger_add(name, trigger)
    return int(result)

##
## dsl_ode_handler_trigger_add_many()
##
#_dsl.dsl_ode_handler_trigger_add_many.argtypes = [??]
_dsl.dsl_ode_handler_trigger_add_many.restype = c_uint
def dsl_ode_handler_trigger_add_many(name, triggers):
    global _dsl
    arr = (c_wchar_p * len(triggers))()
    arr[:] = triggers
    result =_dsl.dsl_ode_handler_trigger_add_many(name, arr)
    return int(result)
    
##
## dsl_source_csi_new()
##
_dsl.dsl_source_csi_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_source_csi_new.restype = c_uint
def dsl_source_csi_new(name, width, height, fps_n, fps_d):
    global _dsl
    result =_dsl.dsl_source_csi_new(name, width, height, fps_n, fps_d)
    return int(result)

##
## dsl_source_usb_new()
##
_dsl.dsl_source_usb_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_source_usb_new.restype = c_uint
def dsl_source_usb_new(name, width, height, fps_n, fps_d):
    global _dsl
    result =_dsl.dsl_source_usb_new(name, width, height, fps_n, fps_d)
    return int(result)

##
## dsl_source_uri_new()
##
_dsl.dsl_source_uri_new.argtypes = [c_wchar_p, c_wchar_p, c_bool, c_uint, c_uint, c_uint]
_dsl.dsl_source_uri_new.restype = c_uint
def dsl_source_uri_new(name, uri, is_live, cudadec_mem_type, intra_decode, drop_frame_interval):
    global _dsl
    result = _dsl.dsl_source_uri_new(name, uri, is_live, cudadec_mem_type, intra_decode, drop_frame_interval)
    return int(result)

##
## dsl_source_rtsp_new()
##
_dsl.dsl_source_rtsp_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_source_rtsp_new.restype = c_uint
def dsl_source_rtsp_new(name, uri, protocol, cudadec_mem_type, intra_decode, drop_frame_interval):
    global _dsl
    result = _dsl.dsl_source_rtsp_new(name, uri, protocol, cudadec_mem_type, intra_decode, drop_frame_interval)
    return int(result)

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

##
## dsl_source_decode_uri_get()
##
_dsl.dsl_source_decode_uri_get.argtypes = [c_wchar_p, POINTER(c_wchar_p)]
_dsl.dsl_source_decode_uri_get.restype = c_uint
def dsl_source_decode_uri_get(name):
    global _dsl
    uri = c_wchar_p(0)
    result = _dsl.dsl_source_decode_uri_get(name, DSL_WCHAR_PP(uri))
    return int(result), uri.value 

##
## dsl_source_decode_uri_set()
##
_dsl.dsl_source_decode_uri_set.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_source_decode_uri_set.restype = c_uint
def dsl_source_decode_uri_set(name, uir):
    global _dsl
    result = _dsl.dsl_source_decode_uri_set(name, uir)
    return int(result)

##
## dsl_source_decode_dewarper_add()
##
_dsl.dsl_source_decode_dewarper_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_source_decode_dewarper_add.restype = c_uint
def dsl_source_decode_dewarper_add(name, dewarper):
    global _dsl
    result = _dsl.dsl_source_decode_dewarper_add(name, dewarper)
    return int(result)

##
## dsl_source_decode_dewarper_remove()
##
_dsl.dsl_source_decode_dewarper_remove.argtypes = [c_wchar_p]
_dsl.dsl_source_decode_dewarper_remove.restype = c_uint
def dsl_source_decode_dewarper_remove(name):
    global _dsl
    result = _dsl.dsl_source_decode_dewarper_remove(name)
    return int(result)

##
## dsl_source_is_live()
##
_dsl.dsl_source_is_live.argtypes = [c_wchar_p]
_dsl.dsl_source_is_live.restype = c_bool
def dsl_source_is_live(name):
    global _dsl
    result = _dsl.dsl_source_is_live(name)
    return bool(result)

##
## dsl_source_num_in_use_get()
##
_dsl.dsl_source_num_in_use_get.restype = c_uint
def dsl_source_num_in_use_get():
    global _dsl
    result = _dsl.dsl_source_num_in_use_get()
    return int(result)

##
## dsl_source_num_in_use_max_get()
##
_dsl.dsl_source_num_in_use_max_get.restype = c_uint
def dsl_source_num_in_use_max_get():
    global _dsl
    result = _dsl.dsl_source_num_in_use_max_get()
    return int(result)

##
## dsl_source_num_in_use_max_set()
##
_dsl.dsl_source_num_in_use_max_set.argtypes = [c_uint]
def dsl_source_num_in_use_max_set(max):
    global _dsl
    success = _dsl.dsl_source_num_in_use_max_set(max)
    return bool(success)

##
## dsl_dewarper_new()
##
_dsl.dsl_dewarper_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_dewarper_new.restype = c_uint
def dsl_dewarper_new(name, config_file):
    global _dsl
    result = _dsl.dsl_dewarper_new(name, config_file)
    return int(result)

##
## dsl_gie_primary_new()
##
_dsl.dsl_gie_primary_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_uint]
_dsl.dsl_gie_primary_new.restype = c_uint
def dsl_gie_primary_new(name, infer_config_file, model_engine_file, interval):
    global _dsl
    result = _dsl.dsl_gie_primary_new(name, infer_config_file, model_engine_file, interval)
    return int(result)

##
## dsl_gie_primary_batch_meta_handler_add()
##
_dsl.dsl_gie_primary_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_gie_primary_batch_meta_handler_add.restype = c_uint
def dsl_gie_primary_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    callbacks.append(meta_handler)
    result = _dsl.dsl_gie_primary_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

##
## dsl_gie_primary_batch_meta_handler_remove()
##
_dsl.dsl_gie_primary_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_gie_primary_batch_meta_handler_remove.restype = c_uint
def dsl_gie_primary_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_gie_primary_batch_meta_handler_remove(name, pad)
    return int(result)

##
## dsl_gie_primary_kitti_output_enabled_set()
##
_dsl.dsl_gie_primary_kitti_output_enabled_set.argtypes = [c_wchar_p, c_bool, c_wchar_p]
_dsl.dsl_gie_primary_kitti_output_enabled_set.restype = c_uint
def dsl_gie_primary_kitti_output_enabled_set(name, enabled, path):
    global _dsl
    result = _dsl.dsl_gie_primary_kitti_output_enabled_set(name, enabled, path)
    return int(result)

##
## dsl_gie_secondary_new()
##
_dsl.dsl_gie_secondary_new.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p]
_dsl.dsl_gie_secondary_new.restype = c_uint
def dsl_gie_secondary_new(name, infer_config_file, model_engine_file, infer_on_gie_name, interval):
    global _dsl
    result = _dsl.dsl_gie_secondary_new(name, infer_config_file, model_engine_file, infer_on_gie_name, interval)
    return int(result)

##
## dsl_gie_infer_config_file_get()
##
_dsl.dsl_gie_infer_config_file_get.argtypes = [c_wchar_p, POINTER(c_wchar_p)]
_dsl.dsl_gie_infer_config_file_get.restype = c_uint
def dsl_gie_infer_config_file_get(name):
    global _dsl
    file = c_wchar_p(0)
    result = _dsl.dsl_gie_infer_config_file_get(name, DSL_WCHAR_PP(file))
    return int(result), file.value 

##
## dsl_gie_infer_config_file_set()
##
_dsl.dsl_gie_infer_config_file_set.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_gie_infer_config_file_set.restype = c_uint
def dsl_gie_infer_config_file_set(name, infer_config_file):
    global _dsl
    result = _dsl.dsl_gie_infer_config_file_set(name, infer_config_file)
    return int(result)

##
## dsl_gie_model_engine_file_get()
##
_dsl.dsl_gie_model_engine_file_get.argtypes = [c_wchar_p, POINTER(c_wchar_p)]
_dsl.dsl_gie_model_engine_file_get.restype = c_uint
def dsl_gie_model_engine_file_get(name):
    global _dsl
    file = c_wchar_p(0)
    result = _dsl.dsl_gie_model_engine_file_get(name, DSL_WCHAR_PP(file))
    return int(result), file.value 

##
## dsl_gie_model_engine_file_set()
##
_dsl.dsl_gie_model_engine_file_set.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_gie_model_engine_file_set.restype = c_uint
def dsl_gie_model_engine_file_set(name, model_engine_file):
    global _dsl
    result = _dsl.dsl_gie_model_engine_file_set(name, model_engine_file)
    return int(result)

##
## dsl_gie_raw_output_enabled_set()
##
_dsl.dsl_gie_raw_output_enabled_set.argtypes = [c_wchar_p, c_bool, c_wchar_p]
_dsl.dsl_gie_raw_output_enabled_set.restype = c_uint
def dsl_gie_raw_output_enabled_set(name, enabled, path):
    global _dsl
    result = _dsl.dsl_gie_raw_output_enabled_set(name, enabled, path)
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

##
## dsl_tracker_iou_new()
##
_dsl.dsl_tracker_iou_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint]
_dsl.dsl_tracker_iou_new.restype = c_uint
def dsl_tracker_iou_new(name, config_file, width, height):
    global _dsl
    result = _dsl.dsl_tracker_iou_new(name, config_file, width, height)
    return int(result)

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
    callbacks.append(meta_handler)
    result = _dsl.dsl_tracker_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

##
## dsl_tracker_batch_meta_handler_remove()
##
_dsl.dsl_tracker_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_tracker_batch_meta_handler_remove.restype = c_uint
def dsl_tracker_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_tracker_batch_meta_handler_remove(name, pad)
    return int(result)

##
## dsl_tracker_kitti_output_enabled_set()
##
_dsl.dsl_tracker_kitti_output_enabled_set.argtypes = [c_wchar_p, c_bool, c_wchar_p]
_dsl.dsl_tracker_kitti_output_enabled_set.restype = c_uint
def dsl_tracker_kitti_output_enabled_set(name, enabled, path):
    global _dsl
    result = _dsl.dsl_tracker_kitti_output_enabled_set(name, enabled, path)
    return int(result)

##
## dsl_osd_new()
##
_dsl.dsl_osd_new.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_osd_new.restype = c_uint
def dsl_osd_new(name, is_clock_enabled):
    global _dsl
    result =_dsl.dsl_osd_new(name, is_clock_enabled)
    return int(result)

##
## dsl_osd_clock_enabled_set()
##
_dsl.dsl_osd_clock_enabled_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_osd_clock_enabled_set.restype = c_uint
def dsl_osd_clock_enabled_set(name, enabled):
    global _dsl
    result = _dsl.dsl_osd_clock_enabled_set(name, enabled)
    return int(result)

##
## dsl_osd_clock_enabled_get()
##
_dsl.dsl_osd_clock_enabled_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_osd_clock_enabled_get.restype = c_uint
def dsl_osd_clock_enabled_get(name):
    global _dsl
    enabled = c_bool(False)
    result = _dsl.dsl_osd_clock_enabled_get(name, DSL_BOOL_P(enabled))
    return int(result), enabled.value 

##
## dsl_osd_clock_offsets_get()
##
_dsl.dsl_osd_clock_offsets_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_osd_clock_offsets_get.restype = c_uint
def dsl_osd_clock_offsets_get(name):
    global _dsl
    x_offset = c_uint(0)
    y_offset = c_uint(0)
    result = _dsl.dsl_osd_clock_offsets_get(name, DSL_UINT_P(x_offset), DSL_UINT_P(y_offset))
    return int(result), x_offset.value, y_offset.value 

##
## dsl_osd_clock_offsets_set()
##
_dsl.dsl_osd_clock_offsets_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_osd_clock_offsets_set.restype = c_uint
def dsl_osd_clock_offsets_set(name, x_offset, y_offset):
    global _dsl
    result = _dsl.dsl_osd_clock_offsets_set(name, x_offset, y_offset)
    return int(result)

##
## dsl_osd_clock_font_get()
##
_dsl.dsl_osd_clock_font_get.argtypes = [c_wchar_p, POINTER(c_wchar_p), POINTER(c_uint)]
_dsl.dsl_osd_clock_font_get.restype = c_uint
def dsl_osd_clock_font_get(name):
    global _dsl
    font = c_wchar_p(0)
    size = c_uint(0)
    result = _dsl.dsl_osd_clock_font_get(name, DSL_WCHAR_PP(font), DSL_UINT_P(size))
    return int(result), font.value, size.value 

##
## dsl_osd_clock_font_set()
##
_dsl.dsl_osd_clock_font_set.argtypes = [c_wchar_p, c_wchar_p, c_uint]
_dsl.dsl_osd_clock_font_set.restype = c_uint
def dsl_osd_clock_font_set(name, font, size):
    global _dsl
    result = _dsl.dsl_osd_clock_font_set(name, font, size)
    return int(result)

##
## dsl_osd_clock_color_get()
##
_dsl.dsl_osd_clock_color_get.argtypes = [c_wchar_p, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
_dsl.dsl_osd_clock_color_get.restype = c_uint
def dsl_osd_clock_color_get(name):
    global _dsl
    red = c_double(0)
    green = c_double(0)
    blue = c_double(0)
    alpha = c_double(0)
    result = _dsl.dsl_osd_clock_color_get(name, DSL_DOUBLE_P(red), DSL_DOUBLE_P(green), DSL_DOUBLE_P(blue), DSL_DOUBLE_P(alpha))
    return int(result), red.value, green.value, blue.value, alpha.value

##
## dsl_osd_clock_color_set()
##
_dsl.dsl_osd_clock_color_set.argtypes = [c_wchar_p, c_double, c_double, c_double, c_double]
_dsl.dsl_osd_clock_color_set.restype = c_uint
def dsl_osd_clock_color_set(name, red, green, blue, alpha):
    global _dsl
    result = _dsl.dsl_osd_clock_color_set(name, red, green, blue, alpha)
    return int(result)

##
## dsl_osd_crop_settings_get()
##
_dsl.dsl_osd_crop_settings_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_osd_crop_settings_get.restype = c_uint
def dsl_osd_crop_settings_get(name):
    global _dsl
    left = c_uint(0)
    top = c_uint(0)
    width = c_uint(0)
    height = c_uint(0)
    result = _dsl.dsl_osd_crop_settings_get(name, DSL_UINT_P(left), DSL_UINT_P(top), DSL_UINT_P(width), DSL_UINT_P(height))
    return int(result), left.value, top.value, width.value, height.value 

##
## dsl_osd_crop_settings_set()
##
_dsl.dsl_osd_crop_settings_set.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_osd_crop_settings_set.restype = c_uint
def dsl_osd_crop_settings_set(name, left, top, width, height):
    global _dsl
    result = _dsl.dsl_osd_crop_settings_set(name, left, top, width, height)
    return int(result)

##
## dsl_osd_batch_meta_handler_add()
##
_dsl.dsl_osd_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_osd_batch_meta_handler_add.restype = c_uint
def dsl_osd_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    callbacks.append(meta_handler)
    result = _dsl.dsl_osd_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)
##
## dsl_osd_batch_meta_handler_remove()
##
_dsl.dsl_osd_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_osd_batch_meta_handler_remove.restype = c_uint
def dsl_osd_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_osd_batch_meta_handler_remove(name, pad)
    return int(result)

##
## dsl_tee_demuxer_new()
##
_dsl.dsl_tee_demuxer_new.argtypes = [c_wchar_p]
_dsl.dsl_tee_demuxer_new.restype = c_uint
def dsl_tee_demuxer_new(name):
    global _dsl
    result =_dsl.dsl_tee_demuxer_new(name)
    return int(result)

##
## dsl_tee_demuxer_new_branch_add_many()
##
#_dsl.dsl_tee_demuxer_new_branch_add_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_tee_demuxer_new_branch_add_many.restype = c_uint
def dsl_tee_demuxer_new_branch_add_many(tee, branches):
    global _dsl
    arr = (c_wchar_p * len(branches))()
    arr[:] = branches
    result =_dsl.dsl_tee_demuxer_new_branch_add_many(tee, arr)
    return int(result)

##
## dsl_tee_splitter_new()
##
_dsl.dsl_tee_splitter_new.argtypes = [c_wchar_p]
_dsl.dsl_tee_splitter_new.restype = c_uint
def dsl_tee_splitter_new(name):
    global _dsl
    result =_dsl.dsl_tee_splitter_new(name)
    return int(result)

##
## dsl_tee_splitter_new_branch_add_many()
##
#_dsl.dsl_tee_splitter_new_branch_add_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_tee_splitter_new_branch_add_many.restype = c_uint
def dsl_tee_splitter_new_branch_add_many(tee, branches):
    global _dsl
    arr = (c_wchar_p * len(branches))()
    arr[:] = branches
    result =_dsl.dsl_tee_splitter_new_branch_add_many(tee, arr)
    return int(result)


##
## dsl_tee_branch_add()
##
_dsl.dsl_tee_branch_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_tee_branch_add.restype = c_uint
def dsl_tee_branch_add(tee, branch):
    global _dsl
    result =_dsl.dsl_tee_branch_add(tee, branch)
    return int(result)

##
## dsl_tee_branch_add_many()
##
#_dsl.dsl_tee_branch_add_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_tee_branch_add_many.restype = c_uint
def dsl_tee_branch_add_many(tee, branches):
    global _dsl
    arr = (c_wchar_p * len(branches))()
    arr[:] = branches
    result =_dsl.dsl_tee_branch_add_many(tee, arr)
    return int(result)

##
## dsl_tee_branch_remove()
##
_dsl.dsl_tee_branch_remove.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_tee_branch_remove.restype = c_uint
def dsl_tee_branch_remove(tee, branch):
    global _dsl
    result =_dsl.dsl_tee_branch_remove(tee, branch)
    return int(result)

##
## dsl_tee_branch_remove_many()
##
#_dsl.dsl_tee_branch_remove_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_tee_branch_remove_many.restype = c_uint
def dsl_tee_branch_remove_many(tee, branches):
    global _dsl
    arr = (c_wchar_p * len(branches))()
    arr[:] = branches
    result =_dsl.dsl_tee_branch_remove_many(tee, arr)
    return int(result)
    
##
## dsl_tee_batch_meta_handler_add()
##
_dsl.dsl_tee_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_tee_batch_meta_handler_add.restype = c_uint
def dsl_tee_batch_meta_handler_add(name, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    callbacks.append(meta_handler)
    result = _dsl.dsl_tee_batch_meta_handler_add(name, meta_handler, user_data)
    return int(result)

##
## dsl_tee_batch_meta_handler_remove()
##
_dsl.dsl_tee_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_tee_batch_meta_handler_remove.restype = c_uint
def dsl_tee_batch_meta_handler_remove(name, handler):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    result = _dsl.dsl_tee_batch_meta_handler_remove(name, meta_handler)
    return int(result)

##
## dsl_tiler_new()
##
_dsl.dsl_tiler_new.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_tiler_new.restype = c_uint
def dsl_tiler_new(name, width, height):
    global _dsl
    result =_dsl.dsl_tiler_new(name, width, height)
    return int(result)

##
## dsl_tiler_dimensions_get()
##
_dsl.dsl_tiler_dimensions_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_tiler_dimensions_get.restype = c_uint
def dsl_tiler_dimensions_get(name):
    global _dsl
    width = c_uint(0)
    height = c_uint(0)
    result = _dsl.dsl_tiler_dimensions_get(name, DSL_UINT_P(width), DSL_UINT_P(height))
    return int(result), width.value, height.value 

##
## dsl_tiler_dimensions_set()
##
_dsl.dsl_tiler_dimensions_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_tiler_dimensions_set.restype = c_uint
def dsl_tiler_dimensions_set(name, width, height):
    global _dsl
    result = _dsl.dsl_tiler_dimensions_set(name, width, height)
    return int(result)

##
## dsl_tiler_batch_meta_handler_add()
##
_dsl.dsl_tiler_batch_meta_handler_add.argtypes = [c_wchar_p, c_uint, DSL_META_BATCH_HANDLER, c_void_p]
_dsl.dsl_tiler_batch_meta_handler_add.restype = c_uint
def dsl_tiler_batch_meta_handler_add(name, pad, handler, user_data):
    global _dsl
    meta_handler = DSL_META_BATCH_HANDLER(handler)
    callbacks.append(meta_handler)
    result = _dsl.dsl_tiler_batch_meta_handler_add(name, pad, meta_handler, user_data)
    return int(result)

##
## dsl_tiler_batch_meta_handler_remove()
##
_dsl.dsl_tiler_batch_meta_handler_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_tiler_batch_meta_handler_remove.restype = c_uint
def dsl_tiler_batch_meta_handler_remove(name, pad):
    global _dsl
    result = _dsl.dsl_tiler_batch_meta_handler_remove(name, pad)
    return int(result)

##
## dsl_sink_fake_new()
##
_dsl.dsl_sink_fake_new.argtypes = [c_wchar_p]
_dsl.dsl_sink_fake_new.restype = c_uint
def dsl_sink_fake_new(name):
    global _dsl
    result =_dsl.dsl_sink_fake_new(name)
    return int(result)

##
## dsl_sink_overlay_new()
##
_dsl.dsl_sink_overlay_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_overlay_new.restype = c_uint
def dsl_sink_overlay_new(name, overlay_id, display_id, depth, offsetX, offsetY, width, height):
    global _dsl
    result =_dsl.dsl_sink_overlay_new(name, overlay_id, display_id, depth, offsetX, offsetY, width, height)
    return int(result)

##
## dsl_sink_window_new()
##
_dsl.dsl_sink_window_new.argtypes = [c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_window_new.restype = c_uint
def dsl_sink_window_new(name, offsetX, offsetY, width, height):
    global _dsl
    result =_dsl.dsl_sink_window_new(name, offsetX, offsetY, width, height)
    return int(result)

##
## dsl_sink_file_new()
##
_dsl.dsl_sink_file_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_file_new.restype = c_uint
def dsl_sink_file_new(name, filepath, codec, container, bitrate, interval):
    global _dsl
    result =_dsl.dsl_sink_file_new(name, filepath, codec, container, bitrate, interval)
    return int(result)

##
## dsl_sink_record_new()
##
_dsl.dsl_sink_record_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint, c_uint, DSL_SINK_RECORD_CLIENT_LISTNER]
_dsl.dsl_sink_record_new.restype = c_uint
def dsl_sink_record_new(name, outdir, codec, container, bitrate, interval, client_listener):
    global _dsl
    checker_cb = DSL_SINK_RECORD_CLIENT_LISTNER(client_listener)
    callbacks.append(checker_cb)
    result =_dsl.dsl_sink_record_new(name, outdir, codec, container, bitrate, interval, checker_cb)
    return int(result)
    
##
## dsl_sink_record_session_start()
##
_dsl.dsl_sink_record_session_start.argtypes = [c_wchar_p, POINTER(c_uint), c_uint]
_dsl.dsl_sink_record_session_start.restype = c_uint
def dsl_sink_record_session_start(name, start, duration, client_data):
    global _dsl
    session = c_uint(0)
    result = _dsl.dsl_sink_record_session_start(name, DSL_UINT_P(session), start, duration, client_data)
    return int(result), session.value 

##
## dsl_sink_record_session_stop()
##
_dsl.dsl_sink_record_session_stop.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_sink_record_session_stop.restype = c_uint
def dsl_sink_record_session_stop(name, session):
    global _dsl
    result = _dsl.dsl_sink_record_session_stop(name, session)
    return int(result)

##
## dsl_sink_record_cache_size_get()
##
_dsl.dsl_sink_record_cache_size_get.argtypes = [c_wchar_p, POINTER(c_uint)]
_dsl.dsl_sink_record_cache_size_get.restype = c_uint
def dsl_sink_record_cache_size_get(name):
    global _dsl
    cache_size = c_uint(0)
    result = _dsl.dsl_sink_record_cache_size_get(name, DSL_UINT_P(cache_size))
    return int(result), cache_size.value 

##
## dsl_sink_record_cache_size_set()
##
_dsl.dsl_sink_record_cache_size_set.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_sink_record_cache_size_set.restype = c_uint
def dsl_sink_record_cache_size_set(name, cache_size):
    global _dsl
    result = _dsl.dsl_sink_record_cache_size_set(name, cache_size)
    return int(result)

##
## dsl_sink_record_dimensions_get()
##
_dsl.dsl_sink_record_dimensions_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_sink_record_dimensions_get.restype = c_uint
def dsl_sink_record_dimensions_get(name):
    global _dsl
    width = c_uint(0)
    height = c_uint(0)
    result = _dsl.dsl_sink_record_dimensions_get(name, DSL_UINT_P(width), DSL_UINT_P(height))
    return int(result), width.value, height.value 

##
## dsl_sink_record_dimensions_set()
##
_dsl.dsl_sink_record_dimensions_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_sink_record_dimensions_set.restype = c_uint
def dsl_sink_record_dimensions_set(name, width, height):
    global _dsl
    result = _dsl.dsl_sink_record_dimensions_set(name, width, height)
    return int(result)

##
## dsl_sink_record_is_on_get()
##
_dsl.dsl_sink_record_is_on_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_sink_record_is_on_get.restype = c_uint
def dsl_sink_record_is_on_get(name):
    global _dsl
    is_on = c_uint(0)
    result = _dsl.dsl_sink_record_is_on_get(name, DSL_BOOL_P(is_on))
    return int(result), is_on.value 

##
## dsl_sink_record_reset_done_get()
##
_dsl.dsl_sink_record_reset_done_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_sink_record_reset_done_get.restype = c_uint
def dsl_sink_record_reset_done_get(name):
    global _dsl
    reset_done = c_uint(0)
    result = _dsl.dsl_sink_record_reset_done_get(name, DSL_BOOL_P(reset_done))
    return int(result), reset_done.value 

##
## dsl_sink_encode_video_formats_get()
##
_dsl.dsl_sink_encode_video_formats_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_sink_encode_video_formats_get.restype = c_uint
def dsl_sink_encode_video_formats_get(name):
    global _dsl
    codec = c_uint(0)
    container = c_uint(0)
    result = _dsl.dsl_sink_encode_video_formats_get(name, DSL_UINT_P(codec), DSL_UINT_P(container))
    return int(result), codec.value, container.value 

##
## dsl_sink_encode_settings_get()
##
_dsl.dsl_sink_encode_settings_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_sink_encode_settings_get.restype = c_uint
def dsl_sink_encode_settings_get(name):
    global _dsl
    bitrate = c_uint(0)
    interval = c_uint(0)
    result = _dsl.dsl_sink_encode_settings_get(name, DSL_UINT_P(bitrate), DSL_UINT_P(interval))
    return int(result), bitrate.value, interval.value 

##
## dsl_sink_encode_settings_set()
##
_dsl.dsl_sink_encode_settings_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_sink_encode_settings_set.restype = c_uint
def dsl_sink_encode_settings_set(name, bitrate, interval):
    global _dsl
    result = _dsl.dsl_sink_encode_settings_set(name, bitrate, interval)
    return int(result)

##
## dsl_sink_rtsp_new()
##
_dsl.dsl_sink_rtsp_new.argtypes = [c_wchar_p, c_wchar_p, c_uint, c_uint, c_uint, c_uint, c_uint]
_dsl.dsl_sink_rtsp_new.restype = c_uint
def dsl_sink_rtsp_new(name, host, udp_port, rtsp_port, codec, bitrate, interval):
    global _dsl
    result =_dsl.dsl_sink_rtsp_new(name, host, udp_port, rtsp_port, codec, bitrate, interval)
    return int(result)

##
## dsl_sink_rtsp_server_settings_get()
##
_dsl.dsl_sink_rtsp_server_settings_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_sink_rtsp_server_settings_get.restype = c_uint
def dsl_sink_rtsp_server_settings_get(name):
    global _dsl
    udp_port = c_uint(0)
    rtsp_port = c_uint(0)
    codec = c_uint(0)
    result = _dsl.dsl_sink_rtsp_server_settings_get(name, DSL_UINT_P(udp_port), DSL_UINT_P(rtsp_port), DSL_UINT_P(codec))
    return int(result), udp_port.value, rtsp_port.value, codec.value 

##
## dsl_sink_rtsp_encoder_settings_get()
##
_dsl.dsl_sink_rtsp_encoder_settings_get.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
_dsl.dsl_sink_rtsp_encoder_settings_get.restype = c_uint
def dsl_sink_rtsp_encoder_settings_get(name):
    global _dsl
    bitrate = c_uint(0)
    interval = c_uint(0)
    result = _dsl.dsl_sink_rtsp_encoder_settings_get(name, DSL_UINT_P(bitrate), DSL_UINT_P(interval))
    return int(result), bitrate.value, interval.value 

##
## dsl_sink_rtsp_encoder_settings_set()
##
_dsl.dsl_sink_rtsp_encoder_settings_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_sink_rtsp_encoder_settings_set.restype = c_uint
def dsl_sink_rtsp_encoder_settings_set(name, bitrate, interval):
    global _dsl
    result = _dsl.dsl_sink_rtsp_encoder_settings_set(name, bitrate, interval)
    return int(result)

##
## dsl_sink_image_new()
##
_dsl.dsl_sink_image_new.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_sink_image_new.restype = c_uint
def dsl_sink_image_new(name, outdir):
    global _dsl
    result =_dsl.dsl_sink_image_new(name, outdir)
    return int(result)

##
## dsl_sink_image_outdir_get()
##
_dsl.dsl_sink_image_outdir_get.argtypes = [c_wchar_p, POINTER(c_wchar_p)]
_dsl.dsl_sink_image_outdir_get.restype = c_uint
def dsl_sink_image_outdir_get(name):
    global _dsl
    outdir = c_wchar_p(0)
    result = _dsl.dsl_sink_image_outdir_get(name, DSL_WCHAR_PP(outdir))
    return int(result), outdir.value 

##
## dsl_sink_image_outdir_set()
##
_dsl.dsl_sink_image_outdir_set.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_sink_image_outdir_set.restype = c_uint
def dsl_sink_image_outdir_set(name, outdir):
    global _dsl
    result = _dsl.dsl_sink_image_outdir_set(name, outdir)
    return int(result)

##
## dsl_sink_image_frame_capture_interval_get()
##
_dsl.dsl_sink_image_frame_capture_interval_get.argtypes = [c_wchar_p, POINTER(c_uint)]
_dsl.dsl_sink_image_frame_capture_interval_get.restype = c_uint
def dsl_sink_image_frame_capture_interval_get(name):
    global _dsl
    interval = c_uint(0)
    result = _dsl.dsl_sink_image_frame_capture_interval_get(name, DSL_UINT_P(interval))
    return int(result), interval.value 

##
## dsl_sink_image_frame_capture_interval_set()
##
_dsl.dsl_sink_image_frame_capture_interval_set.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_sink_image_frame_capture_interval_set.restype = c_uint
def dsl_sink_image_frame_capture_interval_set(name, interval):
    global _dsl
    result = _dsl.dsl_sink_image_frame_capture_interval_set(name, interval)
    return int(result)

##
## dsl_sink_image_frame_capture_enabled_get()
##
_dsl.dsl_sink_image_frame_capture_enabled_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_sink_image_frame_capture_enabled_get.restype = c_uint
def dsl_sink_image_frame_capture_enabled_get(name):
    global _dsl
    enabled = c_bool(0)
    result = _dsl.dsl_sink_image_frame_capture_enabled_get(name, DSL_BOOL_P(enabled))
    return int(result), enabled.value 

##
## dsl_sink_image_frame_capture_enabled_set()
##
_dsl.dsl_sink_image_frame_capture_enabled_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_sink_image_frame_capture_enabled_set.restype = c_uint
def dsl_sink_image_frame_capture_enabled_set(name, enabled):
    global _dsl
    result = _dsl.dsl_sink_image_frame_capture_enabled_set(name, enabled)
    return int(result)

##
## dsl_sink_image_object_capture_enabled_get()
##
_dsl.dsl_sink_image_object_capture_enabled_get.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_sink_image_object_capture_enabled_get.restype = c_uint
def dsl_sink_image_object_capture_enabled_get(name):
    global _dsl
    enabled = c_bool(0)
    result = _dsl.dsl_sink_image_object_capture_enabled_get(name, DSL_BOOL_P(enabled))
    return int(result), enabled.value 

##
## dsl_sink_image_object_capture_enabled_set()
##
_dsl.dsl_sink_image_object_capture_enabled_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_sink_image_object_capture_enabled_set.restype = c_uint
def dsl_sink_image_object_capture_enabled_set(name, enabled):
    global _dsl
    result = _dsl.dsl_sink_image_object_capture_enabled_set(name, enabled)
    return int(result)

##
## dsl_sink_image_object_capture_class_add()
##
_dsl.dsl_sink_image_object_capture_class_add.argtypes = [c_wchar_p, c_uint, c_bool, c_uint]
_dsl.dsl_sink_image_object_capture_class_add.restype = c_uint
def dsl_sink_image_object_capture_class_add(name, class_id, full_frame, capture_limit):
    global _dsl
    enabled = c_bool(0)
    result = _dsl.dsl_sink_image_object_capture_class_add(name, class_id, full_frame, capture_limit)
    return int(result)

##
## dsl_sink_image_object_capture_class_remove()
##
_dsl.dsl_sink_image_object_capture_class_remove.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_sink_image_object_capture_class_remove.restype = c_uint
def dsl_sink_image_object_capture_class_remove(name, class_id):
    global _dsl
    result = _dsl.dsl_sink_image_object_capture_class_remove(name, class_id)
    return int(result)

##
## dsl_sink_num_in_use_get()
##
_dsl.dsl_sink_num_in_use_get.restype = c_uint
def dsl_sink_num_in_use_get():
    global _dsl
    result = _dsl.dsl_sink_num_in_use_get()
    return int(result)

##
## dsl_sink_num_in_use_max_get()
##
_dsl.dsl_sink_num_in_use_max_get.restype = c_uint
def dsl_sink_num_in_use_max_get():
    global _dsl
    result = _dsl.dsl_sink_num_in_use_max_get()
    return int(result)

##
## dsl_sink_num_in_use_max_set()
##
_dsl.dsl_sink_num_in_use_max_set.argtypes = [c_uint]
def dsl_sink_num_in_use_max_set(max):
    global _dsl
    result = _dsl.dsl_sink_num_in_use_max_set(max)
dsl_sink_num_in_use_max_set(20)

##
## dsl_component_delete()
##
_dsl.dsl_component_delete.argtypes = [c_wchar_p]
_dsl.dsl_component_delete.restype = c_uint
def dsl_component_delete(name):
    global _dsl
    result =_dsl.dsl_component_delete(name)
    return int(result)

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

##
## dsl_component_delete_all()
##
_dsl.dsl_component_delete_all.restype = c_uint
def dsl_component_delete_all():
    global _dsl
    result =_dsl.dsl_component_delete_all()
    return int(result)

##
## dsl_component_list_size()
##
_dsl.dsl_component_list_size.restype = c_uint
def dsl_component_list_size():
    global _dsl
    result =_dsl.dsl_component_list_size()
    return int(result)

##
## dsl_component_gpuid_get()
##
_dsl.dsl_component_gpuid_get.argtypes = [c_wchar_p, POINTER(c_uint)]
_dsl.dsl_component_gpuid_get.restype = c_uint
def dsl_component_gpuid_get(name):
    global _dsl
    gpuid = c_uint(0)
    result = _dsl.dsl_component_gpuid_get(name, DSL_UINT_P(gpuid))
    return int(result), gpuid.value

##
## dsl_component_gpuid_set()
##
_dsl.dsl_component_gpuid_set.argtypes = [c_wchar_p, c_uint]
_dsl.dsl_component_gpuid_set.restype = c_uint
def dsl_component_gpuid_set(name):
    global _dsl
    result =_dsl.dsl_component_gpuid_set(gpuid)
    return int(result)

##
## dsl_component_gpuid_set_many()
##
#_dsl.dsl_component_gpuid_set_many.argtypes = [Array]
_dsl.dsl_component_gpuid_set_many.restype = c_uint
def dsl_component_gpuid_set_many(components, gpuid):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_component_gpuid_set_many(arr, gpuid)
    return int(result)

##
## dsl_branch_new()
##
_dsl.dsl_branch_new.argtypes = [c_wchar_p]
_dsl.dsl_branch_new.restype = c_uint
def dsl_branch_new(name):
    global _dsl
    result =_dsl.dsl_branch_new(name)
    return int(result)

##
## dsl_branch_new_many()
##
#_dsl.dsl_branch_new_many.argtypes = []
_dsl.dsl_branch_new_many.restype = c_uint
def dsl_branch_new_many(branches):
    global _dsl
    arr = (c_wchar_p * len(branches))()
    arr[:] = branches
    result =_dsl.dsl_branch_new_many(arr)
    return int(result)

##
## dsl_branch_new_component_add_many()
##
_dsl.dsl_branch_new_component_add_many.restype = c_uint
def dsl_branch_new_component_add_many(branch, components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_branch_new_component_add_many(branch, arr)
    return int(result)

##
## dsl_branch_component_add()
##
_dsl.dsl_branch_component_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_branch_component_add.restype = c_uint
def dsl_branch_component_add(branch, component):
    global _dsl
    result =_dsl.dsl_branch_component_add(branch, component)
    return int(result)

##
## dsl_branch_component_add_many()
##
#_dsl.dsl_branch_component_add_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_branch_component_add_many.restype = c_uint
def dsl_branch_component_add_many(branch, components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_branch_component_add_many(branch, arr)
    return int(result)

##
## dsl_branch_component_remove()
##
_dsl.dsl_branch_component_remove.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_branch_component_remove.restype = c_uint
def dsl_branch_component_remove(branch, component):
    global _dsl
    result =_dsl.dsl_branch_component_remove(branch, component)
    return int(result)

##
## dsl_branch_component_remove_many()
##
#_dsl.dsl_branch_component_remove_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_branch_component_remove_many.restype = c_uint
def dsl_branch_component_remove_many(branch, components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_branch_component_remove_many(branch, arr)
    return int(result)

##
## dsl_pipeline_new()
##
_dsl.dsl_pipeline_new.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_new.restype = c_uint
def dsl_pipeline_new(name):
    global _dsl
    result =_dsl.dsl_pipeline_new(name)
    return int(result)

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

##
## dsl_pipeline_new_component_add_many()
##
#_dsl.dsl_pipeline_new_component_add_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_new_component_add_many.restype = c_uint
def dsl_pipeline_new_component_add_many(pipeline, components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_pipeline_new_component_add_many(pipeline, arr)
    return int(result)

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

##
## dsl_pipeline_delete_all()
##
_dsl.dsl_pipeline_delete_all.restype = c_uint
def dsl_pipeline_delete_all():
    global _dsl
    result =_dsl.dsl_pipeline_delete_all()
    return int(result)

##
## dsl_pipeline_list_size()
##
_dsl.dsl_pipeline_list_size.restype = c_uint
def dsl_pipeline_list_size():
    global _dsl
    result =_dsl.dsl_pipeline_list_size()
    return int(result)

##
## dsl_pipeline_component_add()
##
_dsl.dsl_pipeline_component_add.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_component_add.restype = c_uint
def dsl_pipeline_component_add(pipeline, component):
    global _dsl
    result =_dsl.dsl_pipeline_component_add(pipeline, component)
    return int(result)

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

##
## dsl_pipeline_component_remove()
##
_dsl.dsl_pipeline_component_remove.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_component_remove.restype = c_uint
def dsl_pipeline_component_remove(pipeline, component):
    global _dsl
    result =_dsl.dsl_pipeline_component_remove(pipeline, component)
    return int(result)

##
## dsl_pipeline_component_remove_many()
##
#_dsl.dsl_pipeline_component_remove_many.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_component_remove_many.restype = c_uint
def dsl_pipeline_component_remove_many(pipeline, components):
    global _dsl
    arr = (c_wchar_p * len(components))()
    arr[:] = components
    result =_dsl.dsl_pipeline_component_remove_many(pipeline, arr)
    return int(result)

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

##
## dsl_pipeline_streammux_batch_properties_set()
##
_dsl.dsl_pipeline_streammux_batch_properties_set.argtypes = [c_wchar_p, c_uint, c_uint]
_dsl.dsl_pipeline_streammux_batch_properties_set.restype = c_uint
def dsl_pipeline_streammux_batch_properties_set(name, batch_size, batch_timeout):
    global _dsl
    result = _dsl.dsl_pipeline_streammux_batch_properties_set(name, batch_size, batch_timeout)
    return int(result)

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

##
## dsl_pipeline_streammux_padding_set()
##
_dsl.dsl_pipeline_streammux_padding_set.argtypes = [c_wchar_p, c_bool]
_dsl.dsl_pipeline_streammux_padding_set.restype = c_uint
def dsl_pipeline_streammux_padding_set(name, enabled):
    global _dsl
    result = _dsl.dsl_pipeline_streammux_padding_set(name, enabled)
    return int(result)

##
## dsl_pipeline_xwindow_clear()
##
_dsl.dsl_pipeline_xwindow_clear.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_xwindow_clear.restype = c_uint
def dsl_pipeline_xwindow_clear(name):
    global _dsl
    result = _dsl.dsl_pipeline_xwindow_clear(name)
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
    return int(result), int(width.value), int(height.value) 

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

##
## dsl_pipeline_play()
##
_dsl.dsl_pipeline_play.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_play.restype = c_uint
def dsl_pipeline_play(name):
    global _dsl
    result =_dsl.dsl_pipeline_play(name)
    return int(result)

##
## dsl_pipeline_stop()
##
_dsl.dsl_pipeline_stop.argtypes = [c_wchar_p]
_dsl.dsl_pipeline_stop.restype = c_uint
def dsl_pipeline_stop(name):
    global _dsl
    result =_dsl.dsl_pipeline_stop(name)
    return int(result)

##
## dsl_pipeline_state_get()
##
_dsl.dsl_pipeline_state_get.argtypes = [c_wchar_p, POINTER(c_uint)]
_dsl.dsl_pipeline_state_get.restype = c_uint
def dsl_pipeline_state_get(name):
    global _dsl
    state = c_uint(0)
    result =_dsl.dsl_pipeline_state_get(name,  DSL_UINT_P(state))
    return int(result), int(state.value)

##
## dsl_pipeline_is_live()
##
_dsl.dsl_pipeline_is_live.argtypes = [c_wchar_p, POINTER(c_bool)]
_dsl.dsl_pipeline_is_live.restype = c_uint
def dsl_pipeline_is_live(name):
    global _dsl
    is_live = c_bool(0)
    result =_dsl.dsl_pipeline_is_live(name,  DSL_BOOL_P(is_live))
    return int(result), is_live.value

##
## dsl_pipeline_dump_to_dot()
##
_dsl.dsl_pipeline_dump_to_dot.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_dump_to_dot.restype = c_uint
def dsl_pipeline_dump_to_dot(pipeline, filename):
    global _dsl
    result =_dsl.dsl_pipeline_dump_to_dot(pipeline, filename)
    return int(result)

##
## dsl_pipeline_dump_to_dot_with_ts()
##
_dsl.dsl_pipeline_dump_to_dot_with_ts.argtypes = [c_wchar_p, c_wchar_p]
_dsl.dsl_pipeline_dump_to_dot_with_ts.restype = c_uint
def dsl_pipeline_dump_to_dot_with_ts(pipeline, filename):
    global _dsl
    result =_dsl.dsl_pipeline_dump_to_dot_with_ts(pipeline, filename)
    return int(result)

##
## dsl_pipeline_state_change_listener_add()
##
_dsl.dsl_pipeline_state_change_listener_add.argtypes = [c_wchar_p, DSL_STATE_CHANGE_LISTENER, c_void_p]
_dsl.dsl_pipeline_state_change_listener_add.restype = c_uint
def dsl_pipeline_state_change_listener_add(name, listener, user_data):
    global _dsl
    client_listener = DSL_STATE_CHANGE_LISTENER(listener)
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
    result = _dsl.dsl_pipeline_state_change_listener_remove(name, client_listener)
    return int(result)

##
## dsl_pipeline_eos_listener_add()
##
_dsl.dsl_pipeline_eos_listener_add.argtypes = [c_wchar_p, DSL_EOS_LISTENER, c_void_p]
_dsl.dsl_pipeline_eos_listener_add.restype = c_uint
def dsl_pipeline_eos_listener_add(name, listener, user_data):
    global _dsl
    client_listener = DSL_EOS_LISTENER(listener)
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
    result = _dsl.dsl_pipeline_eos_listener_remove(name, client_listener)
    return int(result)

##
## dsl_pipeline_xwindow_key_event_handler_add()
##
_dsl.dsl_pipeline_xwindow_key_event_handler_add.argtypes = [c_wchar_p, DSL_XWINDOW_KEY_EVENT_HANDLER, c_void_p]
_dsl.dsl_pipeline_xwindow_key_event_handler_add.restype = c_uint
def dsl_pipeline_xwindow_key_event_handler_add(name, handler, user_data):
    global _dsl
    client_handler = DSL_XWINDOW_KEY_EVENT_HANDLER(handler)
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
    result = _dsl.dsl_pipeline_xwindow_key_event_handler_remove(name, client_handler)
    return int(result)

##
## dsl_pipeline_xwindow_button_event_handler_add()
##
_dsl.dsl_pipeline_xwindow_button_event_handler_add.argtypes = [c_wchar_p, DSL_XWINDOW_BUTTON_EVENT_HANDLER, c_void_p]
_dsl.dsl_pipeline_xwindow_button_event_handler_add.restype = c_uint
def dsl_pipeline_xwindow_button_event_handler_add(name, handler, user_data):
    global _dsl
    client_handler = DSL_XWINDOW_BUTTON_EVENT_HANDLER(handler)
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
    result = _dsl.dsl_pipeline_xwindow_button_event_handler_remove(name, client_handler)
    return int(result)

##
## dsl_pipeline_xwindow_delete_event_handler_add()
##
_dsl.dsl_pipeline_xwindow_delete_event_handler_add.argtypes = [c_wchar_p, DSL_XWINDOW_DELETE_EVENT_HANDLER, c_void_p]
_dsl.dsl_pipeline_xwindow_delete_event_handler_add.restype = c_uint
def dsl_pipeline_xwindow_delete_event_handler_add(name, handler, user_data):
    global _dsl
    client_handler = DSL_XWINDOW_DELETE_EVENT_HANDLER(handler)
    callbacks.append(client_handler)
    result = _dsl.dsl_pipeline_xwindow_delete_event_handler_add(name, client_handler, user_data)
    return int(result)

##
## dsl_pipeline_xwindow_delete_event_handler_remove()
##
_dsl.dsl_pipeline_xwindow_delete_event_handler_remove.argtypes = [c_wchar_p, DSL_XWINDOW_DELETE_EVENT_HANDLER]
_dsl.dsl_pipeline_xwindow_delete_event_handler_remove.restype = c_uint
def dsl_pipeline_xwindow_delete_event_handler_remove(name, handler):
    global _dsl
    client_handler = DSL_XWINDOW_DELETE_EVENT_HANDLER(handler)
    result = _dsl.dsl_pipeline_xwindow_delete_event_handler_remove(name, client_handler)
    return int(result)

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

##
## dsl_return_value_to_string()
##
_dsl.dsl_return_value_to_string.argtypes = [c_uint]
_dsl.dsl_return_value_to_string.restype = c_wchar_p
def dsl_return_value_to_string(result):
    global _dsl
    return _dsl.dsl_return_value_to_string(result)

##
## dsl_state_value_to_string()
##
_dsl.dsl_state_value_to_string.argtypes = [c_uint]
_dsl.dsl_state_value_to_string.restype = c_wchar_p
def dsl_state_value_to_string(state):
    global _dsl
    return _dsl.dsl_state_value_to_string(state)

##
## dsl_version_get()
##
_dsl.dsl_version_get.restype = c_wchar_p
def dsl_version_get():
    global _dsl
    return _dsl.dsl_version_get()

##
## dsl_delete_all()
##
_dsl.dsl_delete_all.restype = c_bool
def dsl_delete_all():
    global _dsl
    return _dsl.dsl_delete_all()