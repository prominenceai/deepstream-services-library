import sys
sys.path.insert(0, "../../")
import time

from dsl import *

##
## dsl_source_csi_new()
##
print("dsl_source_csi_new")
print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
print(dsl_component_delete("csi-source"))

##
## dsl_source_usb_new()
##
print("dsl_source_usb_new")
print(dsl_source_usb_new("usb-source", 1280, 720, 30, 1))
print(dsl_component_delete("usb-source"))

##
## dsl_source_uri_new()
##
print("dsl_source_uri_new")
print(dsl_source_uri_new("uri-source", "../../test/streams/sample_1080p_h264.mp4", False, 0, 0, 0))
print(dsl_component_delete("uri-source"))

##
## dsl_source_rtsp_new()
##
#print("dsl_source_rtsp_new")
#print(dsl_source_rtsp_new("rtsp-source", "???????", DSL_RTP_ALL, 0, 0, 0, 100))
#print(dsl_component_delete("rtsp-source"))

##
## dsl_source_dimensions_get()
##
print("dsl_source_dimensions_get")
print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
print(dsl_source_dimensions_get("csi-source"))
print(dsl_component_delete("csi-source"))

##
## dsl_source_frame_rate_get()
##
print("dsl_source_frame_rate_get")
print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
print(dsl_source_frame_rate_get("csi-source"))
print(dsl_component_delete("csi-source"))

##
## dsl_source_osd_add()
## dsl_source_osd_remove()
##
print("dsl_source_osd_add")
print("dsl_source_osd_remove")
print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
print(dsl_osd_new("osd", 0, 0, 1280, 720))
print(dsl_source_osd_add("csi-source", "osd"))
print(dsl_source_osd_remove("csi-source"))
print(dsl_component_delete("csi-source"))
print(dsl_component_delete("osd"))

##
## dsl_source_sink_add()
## dsl_source_sink_remove()
##
print("dsl_source_sink_add")
print("dsl_source_sink_remove")
print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
print(dsl_sink_window_new("overlay-sink", 0, 0, 1280, 720))
print(dsl_source_sink_add("csi-source", "overlay-sink"))
print(dsl_source_sink_remove("csi-source", "overlay-sink"))
print(dsl_component_delete("csi-source"))
print(dsl_component_delete("overlay-sink"))

##
## dsl_source_decode_uri_get()
## dsl_source_decode_uri_set()
##
print("dsl_source_decode_uri_get")
print("dsl_source_decode_uri_set")
print(dsl_source_uri_new("uri-source", "../../test/streams/sample_1080p_h264.mp4", False, 0, 0, 0))
print(dsl_source_decode_uri_set("uri-source", "../../test/streams/sample_1080p_h264.mp4"))
print(dsl_source_decode_uri_get("uri-source"))
print(dsl_component_delete("uri-source"))

##
## dsl_source_is_live()
##
print("dsl_source_is_live")
print(dsl_source_csi_new("csi-source", 1280, 720, 30, 1))
print(dsl_source_is_live("csi-source"))
print(dsl_component_delete("csi-source"))

##
## dsl_source_num_in_use_get()
##
print("dsl_source_num_in_use_get")
print(dsl_source_num_in_use_get())

##
## dsl_source_num_in_use_max_get()
##
print("dsl_source_num_in_use_max_get")
print(dsl_source_num_in_use_max_get())

##
## dsl_source_num_in_use_max_set()
##
print("dsl_source_num_in_use_max_set")
print(dsl_source_num_in_use_max_set(20))

##
## dsl_dewarper_new()
##
print("dsl_dewarper_new")
print(dsl_dewarper_new("dewarper", "./test/configs/config_dewarper.txt"))
print(dsl_component_delete("dewarper"))

##
## dsl_gie_primary_new()
##
print("dsl_gie_primary_new")
print(dsl_gie_primary_new("primary-gie", "./test/configs/config_infer_primary_nano.txt", 
    "./test/models/Primary_Detector_Nano/resnet10.caffemodel", 0))
print(dsl_component_delete("primary-gie"))

##
## dsl_gie_primary_batch_meta_handler_add()
## dsl_gie_primary_batch_meta_handler_remove()
##
print("dsl_gie_primary_batch_meta_handler_add")
print("dsl_gie_primary_batch_meta_handler_remove")
def mb_handler(buffer, user_data):
    print(buffer)
    print(user_data)
print(dsl_gie_primary_new("primary-gie", "./test/configs/config_infer_primary_nano.txt", 
    "./test/models/Primary_Detector_Nano/resnet10.caffemodel", 0))
print(dsl_gie_primary_batch_meta_handler_add("primary-gie", DSL_PAD_SRC, mb_handler, None))
print(dsl_gie_primary_batch_meta_handler_remove("primary-gie", DSL_PAD_SRC, mb_handler))
print(dsl_component_delete("primary-gie"))

##
## dsl_gie_primary_kitti_output_enabled_set()
##
print("dsl_gie_primary_kitti_output_enabled_set")
print(dsl_gie_primary_new("primary-gie", "./test/configs/config_infer_primary_nano.txt", 
    "./test/models/Primary_Detector_Nano/resnet10.caffemodel", 0))
print(dsl_gie_primary_kitti_output_enabled_set("primary-gie", True, "./"))
print(dsl_gie_primary_kitti_output_enabled_set("primary-gie", False, ""))
print(dsl_component_delete("primary-gie"))

##
## dsl_gie_secondary_new()
##
print("dsl_gie_secondary_new")
print(dsl_gie_secondary_new("secondary-gie", "./test/configs/config_infer_secondary_carcolor_nano.txt", 
    "./test/models/Secondary_CarColor/resnet18.caffemodel_b16_fp16.engine", "primary-gie", 0))
print(dsl_component_delete("secondary-gie"))

##
## dsl_gie_raw_output_enabled_set()
##
print("dsl_gie_raw_output_enabled_set")
print(dsl_gie_primary_new("primary-gie", "./test/configs/config_infer_primary_nano.txt", 
    "./test/models/Primary_Detector_Nano/resnet10.caffemodel", 0))
print(dsl_gie_raw_output_enabled_set("primary-gie", True, "./"))
print(dsl_gie_raw_output_enabled_set("primary-gie", False, ""))
print(dsl_component_delete("primary-gie"))

##
## dsl_tracker_ktl_new()
##
print("dsl_tracker_ktl_new")
print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))
print(dsl_component_delete("ktl-tracker"))

##
## dsl_tracker_iou_new()
##
print("dsl_tracker_iou_new")
print(dsl_tracker_iou_new("iou-tracker", "./test/configs/iou_config.txt", 300, 150))
print(dsl_component_delete("iou-tracker"))

##
## dsl_tracker_max_dimensions_set()
## dsl_tracker_max_dimensions_get()
##
print("dsl_tracker_max_dimensions_set")
print("dsl_tracker_max_dimensions_get")
print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))
print(dsl_tracker_max_dimensions_set("ktl-tracker", 325, 200))
print(dsl_tracker_max_dimensions_get("ktl-tracker",))
print(dsl_component_delete("ktl-tracker"))

##
## dsl_tracker_batch_meta_handler_add()
## dsl_tracker_batch_meta_handler_remove()
##
print("dsl_tracker_batch_meta_handler_add")
print("dsl_tracker_batch_meta_handler_remove")
def mb_handler(buffer, user_data):
    print(buffer)
    print(user_data)
print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))
print(dsl_tracker_batch_meta_handler_add("ktl-tracker", DSL_PAD_SRC, mb_handler, None))
print(dsl_tracker_batch_meta_handler_remove("ktl-tracker", DSL_PAD_SRC, mb_handler))
print(dsl_component_delete("ktl-tracker"))

##
## dsl_tracker_kitti_output_enabled_set()
##
print("dsl_tracker_kitti_output_enabled_set")
print(dsl_tracker_ktl_new("ktl-tracker", 300, 150))
print(dsl_tracker_kitti_output_enabled_set("ktl-tracker", True, "./"))
print(dsl_tracker_kitti_output_enabled_set("ktl-tracker", False, ""))
print(dsl_component_delete("ktl-tracker"))

##
## dsl_osd_new()
##
print("dsl_osd_new")
print(dsl_osd_new("on-screen-display", False))
print(dsl_component_delete("on-screen-display"))

##
## dsl_osd_clock_enabled_get()
## dsl_osd_clock_enabled_set()
##
print("dsl_osd_clock_enabled_get")
print("dsl_osd_clock_enabled_set")
print(dsl_osd_new("on-screen-display", False))
print(dsl_osd_clock_enabled_set("on-screen-display", True))
print(dsl_osd_clock_enabled_get("on-screen-display"))
print(dsl_component_delete("on-screen-display"))

##
## dsl_osd_clock_offsets_get()
## dsl_osd_clock_offsets_set()
##
print("dsl_osd_clock_offsets_get")
print("dsl_osd_clock_offsets_set")
print(dsl_osd_new("on-screen-display", False))
print(dsl_osd_clock_offsets_set("on-screen-display", 100, 100))
print(dsl_osd_clock_offsets_get("on-screen-display",))
print(dsl_component_delete("on-screen-display"))

##
## dsl_osd_clock_font_get()
## dsl_osd_clock_font_set()
##
print("dsl_osd_clock_font_get")
print("dsl_osd_clock_font_set")
print(dsl_osd_new("on-screen-display", False))
print(dsl_osd_clock_font_set("on-screen-display", 'ariel', 16))
print(dsl_osd_clock_font_get("on-screen-display",))
print(dsl_component_delete("on-screen-display"))

##
## dsl_osd_clock_color_get()
## dsl_osd_clock_color_set()
##
print("dsl_osd_clock_color_get")
print("dsl_osd_clock_color_set")
print(dsl_osd_new("on-screen-display", False))
print(dsl_osd_clock_color_set("on-screen-display", 255, 255, 255))
print(dsl_osd_clock_color_get("on-screen-display",))
print(dsl_component_delete("on-screen-display"))

##
## dsl_osd_batch_meta_handler_add()
## dsl_osd_batch_meta_handler_remove()
##
print("dsl_osd_batch_meta_handler_add")
print("dsl_osd_batch_meta_handler_remove")
def mb_handler(buffer, user_data):
    print(buffer)
    print(user_data)
print(dsl_osd_new("on-screen-display", True))
print(dsl_osd_batch_meta_handler_add("on-screen-display", DSL_PAD_SRC, mb_handler, None))
print(dsl_osd_batch_meta_handler_remove("on-screen-display", DSL_PAD_SRC, mb_handler))
print(dsl_component_delete("on-screen-display"))

##
## dsl_demuxer_new()
##
print("dsl_demuxer_new")
print(dsl_demuxer_new("demuxer"))
print(dsl_component_delete("demuxer"))

##
## dsl_demuxer_batch_meta_handler_add()
## dsl_demuxer_batch_meta_handler_remove()
##
print("dsl_demuxer_batch_meta_handler_add")
print("dsl_demuxer_batch_meta_handler_remove")
def mb_handler(buffer, user_data):
    print(buffer)
    print(user_data)
print(dsl_demuxer_new("demuxer")
print(dsl_demuxer_batch_meta_handler_add("demuxer", mb_handler, None))
print(dsl_demuxer_batch_meta_handler_remove("demuxer", mb_handler))
print(dsl_component_delete("demuxer"))

##
## dsl_tiler_new()
##
print("dsl_tiler_new")
print(dsl_tiler_new("tiler", 1280, 720))
print(dsl_component_delete("tiler"))

##
## dsl_tiler_batch_meta_handler_add()
## dsl_tiler_batch_meta_handler_remove()
##
print("dsl_tiler_batch_meta_handler_add")
print("dsl_tiler_batch_meta_handler_remove")
def mb_handler(buffer, user_data):
    print(buffer)
    print(user_data)
print(dsl_tiler_new("tiler", 1280, 720))
print(dsl_tiler_batch_meta_handler_add("tiler", DSL_PAD_SRC, mb_handler, None))
print(dsl_tiler_batch_meta_handler_remove("tiler", DSL_PAD_SRC, mb_handler))
print(dsl_component_delete("tiler"))

##
## dsl_sink_overlay_new()
##
print("dsl_sink_overlay_new")
print(dsl_sink_overlay_new("overlay-sink", 0, 0, 1280, 720))
print(dsl_component_delete("overlay-sink"))

##
## dsl_sink_window_new()
##
print("dsl_sink_window_new")
print(dsl_sink_window_new("window-sink", 0, 0, 1280, 720))
print(dsl_component_delete("window-sink"))

##
## dsl_sink_file_new()
##
print("dsl_sink_file_new")
print(dsl_sink_file_new("file-sink", "./output.mp4", DSL_CODEC_H265, DSL_CONTAINER_MPEG4, 2000000, 1))
print(dsl_component_delete(" file-sink"))

##
## dsl_sink_file_video_formats_get()
##
print("dsl_sink_file_video_formats_get")
print(dsl_sink_file_new("file-sink", "./output.mp4", DSL_CODEC_H265, DSL_CONTAINER_MPEG4, 2000000, 1))
print(dsl_sink_file_video_formats_get("file-sink"))
print(dsl_component_delete(" file-sink"))

##
## dsl_sink_file_encoder_settings_get()
## dsl_sink_file_encoder_settings_set()
##
print("dsl_sink_file_encoder_settings_get")
print("dsl_sink_file_encoder_settings_set")
print(dsl_sink_file_new("file-sink", "./output.mp4", DSL_CODEC_H265, DSL_CONTAINER_MPEG4, 2000000, 1))
print(dsl_sink_file_encoder_settings_set("file-sink", 2500000, 5))
print(dsl_sink_file_encoder_settings_get("file-sink"))
print(dsl_component_delete(" file-sink"))

##
## dsl_sink_rtsp_new()
##
print("dsl_sink_rtsp_new")
print(dsl_sink_rtsp_new("rtsp-sink", "224.224.255.255", 5400, 8554, DSL_CODEC_H265, 4000000, 0))
print(dsl_component_delete(" rtsp-sink"))

##
## dsl_sink_rtsp_server_settings_get()
##
print("dsl_sink_rtsp_server_settings_get")
print(dsl_sink_rtsp_new("rtsp-sink", "224.224.255.255", 5400, 8554, DSL_CODEC_H265, 4000000, 0))
print(dsl_sink_rtsp_server_settings_get("rtsp-sink"))
print(dsl_component_delete(" rtsp-sink"))

##
## dsl_sink_rtsp_encoder_settings_get()
## dsl_sink_rtsp_encoder_settings_set()
##
print("dsl_sink_rtsp_encoder_settings_get")
print("dsl_sink_rtsp_encoder_settings_set")
print(dsl_sink_rtsp_new("rtsp-sink", "224.224.255.255", 5400, 8554, DSL_CODEC_H265, 4000000, 0))
print(dsl_sink_rtsp_encoder_settings_get("rtsp-sink"))
print(dsl_sink_rtsp_encoder_settings_set("rtsp-sink", 4500000, 5))
print(dsl_sink_rtsp_server_settings_get("rtsp-sink"))
print(dsl_component_delete(" rtsp-sink"))

##
## dsl_sink_num_in_use_get()
##
print("dsl_sink_num_in_use_get")
print(dsl_sink_num_in_use_get())

##
## dsl_sink_num_in_use_max_get()
##
print("dsl_sink_num_in_use_max_get")
print(dsl_sink_num_in_use_max_get())

##
## dsl_sink_num_in_use_max_set()
##
print("dsl_sink_num_in_use_max_set")
print(dsl_sink_num_in_use_max_set(20))
print(dsl_sink_num_in_use_max_get())

##
## dsl_component_delete()
## dsl_component_delete_many()
## dsl_component_delete_all()
##
print("dsl_component_delete")
print("dsl_component_delete_many")
print("dsl_component_delete_all")
print(dsl_sink_window_new("window-sink", 0, 0, 1280, 720))
print(dsl_sink_overlay_new("overlay-sink", 0, 0, 1280, 720))
print(dsl_component_delete_many(["window-display", "overlay-sink", None]))
print(dsl_component_delete_all())

##
## dsl_component_list_size()
##
print("dsl_component_list_size")
print(dsl_component_list_size())

##
## dsl_component_gpuid_get()
## dsl_component_gpuid_set()
## dsl_component_gpuid_set_many()
##
print("dsl_component_gpuid_get")
print("dsl_component_gpuid_set")
print("dsl_component_gpuid_set_many")
print(dsl_sink_window_new("window-sink", 0, 0, 1280, 720))
print(dsl_sink_overlay_new("overlay-sink", 0, 0, 1280, 720))
print(dsl_component_gpuid_set_many(["window-sink", "overlay-sink", None], 1))
print(dsl_component_gpuid_get("window-sink"))
print(dsl_component_gpuid_get("overlay-sink"))
print(dsl_component_delete_all())

##
## dsl_pipeline_new()
## dsl_pipeline_delete()
##
print("dsl_pipeline_new")
print("dsl_pipeline_delete")
print(dsl_pipeline_new("pipeline-1"))
print(dsl_pipeline_delete("pipeline-1"))

##
## dsl_pipeline_new_many()
## dsl_pipeline_delete_many()
## dsl_pipeline_delete_all()
##
print("dsl_pipeline_new_many")
print("dsl_pipeline_delete_many")
print("dsl_pipeline_delete_all")
print(dsl_pipeline_new_many(["pipeline-2", "pipeline-3", "pipeline-4", None]))
print(dsl_pipeline_delete_many(["pipeline-2", "pipeline-3", None]))
print(dsl_pipeline_delete_all())

##
## dsl_pipeline_list_size()
##
print("dsl_pipeline_list_size")
print(dsl_pipeline_list_size())

##
## dsl_pipeline_component_add()
##
print("dsl_pipeline_component_add")
print(dsl_tiler_new("tiler", 1280, 720))
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_component_add("pipeline", "tiler"))
print(dsl_pipeline_delete_all())
print(dsl_component_delete_all())

##
## dsl_pipeline_component_add_many()
##
print("dsl_pipeline_component_add_many")
print(dsl_tiler_new("tiler", 1280, 720))
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_component_add_many("pipeline", ["tiler", None]))
print(dsl_pipeline_delete_all())
print(dsl_component_delete_all())

##
## dsl_pipeline_streammux_batch_properties_get()
##
print("dsl_pipeline_streammux_batch_properties_get")
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_streammux_batch_properties_get("pipeline"))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_streammux_dimensions_get()
## dsl_pipeline_streammux_dimensions_set()
##
print("dsl_pipeline_streammux_dimensions_get")
print("dsl_pipeline_streammux_dimensions_set")
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_streammux_dimensions_get("pipeline"))
print(dsl_pipeline_streammux_dimensions_set("pipeline", 1280, 720))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_streammux_padding_get()
## dsl_pipeline_streammux_padding_set()
##
print("dsl_pipeline_streammux_padding_get")
print("dsl_pipeline_streammux_padding_set")
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_streammux_padding_get("pipeline"))
print(dsl_pipeline_streammux_padding_set("pipeline", True))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_xwindow_dimensions_get()
## dsl_pipeline_xwindow_dimensions_set()
##
print("dsl_pipeline_xwindow_dimensions_get")
print("dsl_pipeline_xwindow_dimensions_set")
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_xwindow_dimensions_get("pipeline"))
print(dsl_pipeline_xwindow_dimensions_set("pipeline", 1280, 720))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_play()
## dsl_pipeline_pause()
## dsl_pipeline_stop()
##
print("dsl_pipeline_play")
print("dsl_pipeline_pause")
print("dsl_pipeline_ptop")
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_play("pipeline"))
print(dsl_pipeline_pause("pipeline"))
print(dsl_pipeline_stop("pipeline"))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_dump_to_dot()
##
print("dsl_pipeline_dump_to_dot")
print(dsl_pipeline_dump_to_dot("pipeline", "dot-file-name"))

##
## dsl_pipeline_dump_to_dot_with_ts()
##
print("dsl_pipeline_dump_to_dot_with_ts")
print(dsl_pipeline_dump_to_dot_with_ts("pipeline-1", "dot-file-name"))

##
## dsl_pipeline_state_change_listener_add()
## dsl_pipeline_state_change_listener_remove()
##
print("dsl_pipeline_state_change_listener_add")
print("dsl_pipeline_state_change_listener_remove")
def listener(prev_state, new_state, user_data):
    print(prev_state)
    print(new_state)
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_state_change_listener_add("pipeline", listener, None))
print(dsl_pipeline_state_change_listener_remove("pipeline", listener))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_eos_listener_add()
## dsl_pipeline_eos_listener_remove()
##
print("dsl_pipeline_eos_listener_add")
print("dsl_pipeline_eos_listener_remove")
def listener(user_data):
    print(user_data)
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_eos_listener_add("pipeline", listener, None))
print(dsl_pipeline_eos_listener_remove("pipeline", listener))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_xwindow_key_event_handler_add()
## dsl_pipeline_xwindow_key_event_handler_remove()
##
print("dsl_source_csi_new")
print("dsl_source_csi_new")
def key_handler(key, user_data):
    print(key)
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_xwindow_key_event_handler_add("pipeline", key_handler, None))
print(dsl_pipeline_xwindow_key_event_handler_remove("pipeline", key_handler))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_xwindow_button_event_handler_add()
## dsl_pipeline_xwindow_button_event_handler_remove()
##
print("dsl_pipeline_xwindow_button_event_handler_add")
print("dsl_pipeline_xwindow_button_event_handler_remove")
def button_handler(xpos, ypos, user_data):
    print(xpos)
    print(ypos)
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_xwindow_button_event_handler_add("pipeline", button_handler, None))
print(dsl_pipeline_xwindow_button_event_handler_remove("pipeline", button_handler))
print(dsl_pipeline_delete("pipeline"))

##
## dsl_pipeline_xwindow_delete_event_handler_add()
## dsl_pipeline_xwindow_delete_event_handler_remove()
##
print("dsl_pipeline_xwindow_delete_event_handler_add")
print("dsl_pipeline_xwindow_delete_event_handler_remove")
def delete_handler(prev_state, new_state, user_data):
    print(prev_state)
    print(new_state)
print(dsl_pipeline_new("pipeline"))
print(dsl_pipeline_xwindow_delete_event_handler_add("pipeline", delete_handler, None))
print(dsl_pipeline_xwindow_delete_event_handler_remove("pipeline", delete_handler))

##
## dsl_main_loop_run()
## dsl_main_loop_quit()
##
print("dsl_main_loop_run")
print("dsl_main_loop_quit")

print("dsl_version_get")
print(dsl_version_get())