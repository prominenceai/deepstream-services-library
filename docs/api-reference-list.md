# DSL API Reference

### Pipeline API:
* [dsl_pipeline_new](/docs/api-pipeline.md#dsl_pipeline_new)
* [dsl_pipeline_new_many](/docs/api-pipeline.md#dsl_pipeline_new_many)
* [dsl_pipeline_delete](/docs/api-pipeline.md#dsl_pipeline_delete)
* [dsl_pipeline_delete_many](/docs/api-pipeline.md#dsl_pipeline_delete_many)
* [dsl_pipeline_delete_all](/docs/api-pipeline.md#dsl_pipeline_delete_all)
* [dsl_pipeline_list_size](/docs/api-pipeline.md#dsl_pipeline_list_size)
* [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add)
* [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many)
* [dsl_pipeline_component_list_size](/docs/api-pipeline.md#dsl_pipeline_components_list_size)
* [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove)
* [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many)
* [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all)
* [dsl_pipeline_component_replace](/docs/api-pipeline.md#dsl_pipeline_component_replace)
* [dsl_pipeline_streammux_batch_properties_get](/docs/api-pipeline.md#dsl_pipeline_streammux_properties_get)
* [dsl_pipeline_streammux_dimensions_get](/docs/api-pipeline.md#dsl_pipeline_streammux_dimensions_get)
* [dsl_pipeline_streammux_dimensions_set](/docs/api-pipeline.md#dsl_pipeline_streammux_dimensions_set)
* [dsl_pipeline_xwindow_dimensions_get](/docs/api-pipeline.md#dsl_pipeline_xwindow_dimensions_get)
* [dsl_pipeline_xwindow_dimensions_set](/docs/api-pipeline.md#dsl_pipeline_xwindow_dimensions_set)
* [dsl_pipeline_xwindow_handle_get](/docs/api-pipeline.md#dsl_pipeline_xwindow_handle_get)
* [dsl_pipeline_xwindow_handle_set](/docs/api-pipeline.md#dsl_pipeline_xwindow_handle_set)
* [dsl_pipeline_xwindow_key_event_handler_add](/docs/api-pipeline.md#dsl_pipeline_xwindow_key_event_handler_add)
* [dsl_pipeline_xwindow_key_event_handler_remove](/docs/api-pipeline.md#dsl_pipeline_xwindow_key_event_handler_remove)
* [dsl_pipeline_xwindow_button_event_handler_add](/docs/api-pipeline.md#dsl_pipeline_xwindow_button_event_handler_add)
* [dsl_pipeline_xwindow_button_event_handler_remove](/docs/api-pipeline.md#dsl_pipeline_xwindow_button_event_handler_remove)
* [dsl_pipeline_play](/docs/api-pipeline.md#dsl_pipeline_play)
* [dsl_pipeline_pause](/docs/api-pipeline.md#dsl_pipeline_pause)
* [dsl_pipeline_stop](/docs/api-pipeline.md#dsl_pipeline_stop)
* [dsl_pipeline_state_get](/docs/api-pipeline.md#dsl_pipeline_state_get)
* [dsl_pipeline_state_change_listener_add](/docs/api-pipeline.md#dsl_pipeline_state_change_listener_add)
* [dsl_pipeline_state_change_listener_remove](/docs/api-pipeline.md#dsl_pipeline_state_change_listener_remove)
* [dsl_pipeline_eos_listener_add](/docs/api-pipeline.md#dsl_pipeline_eos_listener_add)
* [dsl_pipeline_eos_listener_remove](/docs/api-pipeline.md#dsl_pipeline_eos_listener_remove)
* [dsl_pipeline_qos_listener_add](/docs/api-pipeline.md#dsl_pipeline_qos_listener_add)
* [dsl_pipeline_qos_listener_remove](/docs/api-pipeline.md#dsl_pipeline_qos_listener_remove)
* [dsl_pipeline_dump_to_dot](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot)
* [dsl_pipeline_dump_to_dot_with_ts](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot_with_ts)
* Other TBD


### Component API:
* dsl_component_copy
* [dsl_component_delete](/docs/api-component.md#dsl_component_delete)
* [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many)
* [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)
* [dsl_component_list_size](/docs/api-component.md#dsl_component_list_size)
* [dsl_component_list_all](/docs/api-component.md#dsl_component_list_all)
* [dsl_component_gpuid_get](/docs/api-component.md#dsl_component_gpuid_get)
* [dsl_component_gpuid_set](/docs/api-component.md#dsl_component_gpuid_set)
* [dsl_component_gpuid_set_many](/docs/api-component.md#dsl_component_gpuid_set_many)
* [dsl_component_is_in_use](/docs/api-component.md#dsl_component_is_in_use)
* Other TBD


### Source API:
* [dsl_source_csi_new](/docs/api-source.md#dsl_source_csi_new)
* [dsl_source_v4l2_new](/docs/api-source.md#dsl_source_v4l2_new)
* [dsl_source_uri_new](/docs/api-source.md#dsl_source_uri_new)
* [dsl_source_uri_raw_output_enable](/docs/api-source.md#dsl_source_uri_raw_output_enable)
* [dsl_source_rtsp_new](/docs/api-source.md#dsl_source_rtsp_new)
* [dsl_source_pause](/docs/api-source.md#dsl_source_pause)
* [dsl_source_play](/docs/api-source.md#dsl_source_play)
* [dsl_source_state_is](/docs/api-source.md#dsl_source_state_is)
* [dsl_source_is_in_use](/docs/api-source.md#dsl_source_is_in_use)
* [dsl_source_is_live](/docs/api-source.md#dsl_source_is_live)
* [dsl_source_get_num_in_use](/docs/api-source.md#dsl_source_get_num_in_use)
* [dsl_source_get_num_in_use_max](/docs/api-source.md#dsl_source_get_num_in_use_max)
* [dsl_source_set_num_in_use_max](/docs/api-source.md#dsl_source_set_num_in_use_max)
* Other TBD

### Primary and Secondary GIE API:
* [dsl_gie_primary_new](/docs/api-gie.md#dsl_gie_primary_new)
* [dsl_gie_secondary_new](/docs/api-gie.md#dsl_gie_secondary_new)
* [dsl_gie_infer_config_file_get](/docs/api-gie.md#dsl_gie_infer_config_file_get)
* [dsl_gie_infer_config_file_set](/docs/api-gie.md#dsl_gie_infer_config_file_set)
* [dsl_gie_model_engine_file_get](/docs/api-gie.md#dsl_gie_model_engine_file_get)
* [dsl_gie_model_engine_file_set](/docs/api-gie.md#dsl_gie_model_engine_file_set)
* [dsl_gie_interval_get](/docs/api-gie.md#dsl_gie_interval_get)
* [dsl_gie_interval_set](/docs/api-gie.md#dsl_gie_interval_set)
* [dsl_gie_secondary_infer_on_get](/docs/api-gie.md#dsl_gie_secondary_infer_on_get)
* [dsl_gie_secondary_infer_on_set](/docs/api-gie.md#dsl_gie_secondary_infer_on_set)
* [dsl_gie_primary_meta_batch_handler_add](/docs/api-gie.md#dsl_gie_primary_meta_batch_handler_add)
* [dsl_gie_primary_meta_batch_handler_remove](/docs/api-gie.md#dsl_gie_primary_meta_batch_handler_remove)
* Other TBD

### Tracker:
* [dsl_tracker_ktl_new](/docs/api-tracker.md#dsl_tracker_ktl_new)
* [dsl_tracker_iou_new](/docs/api-tracker.md#dsl_tracker_iou_new)
* [dsl_tracker_max_dimensions_get](/docs/api-tracker.md#dsl_tracker_dimensions_get)
* [dsl_tracker_max_dimensions_set](/docs/api-tracker.md#dsl_tracker_dimensions_set)
* [dsl_tracker_iou_config_file_get](/docs/api-tracker.md#dsl_tracker_iou_config_file_get)
* [dsl_tracker_iou_config_file_set](/docs/api-tracker.md#dsl_tracker_iou_config_file_set)
* [dsl_tracker_meta_batch_handler_add](/docs/api-tracker.md#dsl_tracker_meta_batch_handler_add)
* [dsl_tracker_meta_batch_handler_remove](/docs/api-tracker.md#dsl_tracker_meta_batch_handler_remove)
* Other TBD

### Tiled Display (Tiler):
* [dsl_display_new](/docs/api-display.md#dsl_display_new)
* [dsl_tiler_new](/docs/api-display.md#dsl_tiler_new)
* [dsl_tiler_dimensions_get](/docs/api-display.md#dsl_tiler_dimensions_get)
* [dsl_tiler_dimensions_set](/docs/api-display.md#dsl_tiler_dimensions_set)
* [dsl_tiler_tiles_get](/docs/api-display.md#dsl_display_tiles_get)
* [dsl_tiler_tiles_set](/docs/api-display.md#dsl_display_tiles_set)
* [dsl_tiler_batch_meta_handler_add](/docs/api-display.md#dsl_tiler_batch_meta_handler_add).
* [dsl_tiler_batch_meta_handler_remove](/docs/api-display.md#dsl_tiler_batch_meta_handler_remove).

### On-Screen Display (OSD);
* dsl_osd_new
* dsl_osd_clock_enable
* dsl_osd_clock_disable
* dsl_osd_clock_state_is
* Other TBD

### Sink:
* [dsl_sink_overlay_offsets_get](/docs/api-sink.md#dsl_sink_overlay_offsets_get)
* [dsl_sink_overlay_offsets_set](/docs/api-sink.md#dsl_sink_overlay_offsets_set)
* [dsl_sink_overlay_dimensions_get](/docs/api-sink.md#dsl_sink_overlay_dimensions_get)
* [dsl_sink_overlay_dimensions_set](/docs/api-sink.md#dsl_sink_overlay_dimensions_set)
* [dsl_sink_window_offsets_get](/docs/api-sink.md#dsl_sink_window_offsets_get)
* [dsl_sink_window_offsets_set](/docs/api-sink.md#dsl_sink_window_offsets_set)
* [dsl_sink_window_dimensions_get](/docs/api-sink.md#dsl_sink_window_dimensions_get)
* [dsl_sink_window_dimensions_set](/docs/api-sink.md#dsl_sink_window_dimensions_set)
* [dsl_sink_file_video_formats_get](/docs/api-sink.md#dsl_sink_file_video_formats_get)
* [dsl_sink_file_video_formats_set](/docs/api-sink.md#dsl_sink_file_video_formats_set)
* [dsl_sink_file_code_settings_get](/docs/api-sink.md#dsl_sink_file_code_settings_get)
* [dsl_sink_file_code_settings_set](/docs/api-sink.md#dsl_sink_file_code_settings_set)
* [dsl_sink_rtsp_server_settings_get](/docs/api-sink.md#dsl_sink_rtsp_server_settings_get)
* [dsl_sink_rtsp_server_settings_set](/docs/api-sink.md#dsl_sink_rtsp_server_settings_set)
* [dsl_sink_rtsp_code_settings_get](/docs/api-sink.md#dsl_sink_rtsp_code_settings_get)
* [dsl_sink_rtsp_code_settings_set](/docs/api-sink.md#dsl_sink_rtsp_code_settings_set)
* [dsl_sink_num_in_use_get](/docs/api-sink.md#dsl_sink_num_in_use_get)
* [dsl_sink_num_in_use_max_get](/docs/api-sink.md#dsl_sink_num_in_use_max_get)
* [dsl_sink_num_in_use_max_set](/docs/api-sink.md#dsl_sink_num_in_use_max_set)
* other TBD

### Dewarpper:
* Other TBD
