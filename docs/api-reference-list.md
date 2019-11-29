# DSL API Reference

#### Pipeline API:
* [dsl_pipeline_new](/docs/api-pipeline.md#dsl_pipeline_new)
* [dsl_pipeline_new_many](/docs/api-pipeline.md#dsl_pipeline_new_many)
* [dsl_pipeline_delete](/docs/api-pipeline.md#dsl_pipeline_delete)
* [dsl_pipeline_delete_many](/docs/api-pipeline.md#dsl_pipeline_delete_many)
* [dsl_pipeline_delete_all](/docs/api-pipeline.md#dsl_pipeline_delete_all)
* [dsl_pipeline_list_size](/docs/api-pipeline.md#dsl_pipeline_list_size)
* [dsl_pipeline_list_all](/docs/api-pipeline.md#dsl_pipeline_list_all)
* [dsl_pipeline_component_add](/docs/api-pipeline.md#dsl_pipeline_component_add)
* [dsl_pipeline_component_add_many](/docs/api-pipeline.md#dsl_pipeline_component_add_many)
* [dsl_pipeline_component_list_size](/docs/api-pipeline.md#dsl_pipeline_components_list_size)
* [dsl_pipeline_component_list_all](/docs/api-pipeline.md#dsl_pipeline_components_list_all)
* [dsl_pipeline_component_remove](/docs/api-pipeline.md#dsl_pipeline_component_remove)
* [dsl_pipeline_component_remove_many](/docs/api-pipeline.md#dsl_pipeline_component_remove_many)
* [dsl_pipeline_component_remove_all](/docs/api-pipeline.md#dsl_pipeline_component_remove_all)
* [dsl_pipeline_component_replace](/docs/api-pipeline.md#dsl_pipeline_component_replace)
* [dsl_pipeline_streammux_properties_get](/docs/api-pipeline.md#dsl_pipeline_streammux_properties_get)
* [dsl_pipeline_streammux_properties_set](/docs/api-pipeline.md#dsl_pipeline_streammux_properties_set)
* [dsl_pipeline_play](/docs/api-pipeline.md#dsl_pipeline_play)
* [dsl_pipeline_pause](/docs/api-pipeline.md#dsl_pipeline_pause)
* [dsl_pipeline_stop](/docs/api-pipeline.md#dsl_pipeline_stop)
* [dsl_pipeline_state_get](/docs/api-pipeline.md#dsl_pipeline_state_get)
* [dsl_pipeline_state_change_listener_add](/docs/api-pipeline.md#dsl_pipeline_state_change_listener_add)
* [dsl_pipeline_state_change_listener_remove](/docs/api-pipeline.md#dsl_pipeline_state_change_listener_remove)
* [dsl_pipeline_display_window_handle_get](/docs/api-pipeline.md#dsl_pipeline_display_window_handle_get)
* [dsl_pipeline_display_window_handle_set](/docs/api-pipeline.md#dsl_pipeline_display_window_handle_set)
* [dsl_pipeline_display_event_handler_add](/docs/api-pipeline.md#dsl_pipeline_display_event_handler_add)
* [dsl_pipeline_display_event_handler_remove](/docs/api-pipeline.md#dsl_pipeline_display_event_handler_remove)
* [dsl_pipeline_dump_to_dot](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot)
* [dsl_pipeline_dump_to_dot_with_ts](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot_with_ts)
* Other TBD


#### Component API:
* dsl_component_copy
* [dsl_component_delete](/docs/api-component.md#dsl_component_delete)
* [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many)
* [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)
* [dsl_component_list_size](/docs/api-component.md#dsl_component_list_size)
* [dsl_component_list_all](/docs/api-component.md#dsl_component_list_all)
* Other TBD


#### Source API:
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

#### Primary and Secondary GIE API:
* [dsl_gie_primary_new](#dsl_gie_primary_new)
* [dsl_gie_secondary_new](#dsl_gie_secondary_new)
* [dsl_gie_infer_config_file_get](#dsl_gie_infer_config_file_get)
* [dsl_gie_infer_config_file_set](#dsl_gie_infer_config_file_set)
* [dsl_gie_model_engine_file_get](#dsl_gie_model_engine_file_get)
* [dsl_gie_model_engine_file_set](#dsl_gie_model_engine_file_set)
* [dsl_gie_interval_get](#dsl_gie_interval_get)
* [dsl_gie_interval_set](#dsl_gie_interval_set)
* [dsl_gie_secondary_infer_on_get](#dsl_gie_secondary_infer_on_get)
* [dsl_gie_secondary_infer_on_set](#dsl_gie_secondary_infer_on_set)
* Other TBD

#### Tiled Display:
* dsl_display_new
* dsl_display_attributes_get
* dsl_display_attributes_set
* Other TBD

#### On-Screen Display (OSD);
* dsl_osd_new
* dsl_osd_clock_enable
* dsl_osd_clock_disable
* dsl_osd_clock_state_is
* dsl_osd_clock_attributes_get
* dsl_osd_clock_attributes_set
* Other TBD

#### Sink:
* dsl_sink_overlay_new
* dsl_sink_overlay_attributes_get
* dsl_sink_overlay_attributes_set
* dsl_sink_rtsp_new
* dsl_sink_rtsp_attributes_get
* dsl_sink_rtsp_attributes_set
* dsl_sink_fake_new
* dsl_sink_fake_attributes_get
* dsl_sink_fake_attributes_set
* dsl_sink_egl_new
* dsl_sink_egl_attributes_get
* dsl_sink_egl_attributes_set
* dsl_sink_get_num_in_use
* dsl_sink_get_num_in_use_max
* dsl_sink_set_num_in_use_max
* other TBD

#### Tracker:
* Other TBD

#### Dewarpper:
* Other TBD
