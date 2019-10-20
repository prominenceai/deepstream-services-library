# DSL API Reference

#### Pipeline API:
* [dsl_pipeline_new](/docs/api-pipeline.md#dsl_pipeline_new)
* [dsl_pipeline_delete](/docs/api-pipeline.md#dsl_pipeline_delete)
* [dsl_pipeline_delete_many](/docs/api-pipeline.md#dsl_pipeline_delete_many)
* [dsl_pipeline_delete_all](/docs/api-pipeline.md#dsl_pipeline_list_size)
* [dsl_pipeline_list_all](/docs/api-pipeline.md#dsl_pipeline_list_all)
* dsl_pipeline_play
* dsl_pipeline_pause
* dsl_pipeline_state_is
* [dsl_pipeline_dump_to_dot](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot)
* [dsl_pipeline_dump_to_dot_with_ts](/docs/api-pipeline.md#dsl_pipeline_dump_to_dot_with_ts)
* dsl_pipeline_component_add
* dsl_pipeline_component_add_many
* dsl_pipeline_component_remove
* dsl_pipeline_component_remove_many
* dsl_pipeline_component_remove_all
* dsl_pipeline_component_list_all
* dsl_pipeline_message_subscriber_add
* dsl_pipeline_message_subscriber_remove
* [dsl_pipeline_state_change_listener_add](/docs/api-pipeline.md#dsl_pipeline_state_change_listener_add)
* [dsl_pipeline_state_change_listener_remove](/docs/api-pipeline.md#dsl_pipeline_state_change_listener_remove)
* dsl_pipeline_display_event_handler_add
* dsl_pipeline_display_event_handler_remove

#### Component API:
* dsl_component_copy
* [dsl_component_delete](/docs/api-component.md#dsl_component_delete)
* [dsl_component_delete_many](/docs/api-component.md#dsl_component_delete_many)
* [dsl_component_delete_all](/docs/api-component.md#dsl_component_delete_all)
* [dsl_component_list_size](/docs/api-component.md#dsl_component_list_size)
* [dsl_component_list_all](/docs/api-component.md#dsl_component_list_all)

#### Source API:
* [dsl_source_csi_new](/docs/api-source.md#dsl_source_csi_new)
* [dsl_source_v4l2_new](/docs/api-source.md#dsl_source_v4l2_new)
* [dsl_source_uri_new](/docs/api-source.md#dsl_source_uri_new)
* [dsl_source_rtmp_new](/docs/api-source.md#dsl_source_rtmp_new)
* [dsl_source_pause](/docs/api-source.md#dsl_source_pause)
* [dsl_source_play](/docs/api-source.md#dsl_source_play)
* [dsl_source_state_is](/docs/api-source.md#dsl_source_state_is)

#### Infer Engine (GIE):
* dsl_gie_primary_new
* dsl_gie_secondary_new

#### Tiled Display:
* dsl_display_new
* dsl_display_resize

#### On-Screen Display (OSD);
* dsl_osd_new
* dsl_osd_clock_enable
* dsl_osd_clock_disable
* dsl_osd_clock_state_is
* dsl_osd_clock_attributes_set
* dsl_osd_clock_attributes_get

#### Sink:
* dsl_sink_overlay_new
* dsl_sink_rtmp_new
* dsl_sink_fake_new

