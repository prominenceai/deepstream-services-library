#!/usr/bin/env python

import sys
import time

from dsl import *

# Filespecs for the Primary GIE
# primary_infer_config_file = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt"
# primary_model_engine_file = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine"
primary_infer_config_file = (
    "/home/hercadmin/repos/ACE-nv-inference/model/config_infer_autom.txt"
)
primary_model_engine_file = (
    "/home/hercadmin/repos/ACE-nv-inference/model/autom_b4_gpu0_fp32.engine"
)

# Filespec for the IOU Tracker config file
iou_tracker_config_file = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml"


sample_uri = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"

# converter_config_file = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.txt"
converter_config_file = (
    "/home/hercadmin/repos/ACE-nv-inference/utils/msgconv_config.txt"
)
protocol_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_redis_proto.so"
broker_config_file = "/home/hercadmin/repos/ACE-nv-inference/utils/cfg_redis.txt"

connection_string = "localhost;6379"


# Function to be called on End-of-Stream (EOS) event
def eos_event_listener(client_data):
    print("Pipeline EOS event")
    dsl_pipeline_stop("pipeline")
    dsl_main_loop_quit()


class ReportData:
    def __init__(self, header_interval):
        self.m_report_count = 0
        self.m_header_interval = header_interval


##
# Meter Sink client callback funtion
##
def meter_sink_handler(session_avgs, interval_avgs, source_count, client_data):
    # cast the C void* client_data back to a py_object pointer and deref
    report_data = cast(client_data, POINTER(py_object)).contents.value

    # Print header on interval
    if report_data.m_report_count % report_data.m_header_interval == 0:
        header = ""
        for source in range(source_count):
            subheader = f"FPS {source} (AVG)"
            header += "{:<15}".format(subheader)
        print()
        print(header)

    # Print FPS counters
    counters = ""
    for source in range(source_count):
        counter = "{:.2f} ({:.2f})".format(interval_avgs[source], session_avgs[source])
        counters += "{:<15}".format(counter)
    print(counters)

    # Increment reporting count
    report_data.m_report_count += 1

    return True


def create_and_add_branch(i):
    osd_name = "osd-" + str(i)

    retval = dsl_osd_new(osd_name, True, True, True, False)
    if retval != DSL_RETURN_SUCCESS:
        return retval

    sink_name = sink_name = "sink-" + str(i)

    # Calculate the offset and dimensions for the window
    offset_x = (i % 2) * 640
    offset_y = (i // 2) * 360
    width = 640
    height = 360

    # Create a new window sink with the calculated parameters
    retval = dsl_sink_window_new(sink_name, offset_x, offset_y, width, height)
    if retval != DSL_RETURN_SUCCESS:
        return retval

    branch_name = "branch-" + str(i)
    retval = dsl_branch_new_component_add_many(branch_name, [osd_name, sink_name, None])
    if retval != DSL_RETURN_SUCCESS:
        return retval

    retval = dsl_tee_branch_add("demuxer", branch_name)
    return retval


def main(args):
    # Since we're not using args, we can Let DSL initialize GST on first call
    while True:
        # retval = dsl_source_file_new("uri-source-1", sample_uri, True)
        # if retval != DSL_RETURN_SUCCESS:
        #     break
        # retval = dsl_source_file_new("uri-source-2", sample_uri, True)
        # retval = dsl_source_file_new("uri-source-3", sample_uri, True)
        # retval = dsl_source_file_new("uri-source-4", sample_uri, True)

        retval = dsl_source_usb_new("uri-source-1", 1280, 720, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_usb_new("uri-source-2", 1280, 720, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_usb_device_location_set("uri-source-2", "/dev/video2")
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_usb_new("uri-source-3", 1280, 720, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_usb_device_location_set("uri-source-3", "/dev/video4")
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_usb_new("uri-source-4", 1280, 720, 0, 0)
        if retval != DSL_RETURN_SUCCESS:
            break
        retval = dsl_source_usb_device_location_set("uri-source-4", "/dev/video6")
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Primary GIE using the filespecs above, with interval and Id
        retval = dsl_infer_gie_primary_new(
            "primary-gie", primary_infer_config_file, primary_model_engine_file, 1
        )
        if retval != DSL_RETURN_SUCCESS:
            break

        # New IOU Tracker, setting operational width and hieght
        retval = dsl_tracker_new("iou-tracker", iou_tracker_config_file, 480, 272)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Message broker / Metadata
        retval = dsl_ode_action_message_meta_add_new("add-message-meta")
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_trigger_instance_new(
            "instance-trigger", DSL_ODE_ANY_SOURCE, DSL_ODE_ANY_CLASS, 0
        )
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_ode_trigger_action_add("instance-trigger", "add-message-meta")
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pph_ode_new("msg-ode-handler")
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_pph_ode_trigger_add("msg-ode-handler", "instance-trigger")
        if retval != DSL_RETURN_SUCCESS:
            break
        # -------

        retval = dsl_tracker_pph_add("iou-tracker", "msg-ode-handler", DSL_PAD_SRC)
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_tee_demuxer_new("demuxer", 4)
        if retval != DSL_RETURN_SUCCESS:
            break

        for i in range(4):
            retval = create_and_add_branch(i)
            if retval != DSL_RETURN_SUCCESS:
                break

        retval = dsl_tee_splitter_new("splitter")
        if retval != DSL_RETURN_SUCCESS:
            break

        report_data = ReportData(header_interval=12)
        retval = dsl_pph_meter_new(
            "pph-meter", 5, meter_sink_handler, client_data=report_data
        )
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_tee_pph_add("splitter", "pph-meter")
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_sink_message_new(
            "message-sink",
            converter_config_file=converter_config_file,
            payload_type=DSL_MSG_PAYLOAD_DEEPSTREAM_MINIMAL,
            broker_config_file=broker_config_file,
            protocol_lib=protocol_lib,
            connection_string=connection_string,
            topic="objects",
        )
        if retval != DSL_RETURN_SUCCESS:
            break

        retval = dsl_tee_branch_add_many("splitter", ["demuxer", "message-sink", None])
        if retval != DSL_RETURN_SUCCESS:
            break

        # add components to the pipeline
        retval = dsl_pipeline_new_component_add_many(
            "pipeline",
            [
                "uri-source-1",
                "uri-source-2",
                "uri-source-3",
                "uri-source-4",
                "primary-gie",
                "iou-tracker",
                "splitter",
            ],
        )
        if retval != DSL_RETURN_SUCCESS:
            break

        # New Pipeline to use with the above components
        retval = dsl_pipeline_eos_listener_add("pipeline", eos_event_listener, None)
        if retval != DSL_RETURN_SUCCESS:
            break

        # Play the pipeline
        retval = dsl_pipeline_play("pipeline")
        if retval != DSL_RETURN_SUCCESS:
            break

        dsl_main_loop_run()
        retval = DSL_RETURN_SUCCESS
        break

    # Print out the final result
    print(dsl_return_value_to_string(retval))

    dsl_delete_all()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
