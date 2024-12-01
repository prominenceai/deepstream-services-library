/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in-
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "catch.hpp"
#include "Dsl.h"
#include "DslApi.h"

static const std::wstring uri(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

static const std::wstring uri_source_name(L"uri-source");
static const std::wstring v4l2_source_name(L"v4l2-source");
static const std::wstring preproc_name(L"preprocessor");
static const std::wstring primary_gie_name(L"primary-gie");
static const std::wstring secondary_gie_name(L"secondary-gie");
static const std::wstring tiler_name(L"tiler");
static const std::wstring osd_name(L"osd");
static const std::wstring window_sink_name(L"egl-sink");
static const std::wstring file_sink_name(L"file-sink");
static const std::wstring fake_sink_name(L"fake-sink");


static const std::wstring filePath(L"./output.mp4");
static const std::wstring def_device_location(L"/dev/video0");
     
static const std::wstring record_tap_name(L"record-tap");
static const std::wstring outdir(L"./");
static uint container(DSL_CONTAINER_MP4);

static dsl_record_client_listener_cb client_listener;
static const std::wstring preproc_config(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_preprocess.txt");

static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");

static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

static uint interval(1);

static std::wstring secondary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
static std::wstring secondary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet.etlt_b8_gpu0_int8.engine");

static const std::wstring tracker_name(L"iou-tracker");
static const std::wstring tracker_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");

static const std::wstring dewarperName(L"dewarper");

static const std::wstring dewarper_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper.txt");

SCENARIO( "The Components container is updated correctly on multiple new components", "[component-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "Several new components are created" ) 
        {

            REQUIRE( dsl_source_uri_new(uri_source_name.c_str(), uri.c_str(), 
                false, 0, 0) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
                0, 0, 1280, 720) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
                1280, 720) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                // TODO complete verification after addition of Iterator API
                REQUIRE( dsl_component_list_size() == 3 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "A new component can Set and Get Queue Properties correctly", 
    "[component-api]" )
{
    GIVEN( "Three new components" ) 
    {
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        uint64_t ret_current_level(99);
        REQUIRE( dsl_component_queue_current_level_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_current_level) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_current_level == 0 );

        ret_current_level = 99;
        REQUIRE( dsl_component_queue_current_level_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_current_level) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_current_level == 0 );

        ret_current_level = 99;
        REQUIRE( dsl_component_queue_current_level_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_current_level) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_current_level == 0 );

        uint ret_leaky(99);
        REQUIRE( dsl_component_queue_leaky_get(tracker_name.c_str(), &ret_leaky) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_NO );

        uint64_t ret_max_size(99);
        REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_max_size == 200 );

        ret_max_size = 99;
        REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_max_size == 10485760 );

        ret_max_size = 99;
        REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_max_size == 1000000000 );

        uint64_t ret_min_threshold(99);
        REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_min_threshold == 0 );

        ret_min_threshold = 99;
        REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_min_threshold == 0 );

        ret_min_threshold = 99;
        REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
            DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
            == DSL_RESULT_SUCCESS );
        REQUIRE( ret_min_threshold == 0 );

        WHEN( "The new component is called to set their queue properies" ) 
        {
            uint new_leaky(DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM);
            uint64_t new_max_size_buffers(123);
            uint64_t new_max_size_bytes(4321);
            uint64_t new_max_size_time(987654);
            uint64_t new_min_threshold_buffers(12);
            uint64_t new_min_threshold_bytes(432);
            uint64_t new_min_threshold_time(9876);
            
            REQUIRE( dsl_component_queue_leaky_set(tracker_name.c_str(), 
                new_leaky) == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_max_size_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, new_max_size_buffers) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_max_size_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, new_max_size_bytes) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_max_size_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME, new_max_size_time) 
                == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_min_threshold_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, new_min_threshold_buffers) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_min_threshold_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, new_min_threshold_bytes) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_min_threshold_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME, new_min_threshold_time) 
                == DSL_RESULT_SUCCESS );

            THEN( "The correct values are returned on get" ) 
            {
                REQUIRE( dsl_component_queue_leaky_get(tracker_name.c_str(), &ret_leaky) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "Invalid parameters are used to set queue properies" ) 
        {
            uint new_leaky(DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM);
            uint64_t new_max_size_buffers(123);
            uint64_t new_max_size_bytes(4321);
            uint64_t new_max_size_time(987654);
            uint64_t new_min_threshold_buffers(12);
            uint64_t new_min_threshold_bytes(432);
            uint64_t new_min_threshold_time(9876);
            
            REQUIRE( dsl_component_queue_leaky_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_LEAKY_DOWNSTREAM+1) 
                == DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED );

            REQUIRE( dsl_component_queue_max_size_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME+1, new_max_size_time) 
                == DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED );

            REQUIRE( dsl_component_queue_min_threshold_set(tracker_name.c_str(), 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME+1, new_min_threshold_time) 
                == DSL_RESULT_COMPONENT_SET_QUEUE_PROPERTY_FAILED );

            THEN( "The default values are unchanged" ) 
            {
                REQUIRE( dsl_component_queue_leaky_get(tracker_name.c_str(), &ret_leaky) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_NO );

                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == 200 );

                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == 10485760 );
                
                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == 1000000000 );

                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == 0 );

                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == 0 );
                
                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "Multiple new components can Set and Get Queue Properties correctly", 
    "[component-api]" )
{
    GIVEN( "Three new components" ) 
    {
        uint tacker_width(480);
        uint tacker_height(272);

        uint tiler_width(1280);
        uint tiler_height(720);

        REQUIRE( dsl_source_uri_new(uri_source_name.c_str(), uri.c_str(), 
            false, 0, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_source_v4l2_new(v4l2_source_name.c_str(), 
            def_device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name.c_str(), 
            secondary_infer_config_file.c_str(), secondary_model_engine_file.c_str(), 
            primary_gie_name.c_str(), interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(), 
            tacker_width, tacker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            true, true, true, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            0, 0, 1280, 720) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {
            uri_source_name.c_str(), v4l2_source_name.c_str(), 
            record_tap_name.c_str(), preproc_name.c_str(), 
            primary_gie_name.c_str(), secondary_gie_name.c_str(),
            tracker_name.c_str(), tiler_name.c_str(), osd_name.c_str(), 
            window_sink_name.c_str(), fake_sink_name.c_str(),
            NULL};
        
        WHEN( "The new component is called to set their queue properies" ) 
        {
            uint new_leaky(DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM);
            uint64_t new_max_size_buffers(123);
            uint64_t new_max_size_bytes(4321);
            uint64_t new_max_size_time(987654);
            uint64_t new_min_threshold_buffers(12);
            uint64_t new_min_threshold_bytes(432);
            uint64_t new_min_threshold_time(9876);
            
            REQUIRE( dsl_component_queue_leaky_set_many(components, 
                new_leaky) == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_max_size_set_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, new_max_size_buffers) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_max_size_set_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, new_max_size_bytes) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_max_size_set_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME, new_max_size_time) 
                == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_min_threshold_set_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, new_min_threshold_buffers) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_min_threshold_set_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, new_min_threshold_bytes) 
                == DSL_RESULT_SUCCESS );
 
            REQUIRE( dsl_component_queue_min_threshold_set_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME, new_min_threshold_time) 
                == DSL_RESULT_SUCCESS );

            THEN( "The correct values are returned on get" ) 
            {
                uint64_t ret_current_level(99);
                uint ret_leaky(99);
                uint64_t ret_max_size(99);
                uint64_t ret_min_threshold(99);
                
                REQUIRE( dsl_component_queue_leaky_get(uri_source_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(v4l2_source_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(record_tap_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(preproc_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(primary_gie_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(secondary_gie_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(tracker_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(tiler_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(osd_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(window_sink_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                REQUIRE( dsl_component_queue_leaky_get(fake_sink_name.c_str(), 
                    &ret_leaky) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_leaky == DSL_COMPONENT_QUEUE_LEAKY_UPSTREAM );

                // ---------------------------------

                REQUIRE( dsl_component_queue_max_size_get(uri_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(v4l2_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(record_tap_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(preproc_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(primary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(secondary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(tiler_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(osd_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(window_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                REQUIRE( dsl_component_queue_max_size_get(fake_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_buffers );

                // ---------------------------------

                REQUIRE( dsl_component_queue_max_size_get(uri_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(v4l2_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(record_tap_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(preproc_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(primary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(secondary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(tiler_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(osd_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(window_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(fake_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                // ---------------------------------

                REQUIRE( dsl_component_queue_max_size_get(uri_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(v4l2_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(record_tap_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(preproc_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(primary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(secondary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(tiler_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(osd_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_time );

                REQUIRE( dsl_component_queue_max_size_get(window_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                REQUIRE( dsl_component_queue_max_size_get(fake_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_max_size) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_size == new_max_size_bytes );
                
                // ---------------------------------

                REQUIRE( dsl_component_queue_min_threshold_get(uri_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(v4l2_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(record_tap_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(preproc_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(primary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(secondary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(tiler_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(osd_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(window_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                REQUIRE( dsl_component_queue_min_threshold_get(fake_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_buffers );

                // ---------------------------------

                REQUIRE( dsl_component_queue_min_threshold_get(uri_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(v4l2_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(record_tap_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(preproc_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(primary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(secondary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(tiler_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(osd_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(window_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                REQUIRE( dsl_component_queue_min_threshold_get(fake_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_BYTES, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_bytes );
                
                // ---------------------------------

                REQUIRE( dsl_component_queue_min_threshold_get(uri_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(v4l2_source_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(record_tap_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(preproc_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(primary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(secondary_gie_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(tracker_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(tiler_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(osd_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(window_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_queue_min_threshold_get(fake_sink_name.c_str(), 
                    DSL_COMPONENT_QUEUE_UNIT_OF_TIME, &ret_min_threshold) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( ret_min_threshold == new_min_threshold_time );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Multiple new components can Print and Log their Queue levels/max-sizes correctly", 
    "[component-api]" )
{
    GIVEN( "Three new components" ) 
    {
        uint tacker_width(480);
        uint tacker_height(272);

        uint tiler_width(1280);
        uint tiler_height(720);

        REQUIRE( dsl_source_uri_new(uri_source_name.c_str(), uri.c_str(), 
            false, 0, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_source_v4l2_new(v4l2_source_name.c_str(), 
            def_device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name.c_str(), 
            secondary_infer_config_file.c_str(), secondary_model_engine_file.c_str(), 
            primary_gie_name.c_str(), interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(), 
            tacker_width, tacker_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            tiler_width, tiler_height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            true, true, true, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            0, 0, 1280, 720) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {
            uri_source_name.c_str(), v4l2_source_name.c_str(), 
            record_tap_name.c_str(), preproc_name.c_str(), 
            primary_gie_name.c_str(), secondary_gie_name.c_str(),
            tracker_name.c_str(), tiler_name.c_str(), osd_name.c_str(), 
            window_sink_name.c_str(), fake_sink_name.c_str(),
            NULL};
        
        WHEN( "The new components are called to print and log their queue levels" ) 
        {
            REQUIRE( dsl_component_queue_current_level_print_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_current_level_print_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_current_level_print_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_current_level_log_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_current_level_log_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_queue_current_level_log_many(components, 
                DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == DSL_RESULT_SUCCESS );

            THEN( "The values are printed out correctly" ) 
            {
                // Note: requires manual verification of the print statements.
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}


static void queue_overrun_listener_cb(const wchar_t* name, void* client_data)
{

}
static void queue_underrun_listener_cb(const wchar_t* name, void* client_data)
{
    
}

SCENARIO( "A component can add and remove queue client listeners", "[component-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A Component in memory" ) 
    {
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        WHEN( "A state-change-listener is added" )
        {
            REQUIRE( dsl_component_queue_overrun_listener_add(tracker_name.c_str(),
                queue_overrun_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // Second add of the same listener must fail
            REQUIRE( dsl_component_queue_overrun_listener_add(tracker_name.c_str(),
                queue_overrun_listener_cb, (void*)0x12345678) 
                == DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED );

            REQUIRE( dsl_component_queue_underrun_listener_add(tracker_name.c_str(),
                queue_underrun_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // Second add of the same listener must fail
            REQUIRE( dsl_component_queue_underrun_listener_add(tracker_name.c_str(),
                queue_underrun_listener_cb, (void*)0x12345678) 
                == DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED );

            THEN( "The same listener can be removed" ) 
            {
                REQUIRE( dsl_component_queue_overrun_listener_remove(
                    tracker_name.c_str(), queue_overrun_listener_cb) 
                    == DSL_RESULT_SUCCESS );

                // Second remove of the same listener must fail
                REQUIRE( dsl_component_queue_overrun_listener_remove(
                    tracker_name.c_str(), queue_overrun_listener_cb) 
                    == DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_component_queue_underrun_listener_remove(
                    tracker_name.c_str(), queue_underrun_listener_cb) 
                    == DSL_RESULT_SUCCESS );

                // Second remove of the same listener must fail
                REQUIRE( dsl_component_queue_underrun_listener_remove(
                    tracker_name.c_str(), queue_underrun_listener_cb) 
                    == DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "Multiple components can add and remove a queue client listener",
    "[component-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A Component in memory" ) 
    {
        uint width(480);
        uint height(272);

        REQUIRE( dsl_source_uri_new(uri_source_name.c_str(), uri.c_str(), 
            false, 0, 0) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_source_v4l2_new(v4l2_source_name.c_str(), 
            def_device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {
            uri_source_name.c_str(), v4l2_source_name.c_str(),
            tracker_name.c_str(), NULL};
        
        WHEN( "A state-change-listener is added to multiple clients" )
        {
            REQUIRE( dsl_component_queue_overrun_listener_add_many(components,
                queue_overrun_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // Second add of the same listeners must fail
            REQUIRE( dsl_component_queue_overrun_listener_add_many(components,
                queue_overrun_listener_cb, (void*)0x12345678) 
                == DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED );

            REQUIRE( dsl_component_queue_underrun_listener_add_many(components,
                queue_underrun_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // Second add of the same listener must fail
            REQUIRE( dsl_component_queue_underrun_listener_add_many(components,
                queue_underrun_listener_cb, (void*)0x12345678) 
                == DSL_RESULT_COMPONENT_CALLBACK_ADD_FAILED );

            THEN( "The same listener can be removed" ) 
            {
                REQUIRE( dsl_component_queue_overrun_listener_remove_many(
                    components, queue_overrun_listener_cb) 
                    == DSL_RESULT_SUCCESS );

                // Second remove of the same listener must fail
                REQUIRE( dsl_component_queue_overrun_listener_remove_many(
                    components, queue_overrun_listener_cb) 
                    == DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_component_queue_underrun_listener_remove_many(
                    components, queue_underrun_listener_cb) 
                    == DSL_RESULT_SUCCESS );

                // Second remove of the same listener must fail
                REQUIRE( dsl_component_queue_underrun_listener_remove_many(
                    components, queue_underrun_listener_cb) 
                    == DSL_RESULT_COMPONENT_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "Multiple new components can Set and Get their GPU ID", "[component-api]" )
{
    GIVEN( "Three new components" ) 
    {
        uint GPUID0(0);
        uint GPUID1(1);
        uint retGpuId(0);

        uint width(480);
        uint height(272);

        uint codec(DSL_ENCODER_HW_H265);
        uint container(DSL_CONTAINER_MP4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_source_uri_new(uri_source_name.c_str(), uri.c_str(), 
            false, 0, 0) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_dewarper_new(dewarperName.c_str(), 
            dewarper_config_file.c_str(), 1) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), tracker_config_file.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            true, true, true, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            0, 0, 1280, 720) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            1280, 720) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_file_new(file_sink_name.c_str(), filePath.c_str(),
            codec, container, bitrate, interval) == DSL_RESULT_SUCCESS );

        
        retGpuId = 99;
        REQUIRE( dsl_component_gpuid_get(primary_gie_name.c_str(), 
            &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == 0);
        retGpuId = 99;
        REQUIRE( dsl_component_gpuid_get(tiler_name.c_str(), 
            &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == 0);
        retGpuId = 99;
        REQUIRE( dsl_component_gpuid_get(osd_name.c_str(), 
            &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == 0);

        REQUIRE( dsl_component_gpuid_get(window_sink_name.c_str(), 
            &retGpuId) == DSL_RESULT_SUCCESS );
        REQUIRE( retGpuId == 0);

        WHEN( "Several new components are called to Set their GPU ID" ) 
        {
            uint newGpuId(1);
            
            if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
            {
                const wchar_t* components[] = {uri_source_name.c_str(), 
                    dewarperName.c_str(), primary_gie_name.c_str(), 
                    tracker_name.c_str(), tiler_name.c_str(), 
                    osd_name.c_str(), file_sink_name.c_str(), NULL};
                REQUIRE( dsl_component_gpuid_set_many(components, newGpuId) 
                    == DSL_RESULT_SUCCESS );
            }
            else
            {
                const wchar_t* components[] = {uri_source_name.c_str(), 
                    dewarperName.c_str(), primary_gie_name.c_str(), 
                    tracker_name.c_str(), tiler_name.c_str(), 
                    osd_name.c_str(), window_sink_name.c_str(), 
                    file_sink_name.c_str(), NULL};
                REQUIRE( dsl_component_gpuid_set_many(components, newGpuId) 
                    == DSL_RESULT_SUCCESS );
            }

            THEN( "All components return the correct GPU ID of get" ) 
            {
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(uri_source_name.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(dewarperName.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(primary_gie_name.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(tracker_name.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(tiler_name.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(osd_name.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;
                REQUIRE( dsl_component_gpuid_get(file_sink_name.c_str(), 
                    &retGpuId) == DSL_RESULT_SUCCESS );
                REQUIRE( retGpuId == newGpuId);
                retGpuId = 99;

                if (dsl_info_gpu_type_get(0) != DSL_GPU_TYPE_INTEGRATED)
                {
                    REQUIRE( dsl_component_gpuid_get(window_sink_name.c_str(), 
                        &retGpuId) == DSL_RESULT_SUCCESS );
                    REQUIRE( retGpuId == newGpuId);
                }

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "Multiple new components can Set and Get their NVIDIA mem type", "[component-api]" )
{
    GIVEN( "Three new components" ) 
    {
        uint GPUID0(0);
        uint GPUID1(1);
        uint retGpuId(0);

        uint retNvbufMem(99);

        REQUIRE( dsl_source_uri_new(uri_source_name.c_str(), uri.c_str(), 
            false, 0, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_osd_new(osd_name.c_str(), 
            true, true, true, false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name.c_str(), 
            1280, 720) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_component_nvbuf_mem_type_get(uri_source_name.c_str(), 
              &retNvbufMem) == DSL_RESULT_SUCCESS );
        REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
        retNvbufMem = 99;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            REQUIRE( dsl_component_nvbuf_mem_type_get(tiler_name.c_str(), 
                &retNvbufMem) == DSL_RESULT_SUCCESS );
            REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
            retNvbufMem = 99;
            REQUIRE( dsl_component_nvbuf_mem_type_get(osd_name.c_str(), 
                &retNvbufMem) == DSL_RESULT_SUCCESS );
            REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
        }
        else
        {
            REQUIRE( dsl_component_nvbuf_mem_type_get(tiler_name.c_str(), 
                &retNvbufMem) == DSL_RESULT_SUCCESS );
            REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
            retNvbufMem = 99;
            REQUIRE( dsl_component_nvbuf_mem_type_get(osd_name.c_str(), 
                &retNvbufMem) == DSL_RESULT_SUCCESS );
            REQUIRE( retNvbufMem == DSL_NVBUF_MEM_TYPE_DEFAULT);
        }


        WHEN( "Several new components are called to Set their NVIDIA mem type" ) 
        {
            uint newNvbufMemType;
            if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
            {
                newNvbufMemType = DSL_NVBUF_MEM_TYPE_SURFACE_ARRAY;
            }
            {
                newNvbufMemType = DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED;
            }

            const wchar_t* components[] = {
                uri_source_name.c_str(), tiler_name.c_str(), 
                osd_name.c_str(), NULL};
            REQUIRE( dsl_component_nvbuf_mem_type_set_many(components, 
                newNvbufMemType) == DSL_RESULT_SUCCESS );

            THEN( "All components return the correct NVIDIA mem type on get" ) 
            {
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(uri_source_name.c_str(), 
                    &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(tiler_name.c_str(), 
                    &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
                retNvbufMem = 99;
                REQUIRE( dsl_component_nvbuf_mem_type_get(osd_name.c_str(), 
                    &retNvbufMem) == DSL_RESULT_SUCCESS );
                REQUIRE( retNvbufMem == newNvbufMemType );
//                retNvbufMem = 99;
//                REQUIRE( dsl_component_nvbuf_mem_type_get(window_sink_name.c_str(), 
//                    &retNvbufMem) == DSL_RESULT_SUCCESS );
//                REQUIRE( retNvbufMem == newNvbufMemType );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "The Component API checks for NULL input parameters", "[component-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring component_name1(L"test-component-1");
        std::wstring component_name2(L"test-component-2");
        const wchar_t* components[] = {
            component_name1.c_str(), component_name2.c_str(), NULL};
        uint64_t current_level(0);
        uint leaky(0);
        uint64_t max_size(0);
        uint64_t min_threshold(0);
        uint gpuId(0);
        uint nvbufMemType(DSL_NVBUF_MEM_TYPE_DEFAULT);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_component_delete(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_delete_many(NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_component_media_type_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_media_type_get(component_name1.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_media_type_set(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_media_type_set_many(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_current_level_get(NULL, 
                    0, &current_level) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_current_level_get(component_name1.c_str(), 
                    0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_current_level_print(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_current_level_print_many(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_current_level_log(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_current_level_log_many(NULL, 
                    0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_leaky_get(NULL, &leaky) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_leaky_get(component_name1.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_leaky_set(NULL, leaky) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_leaky_set_many(NULL, leaky) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_max_size_get(NULL, 0, &max_size) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_max_size_get(component_name1.c_str(), 
                    0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_max_size_set(NULL, 
                    0, max_size) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_max_size_set_many(NULL, 
                    0, max_size) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_min_threshold_get(NULL, 
                    0, &min_threshold) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_min_threshold_get(component_name1.c_str(), 
                    0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_min_threshold_set(NULL, 
                    0, min_threshold) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_min_threshold_set_many(NULL, 
                    0, min_threshold) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_overrun_listener_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_overrun_listener_add(
                    component_name1.c_str(), NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_overrun_listener_add_many(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_overrun_listener_add_many(
                    components, NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_overrun_listener_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_overrun_listener_remove(
                    component_name1.c_str(), NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_overrun_listener_remove_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_overrun_listener_remove_many(
                    components, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_underrun_listener_add(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_underrun_listener_add(
                    component_name1.c_str(), NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_underrun_listener_add_many(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_underrun_listener_add_many(
                    components, NULL, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_underrun_listener_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_underrun_listener_remove(
                    component_name1.c_str(), NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_queue_underrun_listener_remove_many(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_queue_underrun_listener_remove_many(
                    components, NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_gpuid_get(NULL, 
                    &gpuId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_gpuid_set(NULL, 
                    gpuId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_gpuid_set_many(NULL, 
                    gpuId) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_nvbuf_mem_type_get(NULL, 
                    &nvbufMemType) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_nvbuf_mem_type_set(NULL, 
                    nvbufMemType) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_nvbuf_mem_type_set_many(NULL, 
                    nvbufMemType) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

