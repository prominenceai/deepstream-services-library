/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslApi.h"
#include "Dsl.h"

static std::wstring pipeline_name(L"test-pipeline");
static std::wstring pipeline_name2(L"test-pipeline2");
static std::wstring pipeline_name3(L"test-pipeline3");

static std::wstring source_name(L"source");
static std::wstring source_name2(L"source2");
static std::wstring source_name3(L"source3");

static uint width(1280);
static uint height(720);
static uint fps_n(30);
static uint fps_d(1);

static uint intra_decode(false);
static uint drop_frame_interval(0);

static std::wstring dewarper_name(L"dewarper");
static std::wstring defConfigFile(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper.txt");

static std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static std::wstring image_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");

static std::wstring jpeg_file_path_multi(L"./test/streams/sample_720p.%d.jpg");

static uint protocol(DSL_RTP_ALL);
static uint latency(100);
static uint timeout(0);
static uint retTimeout(123);

static uint interval(0);

static std::wstring rtsp_uri(L"rtsp://username:password@192.168.0.14:554");

static boolean is_live(false);

static std::wstring def_device_location(L"/dev/video0");


SCENARIO( "The Components container is updated correctly on new source", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is created" ) 
        {
            REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    
    
SCENARIO( "The Components container is updated correctly on Source Delete", "[source-api]" )
{
    GIVEN( "One Source im memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        
        WHEN( "The Source is deleted" )
        {
            REQUIRE( dsl_component_delete(source_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Source in use can't be deleted", "[source-api]" )
{
    GIVEN( "A new Source and new pPipeline" ) 
    {

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(source_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Source, once removed from a Pipeline, can be deleted", "[source-api]" )
{
    GIVEN( "A new Pipeline with a Child CSI Source" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
            source_name.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Source is removed from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(),
                source_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can be deleted successfully" ) 
            {
                REQUIRE( dsl_component_delete(source_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new CSI Camera Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is created" ) 
        {
            REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(source_name.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(source_name.c_str(), &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == width );
                REQUIRE( ret_height == height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(source_name.c_str()) == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new CIS Camera Source set/get its sensor-id correctly", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

        WHEN( "The USB Source's device-location is set" ) 
        {
            // Check default first
            uint sensor_id;
            REQUIRE( dsl_source_csi_sensor_id_get(source_name.c_str(), 
                &sensor_id) == DSL_RESULT_SUCCESS );
            REQUIRE( sensor_id == 0 );
            
            uint new_sensor_id(5);
            REQUIRE( dsl_source_csi_sensor_id_set(source_name.c_str(), 
                new_sensor_id) == DSL_RESULT_SUCCESS );

            THEN( "The correct updated value is returned on get" ) 
            {
                REQUIRE( dsl_source_csi_sensor_id_get(source_name.c_str(), 
                    &sensor_id) == DSL_RESULT_SUCCESS );
                REQUIRE( sensor_id == new_sensor_id );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new USB Camera Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new USB Source is created" ) 
        {
            REQUIRE( dsl_source_usb_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                const wchar_t* device_location;
                REQUIRE( dsl_source_usb_device_location_get(source_name.c_str(), 
                    &device_location) == DSL_RESULT_SUCCESS );
                std::wstring ret_device_location(device_location);
                REQUIRE( ret_device_location == def_device_location );
                REQUIRE( dsl_source_dimensions_get(source_name.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(source_name.c_str(), 
                    &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == width );
                REQUIRE( ret_height == height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(source_name.c_str()) == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new USB Camera Source set/get its device location correctly", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_usb_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

        WHEN( "The USB Source's device-location is set" ) 
        {
            // Check default first
            const wchar_t* device_location;
            REQUIRE( dsl_source_usb_device_location_get(source_name.c_str(), 
                &device_location) == DSL_RESULT_SUCCESS );
            std::wstring ret_device_location(device_location);
            REQUIRE( ret_device_location == def_device_location );
            
            std::wstring new_device_location(L"/dev/video1");
            REQUIRE( dsl_source_usb_device_location_set(source_name.c_str(), 
                new_device_location.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct updated value is returned on get" ) 
            {
                REQUIRE( dsl_source_usb_device_location_get(source_name.c_str(), 
                    &device_location) == DSL_RESULT_SUCCESS );
                ret_device_location = device_location;
                REQUIRE( ret_device_location == new_device_location );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    


SCENARIO( "A Client is able to update the Source in-use max", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_source_num_in_use_max_get() == DSL_DEFAULT_SOURCE_IN_USE_MAX );
        REQUIRE( dsl_source_num_in_use_get() == 0 );
        
        WHEN( "The in-use-max is updated by the client" )   
        {
            uint new_max = 128;
            
            REQUIRE( dsl_source_num_in_use_max_set(new_max) == true );
            
            THEN( "The new in-use-max will be returned to the client on get" )
            {
                REQUIRE( dsl_source_num_in_use_max_get() == new_max );
            }
        }
    }
}

SCENARIO( "A Source added to a Pipeline updates the in-use number", "[source-api]" )
{
    GIVEN( "A new Source and new Pipeline" )
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 1 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source removed from a Pipeline updates the in-use number", "[source-api]" )
{
    GIVEN( "A new Pipeline with a Source" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        
        REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
            source_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 1 );

        WHEN( "The Source is removed from, the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(),
                source_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "Adding multiple Sources to a Pipelines updates the in-use number", "[source-api]" )
{
    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(source_name.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(source_name2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        WHEN( "Each Sources is added to a different Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipeline_name2.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 2 );

                REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                    source_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_component_remove(pipeline_name2.c_str(), 
                    source_name2.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_num_in_use_get() == 0 );
            }
        }
    }
}

SCENARIO( "Adding greater than max Sources to all Pipelines fails", "[source-api]" )
{
    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(source_name.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(source_name2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(source_name3.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name3.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        // Reduce the max to less than 3
        REQUIRE( dsl_source_num_in_use_max_set(2) == true );

        WHEN( "The max number of sources are added to Pipelines" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipeline_name2.c_str(), 
                source_name2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding an additional Source to a Pipeline will fail" )
            {
                REQUIRE( dsl_pipeline_component_add(pipeline_name3.c_str(), 
                    source_name3.c_str()) == DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                
                // Set back to default for other tests
                REQUIRE( dsl_source_num_in_use_max_set(DSL_DEFAULT_SOURCE_IN_USE_MAX) == true );
                REQUIRE( dsl_source_num_in_use_get() == 0 );
            }
        }
    }
}


SCENARIO( "A Source not-in-use can not be Paused or Resumed", "[source-api]" )
{
    GIVEN( "A new Source not in use by a Pipeline" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is not in use by a Pipeline" )
        {
            REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Paused as it's not in use" ) 
            {
                REQUIRE( dsl_source_pause(source_name.c_str())  == DSL_RESULT_SOURCE_NOT_IN_USE );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Source is not in use by a Pipeline" )
        {
            REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
    
            THEN( "The Source can not be Resumed as it's not in use" ) 
            {
                REQUIRE( dsl_source_resume(source_name.c_str())  == DSL_RESULT_SOURCE_NOT_IN_USE );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "A Source in-use but in a null-state can not be Paused or Resumed", "[source-api]" )
{
    GIVEN( "A new Source not in use by a Pipeline" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );


        WHEN( "A new Source is in-use by a new Pipeline in a null-state" )
        {
            REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Paused as it's in a null-state" ) 
            {
                REQUIRE( dsl_source_pause(source_name.c_str())  == DSL_RESULT_SOURCE_NOT_IN_PLAY );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Source is in-use by a new Pipeline in a null-state" )
        {
            REQUIRE( dsl_source_csi_new(source_name.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                source_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Resumed as it's not in use" ) 
            {
                REQUIRE( dsl_source_resume(source_name.c_str())  == DSL_RESULT_SOURCE_NOT_IN_PAUSE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    
    
SCENARIO( "An invalid Source is caught by all Set and Get API calls", "[source-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Source Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            
        uint currBitrate(0);
        uint currInterval(0);
    
        uint newBitrate(2500000);
        uint newInterval(10);

        WHEN( "The File Sink Get-Set API called with a Fake sink" )
        {
            
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The Source Pause and Resume APIs fail correctly")
            {
                uint width(0), height(0);
                uint fps_n(0), fps_d(0);
                REQUIRE( dsl_source_dimensions_get(fakeSinkName.c_str(), &width, &height) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_frame_rate_get(fakeSinkName.c_str(), &fps_n, &fps_d) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_pause(fakeSinkName.c_str()) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_resume(fakeSinkName.c_str()) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);
                REQUIRE( dsl_source_is_live(fakeSinkName.c_str()) == DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Dewarper can be added to and removed from a Decode Source Component", "[source-api]" )
{
    GIVEN( "A new Source and new Dewarper" )
    {
        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(),
            false, intra_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_dewarper_new(dewarper_name.c_str(), defConfigFile.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Dewarper is added to the Source" ) 
        {
            REQUIRE( dsl_source_decode_dewarper_add(source_name.c_str(), 
                dewarper_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Dewarper can be removed" )
            {
                // A second call must fail
                REQUIRE( dsl_source_decode_dewarper_add(source_name.c_str(), 
                    dewarper_name.c_str()) == DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED );

                REQUIRE( dsl_source_decode_dewarper_remove(source_name.c_str()) == DSL_RESULT_SUCCESS );

                // A second time must fail
                REQUIRE( dsl_source_decode_dewarper_remove(source_name.c_str()) == DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "Adding an invalid Dewarper to a Decode Source Component fails", "[source-api]" )
{
    GIVEN( "A new Source and a Fake Sink as invalid Dewarper" )
    {
        std::wstring fakeSinkName(L"fake-sink");

        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(),
            false, intra_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );

        WHEN( "A Fake Sink is used as Dewarper" ) 
        {
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding the Fake Sink as a Dewarper will fail" )
            {
                REQUIRE( dsl_source_decode_dewarper_add(source_name.c_str(), 
                    fakeSinkName.c_str()) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                    
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An RTSP Source's Timeout can be updated correctly", "[source-api]" )
{
    GIVEN( "A new RTSP Source with a 0 timeout" )
    {
        REQUIRE( dsl_source_rtsp_new(source_name.c_str(), rtsp_uri.c_str(), protocol,
            intra_decode, interval, latency, timeout) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_source_rtsp_timeout_get(source_name.c_str(), &retTimeout) == DSL_RESULT_SUCCESS );
        REQUIRE( retTimeout == timeout );

        WHEN( "The RTSP Source's buffer timeout is updated" ) 
        {
            uint timeout(321);
            uint retTimeout(0);
            REQUIRE( dsl_source_rtsp_timeout_set(source_name.c_str(), timeout) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned after update" )
            {
                REQUIRE( dsl_source_rtsp_timeout_get(source_name.c_str(), &retTimeout) == DSL_RESULT_SUCCESS );
                REQUIRE( retTimeout == timeout );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An RTSP Source's Reconnect Stats can gotten and cleared", "[source-api]" )
{
    GIVEN( "A new RTSP Source with a 0 timeout" )
    {
        REQUIRE( dsl_source_rtsp_new(source_name.c_str(), rtsp_uri.c_str(), protocol,
            intra_decode, interval, latency, timeout) == DSL_RESULT_SUCCESS );
            
        WHEN( "A client gets an RTSP Source's connection data" ) 
        {
            dsl_rtsp_connection_data data{0};
            data.first_connected = 123;
            data.last_connected = 456;
            data.last_disconnected = 789;
            data.count = 654;
            data.is_in_reconnect = true;
            data.retries = 444;
            REQUIRE( dsl_source_rtsp_connection_data_get(source_name.c_str(), 
                &data) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned after update" )
            {
                REQUIRE( data.first_connected == 0 );
                REQUIRE( data.last_connected == 0 );
                REQUIRE( data.last_disconnected == 0 );
                REQUIRE( data.count == 0 );
                REQUIRE( data.is_in_reconnect == 0 );
                REQUIRE( data.retries == 0 );

                REQUIRE( dsl_source_rtsp_connection_stats_clear(source_name.c_str()) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static void source_state_change_listener_cb1(uint prev_state, uint curr_state, void* user_data)
{
}

SCENARIO( "An RTSP state-change-listener can be added and removed", "[source-api]" )
{
    GIVEN( "A new RTSP Source and client listener callback" )
    {
        REQUIRE( dsl_source_rtsp_new(source_name.c_str(), rtsp_uri.c_str(), protocol,
            intra_decode, interval, latency, timeout) == DSL_RESULT_SUCCESS );

        WHEN( "A state-change-listner is added" )
        {
            REQUIRE( dsl_source_rtsp_state_change_listener_add(source_name.c_str(),
                source_state_change_listener_cb1, NULL) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_source_rtsp_state_change_listener_add(source_name.c_str(),
                source_state_change_listener_cb1, NULL) == DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED );

            THEN( "The same listner can be remove" ) 
            {
                REQUIRE( dsl_source_rtsp_state_change_listener_remove(source_name.c_str(),
                    source_state_change_listener_cb1) == DSL_RESULT_SUCCESS );

                // calling a second time must faile
                REQUIRE( dsl_source_rtsp_state_change_listener_remove(source_name.c_str(),
                    source_state_change_listener_cb1) == DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new File Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "Attributes for a new File Source" ) 
    {
        std::wstring w_file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
        std::string file_path(w_file_path.begin(), w_file_path.end());
        
        char absolutePath[PATH_MAX+1];
        std::string full_file_path(realpath(file_path.c_str(), absolutePath));
        std::wstring w_full_file_path(full_file_path.begin(), full_file_path.end());
        w_full_file_path.insert(0, L"file:");
        
        boolean repeat_enabled(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new File Source is created with a File Path" ) 
        {
            REQUIRE( dsl_source_file_new(source_name.c_str(), 
                w_file_path.c_str(), repeat_enabled) == DSL_RESULT_SUCCESS );

            THEN( "The correct attribute values are returned" ) 
            {
                const wchar_t* pRetFilePath;
                REQUIRE( dsl_source_file_path_get(source_name.c_str(), 
                    &pRetFilePath) == DSL_RESULT_SUCCESS );
                std::wstring w_ret_file_path(pRetFilePath);
                REQUIRE( w_ret_file_path == w_full_file_path);
                
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(source_name.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(source_name.c_str(), 
                    &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == 1920 );
                REQUIRE( ret_height == 1080 );
                REQUIRE( ret_fps_n == 0 );
                REQUIRE( ret_fps_d == 0 );
                REQUIRE( dsl_source_is_live(source_name.c_str()) == false );
                boolean ret_repeat_enabled(0);
                REQUIRE( dsl_source_file_repeat_enabled_get(source_name.c_str(), 
                    &ret_repeat_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_repeat_enabled == repeat_enabled );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A new File Source is created with a File Path" ) 
        {
            REQUIRE( dsl_source_file_new(source_name.c_str(), 
                NULL, repeat_enabled) == DSL_RESULT_SUCCESS );

            THEN( "The correct attribute values are returned" ) 
            {
                const wchar_t* pRetFilePath; 
                std::wstring empty_file_path;
                REQUIRE( dsl_source_file_path_get(source_name.c_str(), 
                    &pRetFilePath) == DSL_RESULT_SUCCESS );
                std::wstring ret_file_path(pRetFilePath);
                REQUIRE( ret_file_path == empty_file_path );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A File Source Component can Set/Get its Repeat Enabled setting", "[source-api]" )
{
    GIVEN( "A new File Source" )
    {
        REQUIRE( dsl_source_file_new(source_name.c_str(), 
            uri.c_str(), false) == DSL_RESULT_SUCCESS );

        boolean retRepeatEnabled(true);
        REQUIRE( dsl_source_file_repeat_enabled_get(source_name.c_str(), 
            &retRepeatEnabled) == DSL_RESULT_SUCCESS );
        REQUIRE( retRepeatEnabled == false );

        WHEN( "The Source's Repeat Enabled setting is set" ) 
        {
            REQUIRE( dsl_source_file_repeat_enabled_set(source_name.c_str(), 
                true) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_source_file_repeat_enabled_get(source_name.c_str(), 
                    &retRepeatEnabled) == DSL_RESULT_SUCCESS );
                REQUIRE( retRepeatEnabled == true );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Multi-Image Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "Attributes for a new Multi Image Source" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Multi Image Source is created" ) 
        {
            REQUIRE( dsl_source_image_multi_new(source_name.c_str(), 
                jpeg_file_path_multi.c_str(), fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                boolean loop_enabled(true);

                REQUIRE( dsl_source_image_multi_loop_enabled_get(source_name.c_str(), 
                    &loop_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( loop_enabled == false );

                int start_index(99), stop_index(99);
                REQUIRE( dsl_source_image_multi_indices_get(source_name.c_str(), 
                    &start_index, &stop_index) == DSL_RESULT_SUCCESS );
                REQUIRE( start_index == 0 );
                REQUIRE( stop_index == -1 );
                
                REQUIRE( dsl_source_is_live(source_name.c_str()) == false );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Multi Image Source Component can Set/Get its settings correctly", "[source-api]" )
{
    GIVEN( "A new Multi-Image Source" )
    {
        REQUIRE( dsl_source_image_multi_new(source_name.c_str(), 
            jpeg_file_path_multi.c_str(), fps_n, fps_d) == DSL_RESULT_SUCCESS );

        WHEN( "The Source's loop-enabled setting is set" ) 
        {
            boolean new_loop_enabled(true);
            REQUIRE( dsl_source_image_multi_loop_enabled_set(source_name.c_str(), 
                new_loop_enabled) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" )
            {
                boolean ret_loop_enabled(false);

                REQUIRE( dsl_source_image_multi_loop_enabled_get(source_name.c_str(), 
                    &ret_loop_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_loop_enabled == new_loop_enabled );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "The Source's start and stop index setting is set" ) 
        {
            int new_start_index(4), new_stop_index(5);
            REQUIRE( dsl_source_image_multi_indices_set(source_name.c_str(), 
                new_start_index, new_stop_index) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" )
            {
                int ret_start_index(99), ret_stop_index(99);
                REQUIRE( dsl_source_image_multi_indices_get(source_name.c_str(), 
                    &ret_start_index, &ret_stop_index) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_start_index == new_start_index );
                REQUIRE( ret_stop_index == new_stop_index );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A new Image Stream Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "Attributes for a new Image Source" ) 
    {
        uint actual_width(1280);
        uint actual_height(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Image Source is created" ) 
        {
            REQUIRE( dsl_source_image_stream_new(source_name.c_str(), image_path.c_str(),
                is_live, fps_n, fps_d, timeout) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(source_name.c_str(), 
                    &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(source_name.c_str(), 
                    &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == actual_width );
                REQUIRE( ret_height == actual_height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(source_name.c_str()) == false );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Image Stream Source Component can Set/Get its Display Timeout setting", "[source-api]" )
{
    GIVEN( "A new File Source" )
    {
        REQUIRE( dsl_source_image_stream_new(source_name.c_str(), image_path.c_str(),
            is_live, fps_n, fps_d, timeout) == DSL_RESULT_SUCCESS );

        uint retTimeout(321);
        REQUIRE( dsl_source_image_stream_timeout_get(source_name.c_str(), 
            &retTimeout) == DSL_RESULT_SUCCESS );
        REQUIRE( retTimeout == timeout );

        WHEN( "The Source's Timeout setting is set" ) 
        {
            uint newTimeout(444);
            REQUIRE( dsl_source_image_stream_timeout_set(source_name.c_str(), 
                newTimeout) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_source_image_stream_timeout_get(source_name.c_str(), 
                    &retTimeout) == DSL_RESULT_SUCCESS );
                REQUIRE( retTimeout == newTimeout );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Source API checks for NULL input parameters", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        
        int start_index(0);

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_source_csi_new(NULL, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_csi_sensor_id_get(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_csi_sensor_id_get(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_csi_sensor_id_set(NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_usb_new(NULL, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_usb_device_location_get(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_usb_device_location_get(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_usb_device_location_set(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_usb_device_location_set(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_source_uri_new(NULL, NULL, false, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_uri_new(source_name.c_str(), NULL, false, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_new(NULL, NULL, 0, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_new(source_name.c_str(), NULL, 0, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_file_new(NULL, NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                // Note NULL file_path is valid for File and Image Sources

                REQUIRE( dsl_source_dimensions_get(NULL, &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_frame_rate_get(NULL, &fps_n, &fps_d) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_decode_uri_get(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_uri_get(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_uri_set(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_uri_set(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_decode_dewarper_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_dewarper_add(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_dewarper_remove(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_rtsp_tap_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_tap_add(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_tap_remove(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_image_multi_new(NULL, 
                    NULL, fps_n, fps_d) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_multi_new(source_name.c_str(), 
                    NULL, fps_n, fps_d) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_multi_loop_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_multi_loop_enabled_get(source_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_multi_loop_enabled_set(NULL, 
                    false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_multi_indices_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_multi_indices_get(source_name.c_str(), 
                    &start_index, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_pause(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_resume(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                
                REQUIRE( dsl_source_pph_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_pph_add(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_pph_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_pph_remove(source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
