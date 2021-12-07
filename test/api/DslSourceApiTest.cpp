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

SCENARIO( "The Components container is updated correctly on new source", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is created" ) 
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

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
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );


        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        
        WHEN( "The Source is deleted" )
        {
            REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
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
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

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
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Source is removed from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can be deleted successfully" ) 
            {
                REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );

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
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is created" ) 
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(sourceName.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(sourceName.c_str(), &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == width );
                REQUIRE( ret_height == height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(sourceName.c_str()) == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A new USB Camera Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sourceName(L"usb-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new USB Source is created" ) 
        {
            REQUIRE( dsl_source_usb_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                uint ret_width(0), ret_height(0), ret_fps_n(0), ret_fps_d(0);
                REQUIRE( dsl_source_dimensions_get(sourceName.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_frame_rate_get(sourceName.c_str(), &ret_fps_n, &ret_fps_d) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == width );
                REQUIRE( ret_height == height );
                REQUIRE( ret_fps_n == fps_n );
                REQUIRE( ret_fps_d == fps_d );
                REQUIRE( dsl_source_is_live(sourceName.c_str()) == 0 );

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
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        WHEN( "The Source is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

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
        std::wstring pipelineName(L"test-pipeline");
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 1 );

        WHEN( "The Source is removed from, the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

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
    std::wstring sourceName1  = L"csi-source1";
    std::wstring pipelineName1  = L"test-pipeline1";
    std::wstring sourceName2  = L"csi-source2";
    std::wstring pipelineName2  = L"test-pipeline2";

    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(sourceName1.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        WHEN( "Each Sources is added to a different Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sourceName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sourceName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_source_num_in_use_get() == 2 );

                REQUIRE( dsl_pipeline_component_remove(pipelineName1.c_str(), 
                    sourceName1.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_component_remove(pipelineName2.c_str(), 
                    sourceName2.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_num_in_use_get() == 0 );
            }
        }
    }
}

SCENARIO( "Adding greater than max Sources to all Pipelines fails", "[source-api]" )
{
    std::wstring sourceName1  = L"csi-source1";
    std::wstring pipelineName1  = L"test-pipeline1";
    std::wstring sourceName2  = L"csi-source2";
    std::wstring pipelineName2  = L"test-pipeline2";
    std::wstring sourceName3  = L"csi-source3";
    std::wstring pipelineName3  = L"test-pipeline3";

    GIVEN( "Two new Sources and two new Pipeline" )
    {
        REQUIRE( dsl_source_csi_new(sourceName1.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName2.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName3.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName3.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_num_in_use_get() == 0 );

        // Reduce the max to less than 3
        REQUIRE( dsl_source_num_in_use_max_set(2) == true );

        WHEN( "The max number of sources are added to Pipelines" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sourceName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sourceName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding an additional Source to a Pipeline will fail" )
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName3.c_str(), 
                    sourceName3.c_str()) == DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED );

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
        std::wstring sourceName(L"csi-source");
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Source is not in use by a Pipeline" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Paused as it's not in use" ) 
            {
                REQUIRE( dsl_source_pause(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_USE );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Source is not in use by a Pipeline" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
    
            THEN( "The Source can not be Resumed as it's not in use" ) 
            {
                REQUIRE( dsl_source_resume(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_USE );
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
        std::wstring pipelineName  = L"test-pipeline";

        std::wstring sourceName  = L"csi-source";
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);

        REQUIRE( dsl_component_list_size() == 0 );


        WHEN( "A new Source is in-use by a new Pipeline in a null-state" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Paused as it's in a null-state" ) 
            {
                REQUIRE( dsl_source_pause(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_PLAY );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "A new Source is in-use by a new Pipeline in a null-state" )
        {
            REQUIRE( dsl_source_csi_new(sourceName.c_str(), width, height, fps_n, fps_d) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                sourceName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Source can not be Resumed as it's not in use" ) 
            {
                REQUIRE( dsl_source_resume(sourceName.c_str())  == DSL_RESULT_SOURCE_NOT_IN_PAUSE );
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
        std::wstring sourceName = L"uri-source";
        std::wstring uri = L"./test/streams/sample_1080p_h264.mp4";
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring dewarperName(L"dewarper");
        std::wstring defConfigFile(L"./test/configs/config_dewarper.txt");

        REQUIRE( dsl_source_uri_new(sourceName.c_str(), uri.c_str(),
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_dewarper_new(dewarperName.c_str(), defConfigFile.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Dewarper is added to the Source" ) 
        {
            REQUIRE( dsl_source_decode_dewarper_add(sourceName.c_str(), 
                dewarperName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Dewarper can be removed" )
            {
                // A second call must fail
                REQUIRE( dsl_source_decode_dewarper_add(sourceName.c_str(), 
                    dewarperName.c_str()) == DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED );

                REQUIRE( dsl_source_decode_dewarper_remove(sourceName.c_str()) == DSL_RESULT_SUCCESS );

                // A second time must fail
                REQUIRE( dsl_source_decode_dewarper_remove(sourceName.c_str()) == DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "Adding an invalid Dewarper to a Decode Source Component fails", "[source-api]" )
{
    GIVEN( "A new Source and a Fake Sink as invalid Dewarper" )
    {
        std::wstring sourceName = L"uri-source";
        std::wstring uri = L"./test/streams/sample_1080p_h264.mp4";
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring fakeSinkName(L"fake-sink");

        REQUIRE( dsl_source_uri_new(sourceName.c_str(), uri.c_str(),
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        WHEN( "A Fake Sink is used as Dewarper" ) 
        {
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Adding the Fake Sink as a Dewarper will fail" )
            {
                REQUIRE( dsl_source_decode_dewarper_add(sourceName.c_str(), 
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
        std::wstring rtspSourceName(L"rtsp-SOURCE");
        std::wstring uri(L"rtsp://username:password@192.168.0.14:554");
        uint protocol(DSL_RTP_ALL);
        uint intra_decode(false);
        uint interval;
        uint latency(100);
        uint timeout(0);
        uint retTimeout(123);
        
        REQUIRE( dsl_source_rtsp_new(rtspSourceName.c_str(), uri.c_str(), protocol,
            intra_decode, interval, latency, timeout) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_source_rtsp_timeout_get(rtspSourceName.c_str(), &retTimeout) == DSL_RESULT_SUCCESS );
        REQUIRE( retTimeout == timeout );

        WHEN( "The RTSP Source's buffer timeout is updated" ) 
        {
            uint timeout(321);
            uint retTimeout(0);
            REQUIRE( dsl_source_rtsp_timeout_set(rtspSourceName.c_str(), timeout) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned after update" )
            {
                REQUIRE( dsl_source_rtsp_timeout_get(rtspSourceName.c_str(), &retTimeout) == DSL_RESULT_SUCCESS );
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
        std::wstring rtspSourceName(L"rtsp-SOURCE");
        std::wstring uri(L"rtsp://username:password@192.168.0.14:554");
        uint protocol(DSL_RTP_ALL);
        uint intra_decode(false);
        uint interval;
        uint latency(100);
        uint timeout(0);
        uint retTimeout(0);
        
        REQUIRE( dsl_source_rtsp_new(rtspSourceName.c_str(), uri.c_str(), protocol,
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
            REQUIRE( dsl_source_rtsp_connection_data_get(rtspSourceName.c_str(), 
                &data) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned after update" )
            {
                REQUIRE( data.first_connected == 0 );
                REQUIRE( data.last_connected == 0 );
                REQUIRE( data.last_disconnected == 0 );
                REQUIRE( data.count == 0 );
                REQUIRE( data.is_in_reconnect == 0 );
                REQUIRE( data.retries == 0 );

                REQUIRE( dsl_source_rtsp_connection_stats_clear(rtspSourceName.c_str()) == DSL_RESULT_SUCCESS );
                    
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
        std::wstring rtspSourceName(L"rtsp-source");
        std::wstring uri(L"rtsp://username:password@192.168.0.14:554");
        uint protocol(DSL_RTP_ALL);
        uint intra_decode(false);
        uint interval;
        uint latency(100);
        uint timeout(0);
        uint retTimeout(123);
        
        REQUIRE( dsl_source_rtsp_new(rtspSourceName.c_str(), uri.c_str(), protocol,
            intra_decode, interval, latency, timeout) == DSL_RESULT_SUCCESS );

        WHEN( "A state-change-listner is added" )
        {
            REQUIRE( dsl_source_rtsp_state_change_listener_add(rtspSourceName.c_str(),
                source_state_change_listener_cb1, NULL) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_source_rtsp_state_change_listener_add(rtspSourceName.c_str(),
                source_state_change_listener_cb1, NULL) == DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED );

            THEN( "The same listner can be remove" ) 
            {
                REQUIRE( dsl_source_rtsp_state_change_listener_remove(rtspSourceName.c_str(),
                    source_state_change_listener_cb1) == DSL_RESULT_SUCCESS );

                // calling a second time must faile
                REQUIRE( dsl_source_rtsp_state_change_listener_remove(rtspSourceName.c_str(),
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
        std::wstring source_name(L"file-source");
        std::wstring w_file_path(L"./test/streams/sample_1080p_h264.mp4");
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
        std::wstring sourceName = L"file-source";
        std::wstring file_path = L"./test/streams/sample_1080p_h264.mp4";

        REQUIRE( dsl_source_file_new(sourceName.c_str(), 
            file_path.c_str(), false) == DSL_RESULT_SUCCESS );

        boolean retRepeatEnabled(true);
        REQUIRE( dsl_source_file_repeat_enabled_get(sourceName.c_str(), 
            &retRepeatEnabled) == DSL_RESULT_SUCCESS );
        REQUIRE( retRepeatEnabled == false );

        WHEN( "The Source's Repeat Enabled setting is set" ) 
        {
            REQUIRE( dsl_source_file_repeat_enabled_set(sourceName.c_str(), 
                true) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_source_file_repeat_enabled_get(sourceName.c_str(), 
                    &retRepeatEnabled) == DSL_RESULT_SUCCESS );
                REQUIRE( retRepeatEnabled == true );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}


SCENARIO( "A new Image Source returns the correct attribute values", "[source-api]" )
{
    GIVEN( "Attributes for a new Image Source" ) 
    {
        std::wstring source_name(L"image-source");
        std::wstring image_path(L"./test/streams/first-person-occurrence-438.jpeg");
        boolean is_live(false);
        uint fps_n(30);
        uint fps_d(1);
        uint timeout(123);
        uint actual_width(136);
        uint actual_height(391);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Image Source is created" ) 
        {
            REQUIRE( dsl_source_image_new(source_name.c_str(), image_path.c_str(),
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

SCENARIO( "A Image Source Component can Set/Get its Display Timeout setting", "[source-api]" )
{
    GIVEN( "A new File Source" )
    {
        std::wstring source_name(L"image-source");
        std::wstring image_path(L"./test/streams/first-person-occurrence-438.jpeg");
        boolean is_live(false);
        uint fps_n(30);
        uint fps_d(1);
        uint timeout(123);

        REQUIRE( dsl_source_image_new(source_name.c_str(), image_path.c_str(),
            is_live, fps_n, fps_d, timeout) == DSL_RESULT_SUCCESS );

        uint retTimeout(321);
        REQUIRE( dsl_source_image_timeout_get(source_name.c_str(), 
            &retTimeout) == DSL_RESULT_SUCCESS );
        REQUIRE( retTimeout == timeout );

        WHEN( "The Source's Timeout setting is set" ) 
        {
            uint newTimeout(444);
            REQUIRE( dsl_source_image_timeout_set(source_name.c_str(), 
                newTimeout) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" )
            {
                REQUIRE( dsl_source_image_timeout_get(source_name.c_str(), 
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
        std::wstring sourceName  = L"test-source";
        std::wstring otherName  = L"other";
        
        uint cache_size(0), width(0), height(0), fps_n(0), fps_d(0), bitrate(0), interval(0), udpPort(0), rtspPort(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_source_csi_new( NULL, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_usb_new( NULL, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_uri_new( NULL, NULL, false, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_uri_new( sourceName.c_str(), NULL, false, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_new( NULL, NULL, 0, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_new( sourceName.c_str(), NULL, 0, 0, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_image_new( NULL, NULL, false, 0, 0, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_file_new( NULL, NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                // Note NULL file_path is valid for File and Image Sources

                REQUIRE( dsl_source_dimensions_get( NULL, &width, &height ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_frame_rate_get( NULL, &fps_n, &fps_d ) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_decode_uri_get( NULL, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_uri_get( sourceName.c_str(), NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_uri_set( NULL, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_uri_set( sourceName.c_str(), NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_decode_dewarper_add( NULL, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_dewarper_add( sourceName.c_str(), NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_decode_dewarper_remove( NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_rtsp_tap_add( NULL, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_tap_add( sourceName.c_str(), NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_rtsp_tap_remove( NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_source_pause( NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_source_resume( NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
