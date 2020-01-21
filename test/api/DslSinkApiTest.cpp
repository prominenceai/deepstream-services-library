/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

SCENARIO( "The Components container is updated correctly on new Fake Sink", "[fake-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring sinkName = L"fake-sink";

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Overlay Sink is created" ) 
        {
            REQUIRE( dsl_sink_fake_new(sinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Fink Sink delete", "[fake-sink-api]" )
{
    GIVEN( "A Fake Sink Component" ) 
    {
        std::wstring sinkName = L"fake-sink";


        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_fake_new(sinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Overlay Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(sinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Overlay Sink", "[overlay-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Overlay Sink is created" ) 
        {

            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Overlay Sink delete", "[overlay-sink-api]" )
{
    GIVEN( "An Overlay Sink Component" ) 
    {
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A new Overlay Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be deleted", "[overlay-sink-api]" )
{
    GIVEN( "A new Overlay Sink and new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Overlay Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Overlay Sink can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Overlay Sink, once removed from a Pipeline, can be deleted", "[overlay-sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring overlaySinkName = L"overlay-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Overlay Sink is removed the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Overlay Sink can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be added to a second Pipeline", "[overlay-sink-api]" )
{
    GIVEN( "A new Overlay Sink and two new Pipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring overlaySinkName = L"overlay-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Overlay Sink is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Overlay Sink can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Window Sink", "[window-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Window Sink is created" ) 
        {

            REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Window Sink delete", "[window-sink-api]" )
{
    GIVEN( "An Window Sink Component" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A new Window Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink in use can't be deleted", "[window-sink-api]" )
{
    GIVEN( "A new Window Sink and new Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Window Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink, once removed from a Pipeline, can be deleted", "[window-sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink is removed the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink in use can't be added to a second Pipeline", "[window-sink-api]" )
{
    GIVEN( "A new Window Sink and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    windowSinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new File Sink", "[file-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new File Sink is created" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, container, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retCodec(0), retContainer(0);
                REQUIRE( dsl_sink_file_video_formats_get(fileSinkName.c_str(), &retCodec, &retContainer) == DSL_RESULT_SUCCESS );
                REQUIRE( retCodec == codec );
                REQUIRE( retContainer == container );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on File Sink delete", "[file-sink-api]" )
{
    GIVEN( "An File Sink Component" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            codec, container, bitrate, interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new File Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(fileSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Creating a new File Sink with an invalid Codec will fail", "[file-sink-api]" )
{
    GIVEN( "Attributes for a new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_MPEG4 + 1);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When creating a new File Sink with an invalid Codec" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, container, bitrate, interval) == DSL_RESULT_SINK_CODEC_VALUE_INVALID );

            THEN( "The list size is left unchanged" ) 
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "Creating a new File Sink with an invalid Container will fail", "[file-sink-api]" )
{
    GIVEN( "Attributes for a new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_MPEG4);
        uint container(DSL_CONTAINER_MK4 + 1);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When creating a new File Sink with an invalid Container" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, container, bitrate, interval) == DSL_RESULT_SINK_CONTAINER_VALUE_INVALID );

            THEN( "The list size is left unchanged" ) 
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A File Sink's Encoder settings can be updated", "[file-sink-api]" )
{
    GIVEN( "A new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint initBitrate(2000000);
        uint initInterval(0);

        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            codec, container, initBitrate, initInterval) == DSL_RESULT_SUCCESS );
            
        uint currBitrate(0);
        uint currInterval(0);
    
        REQUIRE( dsl_sink_file_encoder_settings_get(fileSinkName.c_str(), &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
        REQUIRE( currBitrate == initBitrate );
        REQUIRE( currInterval == initInterval );

        WHEN( "The FileSinkBintr's Encoder settings are Set" )
        {
            uint newBitrate(2500000);
            uint newInterval(10);
            
            REQUIRE( dsl_sink_file_encoder_settings_set(fileSinkName.c_str(), newBitrate, newInterval) == DSL_RESULT_SUCCESS);

            THEN( "The FileSinkBintr's new Encoder settings are returned on Get")
            {
                REQUIRE( dsl_sink_file_encoder_settings_get(fileSinkName.c_str(), &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
                REQUIRE( currBitrate == newBitrate );
                REQUIRE( currInterval == newInterval );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new DSL_CODEC_H264 RTSP Sink", "[rtsp-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTSP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
                udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retUdpPort(0), retRtspPort(0), retCodec(0);
                dsl_sink_rtsp_server_settings_get(rtspSinkName.c_str(), &retUdpPort, &retRtspPort, &retCodec);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( retCodec == codec );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on DSL_CODEC_H264 RTSP Sink delete", "[rtsp-sink-api]" )
{
    GIVEN( "An RTSP Sink Component" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new RTSP Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(rtspSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new DSL_CODEC_H265 RTSP Sink", "[rtsp-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new RTSP Sink is created" ) 
        {
            REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
                udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint retUdpPort(0), retRtspPort(0), retCodec(0);
                dsl_sink_rtsp_server_settings_get(rtspSinkName.c_str(), &retUdpPort, &retRtspPort, &retCodec);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( retCodec == codec );
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on DSL_CODEC_H265 RTSP Sink delete", "[rtsp-sink-api]" )
{
    GIVEN( "An RTSP Sink Component" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, bitrate, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new RTSP Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(rtspSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
SCENARIO( "An RTSP Sink's Encoder settings can be updated", "[rtsp-sink-api]" )
{
    GIVEN( "A new RTSP Sink" ) 
    {
        std::wstring rtspSinkName(L"rtsp-sink");
        std::wstring host(L"224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint initBitrate(4000000);
        uint initInterval(0);

        REQUIRE( dsl_sink_rtsp_new(rtspSinkName.c_str(), host.c_str(),
            udpPort, rtspPort, codec, initBitrate, initInterval) == DSL_RESULT_SUCCESS );
            
        uint currBitrate(0);
        uint currInterval(0);
    
        REQUIRE( dsl_sink_rtsp_encoder_settings_get(rtspSinkName.c_str(), &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
        REQUIRE( currBitrate == initBitrate );
        REQUIRE( currInterval == initInterval );

        WHEN( "The RTSP Sink's Encoder settings are Set" )
        {
            uint newBitrate(2500000);
            uint newInterval(10);
            
            REQUIRE( dsl_sink_rtsp_encoder_settings_set(rtspSinkName.c_str(), newBitrate, newInterval) == DSL_RESULT_SUCCESS);

            THEN( "The RTSP Sink's new Encoder settings are returned on Get")
            {
                REQUIRE( dsl_sink_rtsp_encoder_settings_get(rtspSinkName.c_str(), &currBitrate, &currInterval) == DSL_RESULT_SUCCESS);
                REQUIRE( currBitrate == newBitrate );
                REQUIRE( currInterval == newInterval );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Client is able to update the Sink in-use max", "[sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_num_in_use_max_get() == DSL_DEFAULT_SINK_IN_USE_MAX );
        REQUIRE( dsl_sink_num_in_use_get() == 0 );
        
        WHEN( "The in-use-max is updated by the client" )   
        {
            uint new_max = 128;
            
            REQUIRE( dsl_sink_num_in_use_max_set(new_max) == true );
            
            THEN( "The new in-use-max will be returned to the client on get" )
            {
                REQUIRE( dsl_sink_num_in_use_max_get() == new_max );
            }
        }
    }
}

SCENARIO( "A Sink added to a Pipeline updates the in-use number", "[sink-api]" )
{
    GIVEN( "A new Sink and new Pipeline" )
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_num_in_use_get() == 0 );

        WHEN( "The Window Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_sink_num_in_use_get() == 1 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Sink removed from a Pipeline updates the in-use number", "[sink-api]" )
{
    GIVEN( "A new Pipeline with a Sink" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            windowSinkName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_num_in_use_get() == 1 );

        WHEN( "The Source is removed from, the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(),
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_sink_num_in_use_get() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "Adding multiple Sinks to multiple Pipelines updates the in-use number", "[sink-api]" )
{
    GIVEN( "Two new Sinks and two new Pipeline" )
    {
        std::wstring sinkName1 = L"window-sink1";
        std::wstring pipelineName1  = L"test-pipeline1";
        std::wstring sinkName2 = L"window-sink2";
        std::wstring pipelineName2  = L"test-pipeline2";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(sinkName1.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_overlay_new(sinkName2.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_sink_num_in_use_get() == 0 );

        WHEN( "Each Sink is added to a different Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                sinkName1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                sinkName2.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct in-use number is returned to the client" )
            {
                REQUIRE( dsl_sink_num_in_use_get() == 2 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_source_num_in_use_get() == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}
