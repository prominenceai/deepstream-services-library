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
#include "Dsl.h"
#include "DslSinkBintr.h"

using namespace DSL;

SCENARIO( "A new OverlaySinkBintr is created correctly",  "[OverlaySinkBintr]" )
{
    GIVEN( "Attributes for a new Overlay Sink" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        WHEN( "The OverlaySinkBintr is created " )
        {
            DSL_OVERLAY_SINK_PTR pSinkBintr = 
                DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->GetDisplayId() == 0 );
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new OverlaySinkBintr can LinkAll Child Elementrs", "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new OverlaySinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The OverlaySinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked OverlaySinkBintr can UnlinkAll Child Elementrs", "[OverlaySinkBintr]" )
{
    GIVEN( "A OsdBintr in a linked state" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A OverlaySinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The OverlaySinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "An OverlaySinkBintr's Display Id can be updated",  "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        uint displayId(123);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
        // ensure display id reflects not is use
        REQUIRE( pSinkBintr->GetDisplayId() == 0 );

        WHEN( "The OverlaySinkBintr's display Id is set " )
        {
            pSinkBintr->SetDisplayId(displayId);
            THEN( "The OverlaySinkBintr's new display Id is returned on Get" )
            {
                REQUIRE( pSinkBintr->GetDisplayId() == displayId );
            }
        }
    }
}

SCENARIO( "An OverlaySinkBintr's Offsets can be updated", "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint initOffsetX(0);
        uint initOffsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), initOffsetX, initOffsetY, sinkW, sinkH);
            
        uint currOffsetX(0);
        uint currOffsetY(0);
    
        pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
        REQUIRE( currOffsetX == initOffsetX );
        REQUIRE( currOffsetY == initOffsetY );

        WHEN( "The OverlaySinkBintr's Offsets are Set" )
        {
            uint newOffsetX(80);
            uint newOffsetY(20);
            
            pSinkBintr->SetOffsets(newOffsetX, newOffsetY);

            THEN( "The OverlaySinkBintr's new demensions are returned on Get")
            {
                pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
                REQUIRE( currOffsetX == newOffsetX );
                REQUIRE( currOffsetY == newOffsetY );
            }
        }
    }
}

SCENARIO( "An OverlaySinkBintr's Dimensions can be updated", "[OverlaySinkBintr]" )
{
    GIVEN( "A new OverlaySinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
            
        uint currSinkW(0);
        uint currSinkH(0);
    
        pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
        REQUIRE( currSinkW == initSinkW );
        REQUIRE( currSinkH == initSinkH );

        WHEN( "The OverlaySinkBintr's dimensions are Set" )
        {
            uint newSinkW(1280);
            uint newSinkH(720);
            
            pSinkBintr->SetDimensions(newSinkW, newSinkH);

            THEN( "The OverlaySinkBintr's new dimensions are returned on Get")
            {
                pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
                REQUIRE( currSinkW == newSinkW );
                REQUIRE( currSinkH == newSinkH );
            }
        }
    }
}

SCENARIO( "A new WindowSinkBintr is created correctly",  "[WindowSinkBintr]" )
{
    GIVEN( "Attributes for a new Window Sink" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        WHEN( "The WindowSinkBintr is created " )
        {
            DSL_WINDOW_SINK_PTR pSinkBintr = 
                DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pSinkBintr->IsWindowCapable() == true );
            }
        }
    }
}

SCENARIO( "A new WindowSinkBintr can LinkAll Child Elementrs", "[WindowSinkBintr]" )
{
    GIVEN( "A new WindowSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new WindowSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The WindowSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked WindowSinkBintr can UnlinkAll Child Elementrs", "[WindowSinkBintr]" )
{
    GIVEN( "A WindowSinkBintr in a linked state" ) 
    {
        std::string sinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A WindowSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The OverlaySinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A WindowSinkBintr's Offsets can be updated", "[WindowSinkBintr]" )
{
    GIVEN( "A new WindowSinkBintr in memory" ) 
    {
        std::string sinkName("window-sink");
        uint initOffsetX(0);
        uint initOffsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), initOffsetX, initOffsetY, sinkW, sinkH);
            
        uint currOffsetX(0);
        uint currOffsetY(0);
    
        pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
        REQUIRE( currOffsetX == initOffsetX );
        REQUIRE( currOffsetY == initOffsetY );

        WHEN( "The WindowSinkBintr's Offsets are Set" )
        {
            uint newOffsetX(80);
            uint newOffsetY(20);
            
            pSinkBintr->SetOffsets(newOffsetX, newOffsetY);

            THEN( "The WindowSinkBintr's new demensions are returned on Get")
            {
                pSinkBintr->GetOffsets(&currOffsetX, &currOffsetY);
                REQUIRE( currOffsetX == newOffsetX );
                REQUIRE( currOffsetY == newOffsetY );
            }
        }
    }
}

SCENARIO( "An WindowSinkBintr's Dimensions can be updated", "[WindowSinkBintr]" )
{
    GIVEN( "A new WindowSinkBintr in memory" ) 
    {
        std::string sinkName("overlay-sink");
        uint offsetX(0);
        uint offsetY(0);
        uint initSinkW(300);
        uint initSinkH(200);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, initSinkW, initSinkH);
            
        uint currSinkW(0);
        uint currSinkH(0);
    
        pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
        REQUIRE( currSinkW == initSinkW );
        REQUIRE( currSinkH == initSinkH );

        WHEN( "The WindowSinkBintr's dimensions are Set" )
        {
            uint newSinkW(1280);
            uint newSinkH(720);
            
            pSinkBintr->SetDimensions(newSinkW, newSinkH);

            THEN( "The WindowSinkBintr's new dimensions are returned on Get")
            {
                pSinkBintr->GetDimensions(&currSinkW, &currSinkH);
                REQUIRE( currSinkW == newSinkW );
                REQUIRE( currSinkH == newSinkH );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_MPEG4 FileSinkBintr is created correctly",  "[FileSinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_MPEG4 File Sink" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_MPEG4);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_MPEG4 FileSinkBintr is created " )
        {
            DSL_FILE_SINK_PTR pSinkBintr = 
                DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retCodec(0), retContainer(0);
                pSinkBintr->GetVideoFormats(&retCodec, &retContainer);
                REQUIRE( retCodec == codec );
                REQUIRE( retContainer == container);
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_MPEG4 FileSinkBintr can LinkAll Child Elementrs", "[FileSinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_MPEG4 FileSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_MPEG4);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_MPEG4 FileSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_MPEG4 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_MPEG4 FileSinkBintr can UnlinkAll Child Elementrs", "[FileSinkBintr]" )
{
    GIVEN( "A DSL_CODEC_MPEG4 FileSinkBintr in a linked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_MPEG4);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_MPEG4 FileSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_MPEG4 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 FileSinkBintr is created correctly",  "[FileSinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H264 File Sink" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H264);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_H264 FileSinkBintr is created " )
        {
            DSL_FILE_SINK_PTR pSinkBintr = 
                DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retCodec(0), retContainer(0);
                pSinkBintr->GetVideoFormats(&retCodec, &retContainer);
                REQUIRE( retCodec == codec );
                REQUIRE( retContainer == container);
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 FileSinkBintr can LinkAll Child Elementrs", "[FileSinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H264 FileSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H264);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H264 FileSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H264 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H264 FileSinkBintr can UnlinkAll Child Elementrs", "[FileSinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H264 FileSinkBintr in a linked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H264);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H264 FileSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H264 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 FileSinkBintr is created correctly",  "[FileSinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H265 File Sink" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_H265 FileSinkBintr is created " )
        {
            DSL_FILE_SINK_PTR pSinkBintr = 
                DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retCodec(0), retContainer(0);
                pSinkBintr->GetVideoFormats(&retCodec, &retContainer);
                REQUIRE( retCodec == codec );
                REQUIRE( retContainer == container);
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 FileSinkBintr can LinkAll Child Elementrs", "[FileSinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H265 FileSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H265 FileSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H265 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H265 FileSinkBintr can UnlinkAll Child Elementrs", "[FileSinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H265 FileSinkBintr in a linked state" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H265 FileSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H265 FileSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A FileSinkBintr's Encoder settings can be updated", "[FileSinkBintr]" )
{
    GIVEN( "A new FileSinkBintr in memory" ) 
    {
        std::string sinkName("file-sink");
        std::string filePath("./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint container(DSL_CONTAINER_MPEG4);
        uint initBitrate(2000000);
        uint initInterval(0);

        DSL_FILE_SINK_PTR pSinkBintr = 
            DSL_FILE_SINK_NEW(sinkName.c_str(), filePath.c_str(), codec, container, initBitrate, initInterval);
            
        uint currBitrate(0);
        uint currInterval(0);
    
        pSinkBintr->GetEncoderSettings(&currBitrate, &currInterval);
        REQUIRE( currBitrate == initBitrate );
        REQUIRE( currInterval == initInterval );

        WHEN( "The FileSinkBintr's Encoder settings are Set" )
        {
            uint newBitrate(3000000);
            uint newInterval(5);
            
            pSinkBintr->SetEncoderSettings(newBitrate, newInterval);

            THEN( "The FileSinkBintr's new Encoder settings are returned on Get")
            {
                pSinkBintr->GetEncoderSettings(&currBitrate, &currInterval);
                REQUIRE( currBitrate == newBitrate );
                REQUIRE( currInterval == newInterval );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 RtspSinkBintr is created correctly",  "[RtspSinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H264 File Sink" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_H264 RtspSinkBintr is created " )
        {
            DSL_RTSP_SINK_PTR pSinkBintr = 
                DSL_RTSP_SINK_NEW(sinkName.c_str(), host.c_str(), udpPort, rtspPort, codec, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retUdpPort(0), retRtspPort(0), retCodec(0);
                pSinkBintr->GetServerSettings(&retUdpPort, &retRtspPort, &retCodec);
                REQUIRE( retUdpPort == udpPort );
                REQUIRE( retRtspPort == rtspPort );
                REQUIRE( retCodec == codec );
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H264 RtspSinkBintr can LinkAll Child Elementrs", "[RtspSinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H264 RtspSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SINK_PTR pSinkBintr = 
            DSL_RTSP_SINK_NEW(sinkName.c_str(), host.c_str(), udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H264 RtspSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H264 RtspSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H264 RtspSinkBintr can UnlinkAll Child Elementrs", "[RtspSinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H264 RtspSinkBintr in a linked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H264);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SINK_PTR pSinkBintr = 
            DSL_RTSP_SINK_NEW(sinkName.c_str(), host.c_str(), udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H264 RtspSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H264 RtspSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 RtspSinkBintr is created correctly",  "[RtspSinkBintr]" )
{
    GIVEN( "Attributes for a new DSL_CODEC_H265 File Sink" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        WHEN( "The DSL_CODEC_H265 RtspSinkBintr is created " )
        {
            DSL_RTSP_SINK_PTR pSinkBintr = 
                DSL_RTSP_SINK_NEW(sinkName.c_str(), host.c_str(), udpPort, rtspPort, codec, bitrate, interval);
            
            THEN( "The correct attribute values are returned" )
            {
                uint retUdpPort(0), retRtspPort(0), retCodec(0);
                pSinkBintr->GetServerSettings(&retUdpPort, &retRtspPort, &retCodec);
                REQUIRE( retUdpPort == udpPort);
                REQUIRE( retRtspPort == rtspPort);
                REQUIRE( retCodec == codec );
                REQUIRE( pSinkBintr->IsWindowCapable() == false );
            }
        }
    }
}

SCENARIO( "A new DSL_CODEC_H265 RtspSinkBintr can LinkAll Child Elementrs", "[RtspSinkBintr]" )
{
    GIVEN( "A new DSL_CODEC_H265 RtspSinkBintr in an Unlinked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SINK_PTR pSinkBintr = 
            DSL_RTSP_SINK_NEW(sinkName.c_str(), host.c_str(), udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );

        WHEN( "A new DSL_CODEC_H265 RtspSinkBintr is Linked" )
        {
            REQUIRE( pSinkBintr->LinkAll() == true );

            THEN( "The DSL_CODEC_H265 RtspSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked DSL_CODEC_H265 RtspSinkBintr can UnlinkAll Child Elementrs", "[RtspSinkBintr]" )
{
    GIVEN( "A DSL_CODEC_H265 RtspSinkBintr in a linked state" ) 
    {
        std::string sinkName("rtsp-sink");
        std::string host("224.224.255.255");
        uint udpPort(5400);
        uint rtspPort(8554);
        uint codec(DSL_CODEC_H265);
        uint bitrate(4000000);
        uint interval(0);

        DSL_RTSP_SINK_PTR pSinkBintr = 
            DSL_RTSP_SINK_NEW(sinkName.c_str(), host.c_str(), udpPort, rtspPort, codec, bitrate, interval);

        REQUIRE( pSinkBintr->IsLinked() == false );
        REQUIRE( pSinkBintr->LinkAll() == true );

        WHEN( "A DSL_CODEC_H265 RtspSinkBintr is Unlinked" )
        {
            pSinkBintr->UnlinkAll();

            THEN( "The DSL_CODEC_H265 RtspSinkBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pSinkBintr->IsLinked() == false );
            }
        }
    }
}
