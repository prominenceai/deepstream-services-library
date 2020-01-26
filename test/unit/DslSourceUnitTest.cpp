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
#include "DslSourceBintr.h"


using namespace DSL;

SCENARIO( "A new CsiSourceBintr is created correctly",  "[CsiSourceBintr]" )
{
    GIVEN( "A name for a new CsiSourceBintr" ) 
    {
        uint width(1280);
        uint height(720);
        uint fpsN(30);
        uint fpsD(1);
        std::string sourceName("test-csi-source");

        WHEN( "The CsiSourceBintr is created " )
        {
        
            DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
                sourceName.c_str(), width, height, fpsN, fpsD);

            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->m_gpuId == 0 );
                REQUIRE( pSourceBintr->m_nvbufMemoryType == 0 );
                REQUIRE( pSourceBintr->GetGstObject() != NULL );
                REQUIRE( pSourceBintr->GetSourceId() == -1 );
                REQUIRE( pSourceBintr->IsInUse() == false );
                REQUIRE( pSourceBintr->IsLive() == true );
                
                uint retWidth, retHeight, retFpsN, retFpsD;
                pSourceBintr->GetDimensions(&retWidth, &retHeight);
                pSourceBintr->GetFrameRate(&retFpsN, &retFpsD);
                REQUIRE( width == retWidth );
                REQUIRE( height == retHeight );
                REQUIRE( fpsN == retFpsN );
                REQUIRE( fpsD == retFpsD );
            }
        }
    }
}

SCENARIO( "Set Sensor Id updates SourceBintr correctly",  "[CsiSourceBintr]" )
{
    GIVEN( "A new CsiSourceBintr in memory" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);
        std::string sourceName("test-csi-source");
        int sensorId(1);

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        WHEN( "The Sensor Id is set " )
        {
            pSourceBintr->SetSourceId(sensorId);

            THEN( "The returned Sensor Id is correct" )
            {
                REQUIRE( pSourceBintr->GetSourceId() == sensorId );
            }
        }
    }
}

SCENARIO( "A CsiSourceBintr can LinkAll child Elementrs correctly",  "[CsiSourceBintr]" )
{
    GIVEN( "A new CsiSourceBintr in memory" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);
        std::string sourceName("test-csi-source");
        int sensorId = 1;

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        WHEN( "The CsiSourceBintr is called to LinkAll" )
        {
            REQUIRE( pSourceBintr->LinkAll() == true );

            THEN( "The CsiSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A CsiSourceBintr can UnlinkAll all child Elementrs correctly",  "[CsiSourceBintr]" )
{
    GIVEN( "A new, linked CsiSourceBintr " ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);
        std::string sourceName("test-csi-source");
        int sensorId = 1;

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        pSourceBintr->LinkAll();
        REQUIRE( pSourceBintr->IsLinked() == true );

        WHEN( "The CsiSourceBintr is called to UnlinkAll" )
        {
            pSourceBintr->UnlinkAll();

            THEN( "The CsiSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new UsbSourceBintr is created correctly",  "[UsbSourceBintr]" )
{
    GIVEN( "A name for a new UsbSourceBintr" ) 
    {
        uint width(1280);
        uint height(720);
        uint fpsN(30);
        uint fpsD(1);
        std::string sourceName("usb-source");

        WHEN( "The UsbSourceBintr is created " )
        {
        
            DSL_USB_SOURCE_PTR pSourceBintr = DSL_USB_SOURCE_NEW(
                sourceName.c_str(), width, height, fpsN, fpsD);

            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->GetGstObject() != NULL );
                REQUIRE( pSourceBintr->GetSourceId() == -1 );
                REQUIRE( pSourceBintr->IsInUse() == false );
                REQUIRE( pSourceBintr->IsLive() == true );
                
                uint retWidth, retHeight, retFpsN, retFpsD;
                pSourceBintr->GetDimensions(&retWidth, &retHeight);
                pSourceBintr->GetFrameRate(&retFpsN, &retFpsD);
                REQUIRE( width == retWidth );
                REQUIRE( height == retHeight );
                REQUIRE( fpsN == retFpsN );
                REQUIRE( fpsD == retFpsD );
            }
        }
    }
}

SCENARIO( "A UsbSourceBintr can LinkAll child Elementrs correctly",  "[UsbSourceBintr]" )
{
    GIVEN( "A new UsbSourceBintr in memory" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);
        std::string sourceName("usb-source");

        DSL_USB_SOURCE_PTR pSourceBintr = DSL_USB_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        WHEN( "The UsbSourceBintr is called to LinkAll" )
        {
            REQUIRE( pSourceBintr->LinkAll() == true );

            THEN( "The UsbSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A UsbSourceBintr can UnlinkAll all child Elementrs correctly",  "[UsbSourceBintr]" )
{
    GIVEN( "A new, linked UsbSourceBintr " ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);
        std::string sourceName("usb-source");

        DSL_USB_SOURCE_PTR pSourceBintr = DSL_USB_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        pSourceBintr->LinkAll();
        REQUIRE( pSourceBintr->IsLinked() == true );

        WHEN( "The UsbSourceBintr is called to UnlinkAll" )
        {
            pSourceBintr->UnlinkAll();

            THEN( "The UsbSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new UriSourceBintr is created correctly",  "[UriSourceBintr]" )
{
    GIVEN( "A name for a new UriSourceBintr" ) 
    {
        std::string sourceName = "test-uri-source";
        std::string uri = "./test/streams/sample_1080p_h264.mp4";
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);
        
        char absolutePath[PATH_MAX+1];
        std::string fullUriPath = realpath(uri.c_str(), absolutePath);
        fullUriPath.insert(0, "file:");

        WHEN( "The UriSourceBintr is created " )
        {
            DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
                sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->m_gpuId == 0 );
                REQUIRE( pSourceBintr->m_nvbufMemoryType == 0 );
                REQUIRE( pSourceBintr->GetGstObject() != NULL );
                REQUIRE( pSourceBintr->GetSourceId() == -1 );
                REQUIRE( pSourceBintr->IsInUse() == false );
                
                // Must reflect use of file stream
                REQUIRE( pSourceBintr->IsLive() == false );
                
                std::string returnedUri = pSourceBintr->GetUri();
                REQUIRE( returnedUri == fullUriPath );
                
                uint retWidth, retHeight, retFpsN, retFpsD;
                pSourceBintr->GetDimensions(&retWidth, &retHeight);
                pSourceBintr->GetFrameRate(&retFpsN, &retFpsD);
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );
                REQUIRE( retFpsN == 0 );
                REQUIRE( retFpsD == 0 );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can LinkAll child Elementrs correctly",  "[UriSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr in memory" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        WHEN( "The UriSourceBintr is called to LinkAll" )
        {
            REQUIRE( pSourceBintr->LinkAll() == true );

            THEN( "The UriSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can UnlinkAll all child Elementrs correctly",  "[UriSourceBintr]" )
{
    GIVEN( "A new, linked UriSourceBintr " ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        REQUIRE( pSourceBintr->LinkAll() == true );
        REQUIRE( pSourceBintr->IsLinked() == true );

        WHEN( "The UriSourceBintr is called to UnlinkAll" )
        {
            pSourceBintr->UnlinkAll();

            THEN( "The UriSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can Add a Child SinkBintr",  "[UriSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr and OverlaySinkBintr in memory" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "The OverlaySinkBintr is added to UriSourceBintr" )
        {
            REQUIRE( pSourceBintr->AddSinkBintr(pSinkBintr) == true );

            THEN( "The OverlaySinkBintr is found to be a child of the UriSourceBintr" )
            {
                REQUIRE( pSourceBintr->IsSinkBintrChild(pSinkBintr) == true );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can Remove a Child SinkBintr",  "[UriSourceBintr]" )
{
    GIVEN( "A a UriSourceBintr with a child OverlaySinkBintr" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSourceBintr->AddSinkBintr(pSinkBintr) == true );
        REQUIRE( pSourceBintr->IsSinkBintrChild(pSinkBintr) == true );

        WHEN( "The OverlaySinkBintr is removed from the UriSourceBintr" )
        {
            REQUIRE( pSourceBintr->RemoveSinkBintr(pSinkBintr) == true );
            THEN( "The OverlaySinkBintr is found to be a child of the UriSourceBintr" )
            {
                REQUIRE( pSourceBintr->IsSinkBintrChild(pSinkBintr) == false );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr with a child SinkBintr can LinkAll correctly",  "[UriSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr in memory" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);
        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSourceBintr->AddSinkBintr(pSinkBintr) == true );

        WHEN( "The UriSourceBintr is called to LinkAll" )
        {
            REQUIRE( pSourceBintr->LinkAll() == true );

            THEN( "The UriSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr with a child SinkBintr can UnlinkAll correctly",  "[UriSourceBintr]" )
{
    GIVEN( "A new, linked UriSourceBintr with a child SinkBintr" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string sinkName("overlay-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);
        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pSourceBintr->AddSinkBintr(pSinkBintr) == true );
        REQUIRE( pSourceBintr->LinkAll() == true );
        REQUIRE( pSourceBintr->IsLinked() == true );

        WHEN( "The UriSourceBintr is called to UnlinkAll" )
        {
            pSourceBintr->UnlinkAll();

            THEN( "The UriSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can Add a Child DewarperBintr",  "[DecodeSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr and DewarperBintr in memory" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

        WHEN( "The DewarperBintr is added to UriSourceBintr" )
        {
            REQUIRE( pSourceBintr->AddDewarperBintr(pDewarperBintr) == true );

            THEN( "The UriSourceBintr correctly returns that it has a dewarper" )
            {
                REQUIRE( pSourceBintr->HasDewarperBintr() == true );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can Remove a Child DewarperBintr",  "[DecodeSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr with a child DewarperBintr" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

        REQUIRE( pSourceBintr->AddDewarperBintr(pDewarperBintr) == true );

        WHEN( "The DewarperBintr is removed from the UriSourceBintr" )
        {
            REQUIRE( pSourceBintr->RemoveDewarperBintr() == true );
            
            THEN( "The UriSourceBintr correctly returns that it does not have a dewarper" )
            {
                REQUIRE( pSourceBintr->HasDewarperBintr() == false );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can ensure a single Child DewarperBintr",  "[DecodeSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr with a child DewarperBintr" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string dewarperName1("dewarper1");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        std::string dewarperName2("dewarper2");

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_DEWARPER_PTR pDewarperBintr1 = 
            DSL_DEWARPER_NEW(dewarperName1.c_str(), defConfigFile.c_str());

        DSL_DEWARPER_PTR pDewarperBintr2 = 
            DSL_DEWARPER_NEW(dewarperName2.c_str(), defConfigFile.c_str());

        REQUIRE( pSourceBintr->AddDewarperBintr(pDewarperBintr1) == true );

        WHEN( "Adding a second DewarperBintr should fail" )
        {
            REQUIRE( pSourceBintr->AddDewarperBintr(pDewarperBintr2) == false );
            
            THEN( "The UriSourceBintr correctly returns that it does not have a dewarper" )
            {
                REQUIRE( pSourceBintr->HasDewarperBintr() == true );
                REQUIRE( pSourceBintr->RemoveDewarperBintr() == true );
                // removing a second time must fail
                REQUIRE( pSourceBintr->RemoveDewarperBintr() == false );
                REQUIRE( pSourceBintr->HasDewarperBintr() == false );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr with a child DewarperBintr can LinkAll child Elementrs correctly",  "[DecodeSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr with a child DewarperBintr" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

        REQUIRE( pSourceBintr->AddDewarperBintr(pDewarperBintr) == true );

        WHEN( "The UriSourceBintr is called to LinkAll" )
        {
            REQUIRE( pSourceBintr->LinkAll() == true );

            THEN( "The UriSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked UriSourceBintr with a child DewarperBintr can UnlinkAll child Elementrs correctly",  "[DecodeSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr with a child DewarperBintr" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

        REQUIRE( pSourceBintr->AddDewarperBintr(pDewarperBintr) == true );

        REQUIRE( pSourceBintr->LinkAll() == true );

        WHEN( "The UriSourceBintr is called to LinkAll" )
        {
            pSourceBintr->UnlinkAll();
            
            THEN( "The UriSourceBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSourceBintr->IsLinked() == false );
            }
        }
    }
}


SCENARIO( "A new RtspSourceBintr is created correctly",  "[UriSourceBintr]" )
{
    GIVEN( "A name for a new RtspSourceBintr" ) 
    {
        std::string sourceName("test-rtps-source");
        std::string uri("https://hddn01.skylinewebcams.com/live.m3u8?a=e8inqgf08vq4rp43gvmkj9ilv0");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);

        WHEN( "The RtspSourceBintr is created " )
        {
            DSL_RTSP_SOURCE_PTR pSourceBintr = DSL_RTSP_SOURCE_NEW(
                sourceName.c_str(), uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval);

            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->m_gpuId == 0 );
                REQUIRE( pSourceBintr->m_nvbufMemoryType == 0 );
                REQUIRE( pSourceBintr->GetGstObject() != NULL );
                REQUIRE( pSourceBintr->GetSourceId() == -1 );
                REQUIRE( pSourceBintr->IsInUse() == false );
                
                // Must reflect use of file stream
                REQUIRE( pSourceBintr->IsLive() == true );
                
                std::string returnedUri = pSourceBintr->GetUri();
                REQUIRE( returnedUri == uri );
                
                uint retWidth, retHeight, retFpsN, retFpsD;
                pSourceBintr->GetDimensions(&retWidth, &retHeight);
                pSourceBintr->GetFrameRate(&retFpsN, &retFpsD);
                REQUIRE( retWidth == 0 );
                REQUIRE( retHeight == 0 );
                REQUIRE( retFpsN == 0 );
                REQUIRE( retFpsD == 0 );
            }
        }
    }
}

