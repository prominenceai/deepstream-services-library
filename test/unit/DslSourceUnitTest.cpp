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
#include "DslSinkBintr.h"
#include "DslSourceBintr.h"
#include "DslPipelineSourcesBintr.h"

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
                REQUIRE( pSourceBintr->GetId() == -1 );
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
            pSourceBintr->SetId(sensorId);

            THEN( "The returned Sensor Id is correct" )
            {
                REQUIRE( pSourceBintr->GetId() == sensorId );
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
                REQUIRE( pSourceBintr->GetId() == -1 );
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

SCENARIO( "A UsbSourceBintr can Get and Set its GPU ID",  "[UsbSourceBintr]" )
{
    GIVEN( "A new UsbSourceBintr in memory" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(30);
        uint fps_d(1);
        std::string sourceName("usb-source");

        DSL_USB_SOURCE_PTR pUsbSourceBintr = DSL_USB_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        uint GPUID0(0);
        uint GPUID1(1);

        REQUIRE( pUsbSourceBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The UsbSourceBintr's  GPU ID is set" )
        {
            REQUIRE( pUsbSourceBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pUsbSourceBintr->GetGpuId() == GPUID1 );
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
                REQUIRE( pSourceBintr->GetId() == -1 );
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

SCENARIO( "A UriSourceBintr can Set and Get its URI",  "[UriSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr in memory" ) 
    {
        std::string sourceName = "test-uri-source";
        std::string uri = "./test/streams/sample_1080p_h264.mp4";
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);
        
        char absolutePath[PATH_MAX+1];
        std::string fullUriPath = realpath(uri.c_str(), absolutePath);
        fullUriPath.insert(0, "file:");

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        std::string returnedUri = pSourceBintr->GetUri();
        REQUIRE( returnedUri == fullUriPath );

        WHEN( "The UriSourceBintr's URI is updated " )
        {
            // TODO: should use a new UIR here
            REQUIRE( pSourceBintr->SetUri(uri.c_str()) == true );
            THEN( "The correct URI is returned on get" )
            {
                std::string returnedUri = pSourceBintr->GetUri();
                REQUIRE( returnedUri == fullUriPath );
            }
        }
    }
}

SCENARIO( "A UriSourceBintr can Get and Set its GPU ID",  "[UriSourceBintr]" )
{
    GIVEN( "A new UriSourceBintr in memory" ) 
    {
        std::string sourceName("test-file-source");
        std::string uri("./test/streams/sample_1080p_h264.mp4");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);
        
        DSL_URI_SOURCE_PTR pUriSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), uri.c_str(), false, cudadecMemType, intrDecode, dropFrameInterval);

        uint GPUID0(0);
        uint GPUID1(1);

        REQUIRE( pUriSourceBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The UriSourceBintr's  GPU ID is set" )
        {
            REQUIRE( pUriSourceBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pUriSourceBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}

SCENARIO( "A new RtspSourceBintr is created correctly",  "[RtspSourceBinter]" )
{
    GIVEN( "A name for a new RtspSourceBintr" ) 
    {
        std::string sourceName("rtsp-source");
        std::string uri("rtsp://208.72.70.171:80/mjpg/video.mjpg");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);
        uint latency(100);
        uint timeout(20);

        WHEN( "The RtspSourceBintr is created " )
        {
            DSL_RTSP_SOURCE_PTR pSourceBintr = DSL_RTSP_SOURCE_NEW(sourceName.c_str(), 
                uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval, latency, timeout);

            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->m_gpuId == 0 );
                REQUIRE( pSourceBintr->m_nvbufMemoryType == 0 );
                REQUIRE( pSourceBintr->GetGstObject() != NULL );
                REQUIRE( pSourceBintr->GetId() == -1 );
                REQUIRE( pSourceBintr->IsInUse() == false );
                REQUIRE( pSourceBintr->GetBufferTimeout() == timeout );
                REQUIRE( pSourceBintr->GetCurrentState() == GST_STATE_NULL );
                
                time_t last(123);
                uint count(456);
                boolean isInReset(true);
                uint retries(123);
                pSourceBintr->GetReconnectionStats(&last, &count, &isInReset, &retries);
                REQUIRE( last == 0 );
                REQUIRE( count == 0 );
                REQUIRE( isInReset == false );
                REQUIRE( retries == 0 );
                
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

SCENARIO( "A new RtspSourceBintr's attributes can be set/get ",  "[RtspSourceBinter]" )
{
    GIVEN( "A new RtspSourceBintr with a timeout" ) 
    {
        std::string sourceName("rtsp-source");
        std::string uri("rtsp://208.72.70.171:80/mjpg/video.mjpg");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);
        uint latency(100);
        uint timeout(20);

        DSL_RTSP_SOURCE_PTR pSourceBintr = DSL_RTSP_SOURCE_NEW(sourceName.c_str(), 
            uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval, latency, timeout);

        WHEN( "The RtspSourceBintr's timeout set " )
        {
            uint newTimeout(0);
            pSourceBintr->SetBufferTimeout(newTimeout);

            THEN( "The correct value is returned on get" )
            {
                REQUIRE( pSourceBintr->GetBufferTimeout() == newTimeout );
            }
        }
        WHEN( "The RtspSourceBintr's reconnect stats are set " )
        {
            time_t newLast(123), last(0);
            uint newCount(0), count(0);
            boolean newIsInReset(true), isInReset(false);
            uint newRetries(123), retries(0);
            pSourceBintr->_setReconnectionStats(newLast, newCount, newIsInReset, newRetries);

            THEN( "The correct value is returned on get" )
            {
                pSourceBintr->GetReconnectionStats(&last, &count, &isInReset, &retries);
                REQUIRE( last == newLast );
                REQUIRE( count == newCount );
                REQUIRE( isInReset == newIsInReset );
                REQUIRE( retries == newRetries );
            }
        }
    }
}

static void source_state_change_listener_cb1(uint prev_state, uint curr_state, void* user_data)
{
    std::cout << "Source state change lister 1 called with prev_state = " 
        << prev_state << " current_state = " << curr_state << "\n";
        *(int*)user_data = 111;
}

static void source_state_change_listener_cb2(uint prev_state, uint curr_state, void* user_data)
{
    std::cout << "Source state change lister 2 called with prev_state = " 
        << prev_state << " current_state = " << curr_state << "\n";
        *(int*)user_data = 222;
}

SCENARIO( "An RtspSourceBintr can add and remove State Change Listeners",  "[RtspSourceBinter]" )
{
    GIVEN( "A new RtspSourceBintr with a timeout" ) 
    {
        std::string sourceName("rtsp-source");
        std::string uri("rtsp://208.72.70.171:80/mjpg/video.mjpg");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);
        uint latency(100);
        uint timeout(20);

        DSL_RTSP_SOURCE_PTR pRtspSourceBintr = DSL_RTSP_SOURCE_NEW(sourceName.c_str(), 
            uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval, latency, timeout);
        
        WHEN( "Client Listeners are added" )
        {
            REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb1, NULL) == true );
            REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb2, NULL) == true );

            THEN( "Adding them a second time must fail" )
            {
                REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb1, NULL) == false );
                REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb2, NULL) == false );
            }
        }
        WHEN( "Client Listeners are added" )
        {
            REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb1, NULL) == true );
            REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb2, NULL) == true );

            THEN( "They can be successfully removed" )
            {
                REQUIRE( pRtspSourceBintr->RemoveStateChangeListener(source_state_change_listener_cb1) == true );
                REQUIRE( pRtspSourceBintr->RemoveStateChangeListener(source_state_change_listener_cb2) == true );
                
                // Calling a second time must fail
                REQUIRE( pRtspSourceBintr->RemoveStateChangeListener(source_state_change_listener_cb1) == false );
                REQUIRE( pRtspSourceBintr->RemoveStateChangeListener(source_state_change_listener_cb2) == false );
            }
        }
    }
}
            
SCENARIO( "An RtspSourceBintr calls all State Change Listeners on change of state", "[RtspSourceBinter]" )
{
    GIVEN( "A new RtspSourceBintr with a timeout" ) 
    {
        std::string sourceName("rtsp-source");
        std::string uri("rtsp://208.72.70.171:80/mjpg/video.mjpg");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);
        uint latency(100);
        uint timeout(20);
        uint userData1(0), userData2(0);

        DSL_RTSP_SOURCE_PTR pRtspSourceBintr = DSL_RTSP_SOURCE_NEW(sourceName.c_str(), 
            uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval, latency, timeout);

        REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb1, &userData1) == true );
        REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb2, &userData2) == true );
        
        WHEN( "The current state is changed" )
        {
            pRtspSourceBintr->SetCurrentState(GST_STATE_READY);

            THEN( "All client listeners are called on state change" )
            {
                REQUIRE( pRtspSourceBintr->GetCurrentState() == GST_STATE_READY );
                
                // simulate timer callback
                REQUIRE( pRtspSourceBintr->NotifyClientListeners() == FALSE );
                // Callbacks will change user data if called
                REQUIRE( userData1 == 111 );
                REQUIRE( userData2 == 222 );
            }
        }
    }
}

SCENARIO( "An RtspSourceBintr's Stream Management callback behaves correctly", "[RtspSourceBinter]" )
{
    GIVEN( "A new RtspSourceBintr with a timeout" ) 
    {
        std::string sourceName("rtsp-source");
        std::string uri("rtsp://208.72.70.171:80/mjpg/video.mjpg");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(false);
        uint dropFrameInterval(0);
        uint latency(100);
        uint timeout(20);
        uint userData1(0), userData2(0);

        DSL_RTSP_SOURCE_PTR pRtspSourceBintr = DSL_RTSP_SOURCE_NEW(sourceName.c_str(), 
            uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval, latency, timeout);

        REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb1, &userData1) == true );
        REQUIRE( pRtspSourceBintr->AddStateChangeListener(source_state_change_listener_cb2, &userData2) == true );

        std::string pipelineSourcesName = "pipeline-sources";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());
            
        DSL_SOURCE_PTR pSourceBintr = std::dynamic_pointer_cast<SourceBintr>(pRtspSourceBintr);
            
        // Source needs a parent to test reconnect - required for source to call "gst_element_sync_state_with_parent"
        pPipelineSourcesBintr->AddChild(pSourceBintr);
        
        WHEN( "The Source is in reset" )
        {
            pRtspSourceBintr->_setReconnectionStats(0, 0, true, 1);

            THEN( "The Stream Management callback returns true immediately" )
            {
                // Note: this test requires (currently) additional manual/visual confirmation of console log output
                REQUIRE( pRtspSourceBintr->StreamManager() == true );
            }
        }
        WHEN( "The Source is NOT in reset and lastBufferTime is uninitialized" )
        {
            pRtspSourceBintr->_setReconnectionStats(0, 0, false, 0);

            THEN( "The Stream Management callback returns true immediately" )
            {
                // Note: this test requires (currently) additional manual/visual confirmation of console log output
                REQUIRE( pRtspSourceBintr->StreamManager() == true );
            }
        }
        WHEN( "The Source is NOT in reset and lastBufferTime = current time" )
        {
            pRtspSourceBintr->_setReconnectionStats(0, 0, false, 0);
            // get the current time and update the Source buffer timestamp
            timeval currentTime{0};
            gettimeofday(&currentTime, NULL);
            pRtspSourceBintr->_getTimestampPph()->SetTime(currentTime);

            THEN( "The Stream Management callback returns true immediately" )
            {
                // Note: this test requires (currently) additional manual/visual confirmation of console log output
                REQUIRE( pRtspSourceBintr->StreamManager() == true );
            }
        }
        WHEN( "The Source is NOT in reset and currentTime-lastBufferTime > timeout" )
        {
            pRtspSourceBintr->_setReconnectionStats(0, 0, false, 0);
            pRtspSourceBintr->SetCurrentState(GST_STATE_PLAYING);
            // get the current time and update the Source buffer timestamp
            timeval currentTime{0};
            gettimeofday(&currentTime, NULL);
            currentTime.tv_sec -= timeout;
            pRtspSourceBintr->_getTimestampPph()->SetTime(currentTime);

            THEN( "The Stream Management callback Initiates a Reconnect Cycle" )
            {
                // Note: this test requires (currently) additional manual/visual confirmation of console log output
                REQUIRE( pRtspSourceBintr->StreamManager() == true );

                // simulate timer callback
                REQUIRE( pRtspSourceBintr->NotifyClientListeners() == FALSE );

                // simulate a reconnection timer - which should fail - unable to sync to parent
                REQUIRE( pRtspSourceBintr->ReconnectionManager() == false );
            }
        }
    }
}

SCENARIO( "A RtspSourceBintr can Get and Set its GPU ID",  "[RtspSourceBintr]" )
{
    GIVEN( "A new RtspSourceBintr in memory" ) 
    {
        std::string sourceName("test-rtsp-source");
        std::string uri("rtsp://hddn01.skylinewebcams.com/live.m3u8?a=e8inqgf08vq4rp43gvmkj9ilv0");
        uint cudadecMemType(DSL_CUDADEC_MEMTYPE_DEVICE);
        uint intrDecode(true);
        uint dropFrameInterval(2);
        uint latency(100);
        uint timeout(20);
        
        DSL_RTSP_SOURCE_PTR pRtspSourceBintr = DSL_RTSP_SOURCE_NEW(sourceName.c_str(),
            uri.c_str(), DSL_RTP_ALL, cudadecMemType, intrDecode, dropFrameInterval, latency, timeout);

        uint GPUID0(0);
        uint GPUID1(1);

        REQUIRE( pRtspSourceBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The RtspSourceBintr's  GPU ID is set" )
        {
            REQUIRE( pRtspSourceBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pRtspSourceBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}

