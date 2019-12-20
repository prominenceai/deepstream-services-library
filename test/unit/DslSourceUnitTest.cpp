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

        WHEN( "The UriSourceBintr is created " )
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
            pSourceBintr->LinkAll();

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

        WHEN( "The FileSourceBintr is called to LinkAll" )
        {
            pSourceBintr->LinkAll();

            THEN( "The FileSourceBintr IsLinked state is updated correctly" )
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

        pSourceBintr->LinkAll();
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
