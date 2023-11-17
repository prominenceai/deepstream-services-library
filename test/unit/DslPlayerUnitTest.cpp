/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslPlayerBintr.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

static std::string mp4FilePath1("/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static std::string mp4FilePath2("/opt/nvidia/deepstream/deepstream/samples/streams/yoga.mp4");

static std::string jpgFilePath1("/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");
static std::string jpgFilePath2("/opt/nvidia/deepstream/deepstream/samples/streams/yoga.jpg");

static const std::string sourceName("source");

static const std::string sinkName("sink");

static const std::string playerName("player");

static const uint displayId(0);
static const uint depth(0);
static uint offsetX(100);
static uint offsetY(140);
static uint sinkW(1280);
static uint sinkH(720);


using namespace DSL;

SCENARIO( "A New PlayerBintr is created correctly", "[PlayerBintr]" )
{
    GIVEN( "A new URI Source and Window Sink" ) 
    {

        
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(mp4FilePath1.c_str(), absolutePath);
        fullFilePath.insert(0, "file:");

        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), mp4FilePath1.c_str(), false);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "The new PlayerBintr is created" )
        {
            DSL_PLAYER_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

            THEN( "All member variables are setup correctly" )
            {
                GstState state;
                REQUIRE( pPlayerBintr->GetName() == playerName );
                pPlayerBintr->GetState(state, 0);
                REQUIRE( state == DSL_STATE_NULL );
            }
        }
    }
}


SCENARIO( "A New PlayerBintr can Link its Child Bintrs correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PlayerBintr with URI Source and Overlay Sink" ) 
    {
        
        char absolutePath[PATH_MAX+1];

        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), mp4FilePath1.c_str(), false);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

        WHEN( "The new PlayerBintr is Linked" )
        {
            REQUIRE( pPlayerBintr->LinkAll() == true );
            
            THEN( "The PlayerBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pPlayerBintr->IsLinked() == true );

            }
        }
    }
}

SCENARIO( "A New PlayerBintr can Unlink its Child Bintrs correctly", "[PlayerBintr]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        char absolutePath[PATH_MAX+1];

        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), mp4FilePath1.c_str(), false);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

        REQUIRE( pPlayerBintr->LinkAll() == true );

        WHEN( "The new PlayerBintr is Unlinked" )
        {
            pPlayerBintr->UnlinkAll();
            
            THEN( "The PlayerBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pPlayerBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A New PlayerBintr with a File Source and Overlay Sink can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
                sourceName.c_str(), mp4FilePath1.c_str(), false);

            DSL_3D_SINK_PTR pSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(),
                    offsetX, offsetY, sinkH, sinkH);

            DSL_PLAYER_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

            WHEN( "The new PlayerBintr is set to a state of PLAYING" )
            {
                REQUIRE( pPlayerBintr->Play() == true);
                
                THEN( "The PlayerBintr can be set back to a state of NULL" )
                {
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A New PlayerBintr with a File Source and Window Sink can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {

        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), mp4FilePath1.c_str(), false);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

        WHEN( "The new PlayerBintr is set to a state of PLAYING" )
        {
            REQUIRE( pPlayerBintr->Play() == true);
            
            THEN( "The PlayerBintr can be set back to a state of NULL" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A New PlayerBintr with ImageStreamSourceBintr and OverlaySinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PlayerBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            DSL_IMAGE_STREAM_SOURCE_PTR pSourceBintr = DSL_IMAGE_STREAM_SOURCE_NEW(
                sourceName.c_str(), jpgFilePath1.c_str(), false, 1, 1, 0);

            // use the image size for the Window sink dimensions
            pSourceBintr->GetDimensions(&sinkW, &sinkH);
            
            DSL_3D_SINK_PTR pSinkBintr = 
                DSL_3D_SINK_NEW(sinkName.c_str(), 
                    offsetX, offsetY, sinkW, sinkH);

            DSL_PLAYER_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

            WHEN( "The new PlayerBintr is set to a state of PLAYING" )
            {
                REQUIRE( pPlayerBintr->Play() == true);
                
                THEN( "The PlayerBintr can be set back to a state of NULL" )
                {
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A New PlayerBintr with ImageStreamSourceBintr and WindowSinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PipelineBintr" ) 
    {
        
        DSL_IMAGE_STREAM_SOURCE_PTR pSourceBintr = DSL_IMAGE_STREAM_SOURCE_NEW(
            sourceName.c_str(), jpgFilePath1.c_str(), false, 1, 1, 0);

        // use the image size for the Window sink dimensions
        pSourceBintr->GetDimensions(&sinkW, &sinkH);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pSinkBintr);

        WHEN( "The new PlayerBintr is set to a state of PLAYING" )
        {
            REQUIRE( pPlayerBintr->Play() == true);
            
            THEN( "The PlayerBintr can be set back to a state of NULL" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A New ImageRenderPlayerBintr with OverlaySinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PipelineBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            uint offsetX(400);
            uint offsetY(200);
            uint zoom(50);
            uint timeout(0);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                    jpgFilePath1.c_str(), DSL_RENDER_TYPE_3D, offsetX, offsetY, zoom, timeout);

            WHEN( "The new PlayerBintr is set to a state of PLAYING" )
            {
                REQUIRE( pPlayerBintr->Play() == true);
                
                THEN( "The PlayerBintr can be set back to a state of NULL" )
                {
                    // Stop and play again
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Play() == true);
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A New ImageRenderPlayerBintr with WindowSinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PipelineBintr" ) 
    {
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        uint timeout(0);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                jpgFilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

        WHEN( "The new PlayerBintr is set to a state of PLAYING" )
        {
            REQUIRE( pPlayerBintr->Play() == true);
            
            THEN( "The PlayerBintr can be set back to a state of NULL" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Play() == true);
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A New VideoRenderPlayerBintr with OverlaySinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            uint offsetX(400);
            uint offsetY(200);
            uint zoom(50);
            bool repeatEnabled(0);

            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                    mp4FilePath1.c_str(), DSL_RENDER_TYPE_3D, offsetX, offsetY, zoom, repeatEnabled);

            WHEN( "The new PlayerBintr is set to a state of PLAYING" )
            {
                REQUIRE( pPlayerBintr->Play() == true);
                
                THEN( "The PlayerBintr can be set back to a state of NULL" )
                {
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Play() == true);
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A New VideoRenderPlayerBintr with WindowSinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(50);
        int timeout(0);

        DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                mp4FilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

        WHEN( "The new PlayerBintr is set to a state of PLAYING" )
        {
            REQUIRE( pPlayerBintr->Play() == true);
            
            THEN( "The PlayerBintr can be set back to a state of NULL" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A VideoRenderPlayerBintr can Set/Get its File Path correctly", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(50);
        bool repeatEnabled(false);

        char absolutePath[PATH_MAX+1];
        std::string fullFilePath1 = realpath(mp4FilePath1.c_str(), absolutePath);
        fullFilePath1.insert(0, "file:");
        std::string fullFilePath2 = realpath(mp4FilePath2.c_str(), absolutePath);
        fullFilePath2.insert(0, "file:");

        DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                mp4FilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, repeatEnabled);
                
        std::string returnedFilePath1 = pPlayerBintr->GetFilePath();
        REQUIRE( returnedFilePath1 == fullFilePath1 );

        REQUIRE( pPlayerBintr->Play() == true);
        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
        REQUIRE( pPlayerBintr->Stop() == true );

        WHEN( "A new File Path is Set" )
        {
            REQUIRE( pPlayerBintr->SetFilePath(mp4FilePath2.c_str()) == true);
            
            THEN( "The same File Path is return on Get" )
            {
                std::string returnedFilePath2 = pPlayerBintr->GetFilePath();
                REQUIRE( returnedFilePath2 == fullFilePath2 );
                REQUIRE( pPlayerBintr->Play() == true);
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A ImageRenderPlayerBintr can Set/Get its File Path correctly", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(50);
        int timeout(0);

        char absolutePath[PATH_MAX+1];
        std::string fullFilePath1 = realpath(jpgFilePath1.c_str(), absolutePath);
        std::string fullFilePath2 = realpath(jpgFilePath2.c_str(), absolutePath);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                jpgFilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);
                
        std::string returnedFilePath1 = pPlayerBintr->GetFilePath();
        REQUIRE( returnedFilePath1 == fullFilePath1 );

        REQUIRE( pPlayerBintr->Play() == true);
        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
        REQUIRE( pPlayerBintr->Stop() == true );

        WHEN( "A new File Path is Set" )
        {
            REQUIRE( pPlayerBintr->SetFilePath(jpgFilePath2.c_str()) == true);
            
            THEN( "The same File Path is return on Get" )
            {
                std::string returnedFilePath2 = pPlayerBintr->GetFilePath();
                REQUIRE( returnedFilePath2 == fullFilePath2 );
                REQUIRE( pPlayerBintr->Play() == true);
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A ImageRenderPlayerBintr with a WindowSinkBintr can Set/Get its Zoom", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(50);
        int timeout(0);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                jpgFilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

        REQUIRE( pPlayerBintr->Play() == true);
        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "The Zoom Setting is updated" )
        {
            uint newZoom(100);
            REQUIRE( pPlayerBintr->SetZoom(newZoom) == true);
            
            THEN( "The correct Zoom setting is return on Get" )
            {
                REQUIRE( pPlayerBintr->GetZoom() == newZoom );
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A ImageRenderPlayerBintr with a OverlaySinkBintr can Set/Get its Zoom", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            uint offsetX(0);
            uint offsetY(0);
            uint zoom(50);
            int timeout(0);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                    jpgFilePath1.c_str(), DSL_RENDER_TYPE_3D, offsetX, offsetY, zoom, timeout);

            REQUIRE( pPlayerBintr->Play() == true);
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            WHEN( "The Zoom Setting is updated" )
            {
                uint newZoom(100);
                REQUIRE( pPlayerBintr->SetZoom(newZoom) == true);
                
                THEN( "The correct Zoom setting is return on Get" )
                {
                    REQUIRE( pPlayerBintr->GetZoom() == newZoom );
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A VideoRenderPlayerBintr with a WindowSinkBintr can Set/Get its Zoom", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(50);
        bool repeatEnabled(false);

        DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                mp4FilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, repeatEnabled);

        REQUIRE( pPlayerBintr->Play() == true);
        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "The Zoom Setting is updated" )
        {
            uint newZoom(100);
            REQUIRE( pPlayerBintr->SetZoom(newZoom) == true);
            
            THEN( "The correct Zoom setting is return on Get" )
            {
                REQUIRE( pPlayerBintr->GetZoom() == newZoom );
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A VideoRenderPlayerBintr with a OverlaySinkBintr can Set/Get its Zoom", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            uint offsetX(0);
            uint offsetY(0);
            uint zoom(50);
            bool repeatEnabled(false);

            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                    mp4FilePath1.c_str(), DSL_RENDER_TYPE_3D, offsetX, offsetY, zoom, repeatEnabled);

            REQUIRE( pPlayerBintr->Play() == true);
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            WHEN( "The Zoom Setting is updated" )
            {
                uint newZoom(100);
                REQUIRE( pPlayerBintr->SetZoom(newZoom) == true);
                
                THEN( "The correct Zoom setting is return on Get" )
                {
                    REQUIRE( pPlayerBintr->GetZoom() == newZoom );
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A ImageRenderPlayerBintr with a WindowSinkBintr can Set/Get its Offsets", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(100);
        int timeout(0);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                jpgFilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

        REQUIRE( pPlayerBintr->Play() == true);
        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "The offsets are updated" )
        {
            uint newOffsetX(123), newOffsetY(123);
            REQUIRE( pPlayerBintr->SetOffsets(newOffsetX, newOffsetY) == true);
            
            THEN( "The correct offsets are return on Get" )
            {
                uint retOffsetX(0), retOffsetY(0);
                pPlayerBintr->GetOffsets(&retOffsetX, &retOffsetY);
                REQUIRE( retOffsetX == newOffsetX );
                REQUIRE( retOffsetY == newOffsetY );
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A ImageRenderPlayerBintr with a OverlaySinkBintr can Set/Get its Offsets", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            uint offsetX(0);
            uint offsetY(0);
            uint zoom(100);
            int timeout(0);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                    jpgFilePath1.c_str(), DSL_RENDER_TYPE_3D, offsetX, offsetY, zoom, timeout);

            REQUIRE( pPlayerBintr->Play() == true);
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            WHEN( "The offsets are updated" )
            {
                uint newOffsetX(123), newOffsetY(123);
                REQUIRE( pPlayerBintr->SetOffsets(newOffsetX, newOffsetY) == true);
                
                THEN( "The correct offsets are return on Get" )
                {
                    uint retOffsetX(0), retOffsetY(0);
                    pPlayerBintr->GetOffsets(&retOffsetX, &retOffsetY);
                    REQUIRE( retOffsetX == newOffsetX );
                    REQUIRE( retOffsetY == newOffsetY );
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A ImageRenderPlayerBintr can play Queued files", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr with a queued file" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(100);
        int timeout(1);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                jpgFilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

        REQUIRE( pPlayerBintr->QueueFilePath(jpgFilePath2.c_str()) == true);

        WHEN( "An EOS event occurs" )
        {
            REQUIRE( pPlayerBintr->Play() == true);
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            // Simulate EOS event
            pPlayerBintr->HandleEos();
            
            THEN( "The Player is Stopped and the Queued file is Played" )
            {
                pPlayerBintr->HandleStopAndPlay();
                // Note: required visual confermation at this time..
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A New VideoRenderPlayerBintr cannot Play without a File Path", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        std::string filePath;
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        bool repeatEnabled(0);

        WHEN( "The PlayerBintr is created with an empty file path " )
        {
            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                    filePath.c_str(), DSL_RENDER_TYPE_WINDOW, 
                    offsetX, offsetY, zoom, repeatEnabled);
            
            THEN( "The PlayerBintr is unable to Play" )
            {
                REQUIRE( pPlayerBintr->Play() == false );
            }
        }
    }
}

SCENARIO( "A New ImageRenderPlayerBintr cannot Play without a File Path", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        std::string filePath;
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        uint timeout(0);

        WHEN( "The PlayerBintr is created with a empty file path " )
        {
            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                    filePath.c_str(), DSL_RENDER_TYPE_WINDOW, 
                    offsetX, offsetY, zoom, timeout);
            
            THEN( "The PlayerBintr is unable to Play" )
            {
                REQUIRE( pPlayerBintr->Play() == false );
            }
        }
    }
}

SCENARIO( "A New VideoRenderPlayerBintr can Play with an updated File Path", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        std::string filePath;
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        bool repeatEnabled(0);

        DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                filePath.c_str(), DSL_RENDER_TYPE_WINDOW, 
                offsetX, offsetY, zoom, repeatEnabled);

        REQUIRE( pPlayerBintr->Play() == false );

        WHEN( "The PlayerBintrs file path is updated" )
        {
            REQUIRE( pPlayerBintr->SetFilePath(mp4FilePath1.c_str()) == true );
            
            THEN( "The PlayerBintr is unable to Play" )
            {
                REQUIRE( pPlayerBintr->Play() == true );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A New ImageRenderPlayerBintr can Play with an updated File Path", "[PlayerBintr]" )
{
    GIVEN( "A new ImageRenderPlayerBintr" ) 
    {
        std::string filePath;
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        uint timeout(0);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                filePath.c_str(), DSL_RENDER_TYPE_WINDOW, 
                offsetX, offsetY, zoom, timeout);

        REQUIRE( pPlayerBintr->Play() == false );

        WHEN( "The PlayerBintrs file path is updated" )
        {
            REQUIRE( pPlayerBintr->SetFilePath(jpgFilePath1.c_str()) == true );
            
            THEN( "The PlayerBintr is unable to Play" )
            {
                REQUIRE( pPlayerBintr->Play() == true );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

SCENARIO( "A New ImageRenderPlayerBintr - Overlay Type - can Play after Reset", "[PlayerBintr]" )
{
    GIVEN( "An ImageRenderPlayerBintr in a playing stae" ) 
    {
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            uint offsetX(0);
            uint offsetY(0);
            uint zoom(100);
            int timeout(1);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
                DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                    jpgFilePath1.c_str(), DSL_RENDER_TYPE_3D, offsetX, offsetY, zoom, timeout);

            REQUIRE( pPlayerBintr->Play() == true );
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

            WHEN( "The PlayerBintr is stopped and reset" )
            {
                REQUIRE( pPlayerBintr->Stop() == true );
                REQUIRE( pPlayerBintr->Reset() == true );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                
                THEN( "The PlayerBintr is Able to Play and Stop again" )
                {
                    REQUIRE( pPlayerBintr->Play() == true );
                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( pPlayerBintr->Stop() == true );
                }
            }
        }
    }
}

SCENARIO( "A New ImageRenderPlayerBintr - Window Type - can Play after Reset", "[PlayerBintr]" )
{
    GIVEN( "An ImageRenderPlayerBintr in a playing stae" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(100);
        int timeout(1);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                jpgFilePath1.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

        REQUIRE( pPlayerBintr->Play() == true );
        std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

        WHEN( "The PlayerBintr is stopped and reset" )
        {
            REQUIRE( pPlayerBintr->Stop() == true );
            REQUIRE( pPlayerBintr->Reset() == true );
            std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
            
            THEN( "The PlayerBintr is Able to Play and Stop again" )
            {
                REQUIRE( pPlayerBintr->Play() == true );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPlayerBintr->Stop() == true );
            }
        }
    }
}

