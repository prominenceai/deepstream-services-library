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
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslPlayerBintr.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

using namespace DSL;

SCENARIO( "A New PlayerBintr is created correctly", "[PlayerBintr]" )
{
    GIVEN( "A new URI Source and Overlay Sink" ) 
    {
        std::string playerName = "player";

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(filePath.c_str(), absolutePath);
        fullFilePath.insert(0, "file:");

        std::string overlaySinkName("overlay-sink");
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        
        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(overlaySinkName.c_str(), displayId, 
                depth, offsetX, offsetY, sinkW, sinkH);

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
        std::string playerName = "player";

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(filePath.c_str(), absolutePath);
        fullFilePath.insert(0, "file:");

        std::string overlaySinkName("overlay-sink");
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        
        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(overlaySinkName.c_str(), displayId, 
                depth, offsetX, offsetY, sinkW, sinkH);

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
        std::string playerName = "player";

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(filePath.c_str(), absolutePath);
        fullFilePath.insert(0, "file:");

        std::string overlaySinkName("overlay-sink");
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        
        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(overlaySinkName.c_str(), displayId, 
                depth, offsetX, offsetY, sinkW, sinkH);

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
        std::string playerName = "player";

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(filePath.c_str(), absolutePath);
        fullFilePath.insert(0, "file:");

        std::string overlaySinkName("overlay-sink");
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        
        std::string windowSinkName("window-sink");

        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false);

        DSL_OVERLAY_SINK_PTR pOverlaySinkBintr = 
            DSL_OVERLAY_SINK_NEW(overlaySinkName.c_str(), displayId, 
                depth, offsetX, offsetY, sinkW, sinkH);

        DSL_WINDOW_SINK_PTR pWindowSinkBintr = 
            DSL_WINDOW_SINK_NEW(windowSinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pOverlaySinkBintr);

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

SCENARIO( "A New PlayerBintr with a File Source and Window Sink can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        std::string playerName = "player";

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        
        char absolutePath[PATH_MAX+1];
        std::string fullFilePath = realpath(filePath.c_str(), absolutePath);
        fullFilePath.insert(0, "file:");

        std::string overlaySinkName("overlay-sink");
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(1280);
        uint sinkH(720);
        
        std::string windowSinkName("window-sink");

        DSL_FILE_SOURCE_PTR pSourceBintr = DSL_FILE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false);

        DSL_WINDOW_SINK_PTR pWindowSinkBintr = 
            DSL_WINDOW_SINK_NEW(windowSinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pWindowSinkBintr);

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

SCENARIO( "A New PlayerBintr with ImageSourceBintr and OverlaySinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PlayerBintr" ) 
    {
        std::string playerName = "player";

        std::string sourceName = "image-source";
        std::string filePath = "./test/streams/first-person-occurrence-438.jpeg";

        std::string overlaySinkName("overlay-sink");
        uint displayId(0);
        uint depth(0);
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(0);
        uint sinkH(0);


        DSL_IMAGE_SOURCE_PTR pSourceBintr = DSL_IMAGE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false, 1, 1, 0);

        // use the image size for the Window sink dimensions
        pSourceBintr->GetDimensions(&sinkW, &sinkH);
        
        DSL_OVERLAY_SINK_PTR pOverlaySinkBintr = 
            DSL_OVERLAY_SINK_NEW(overlaySinkName.c_str(), displayId, 
                depth, offsetX, offsetY, sinkW, sinkH);

        DSL_PLAYER_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_BINTR_NEW(playerName.c_str(), pSourceBintr, pOverlaySinkBintr);

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

SCENARIO( "A New PlayerBintr with ImageSourceBintr and WindowSinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PipelineBintr" ) 
    {
        std::string playerName = "player";

        std::string sourceName = "image-source";
        std::string filePath = "./test/streams/first-person-occurrence-438.jpeg";

        std::string windowSinkName("window-sink");
        uint offsetX(100);
        uint offsetY(140);
        uint sinkW(0);
        uint sinkH(0);

        DSL_IMAGE_SOURCE_PTR pSourceBintr = DSL_IMAGE_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false, 1, 1, 0);

        // use the image size for the Window sink dimensions
        pSourceBintr->GetDimensions(&sinkW, &sinkH);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(windowSinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

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
        std::string playerName("player");

        std::string filePath("./test/streams/first-person-occurrence-438.jpeg");
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        uint timeout(0);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                filePath.c_str(), DSL_RENDER_TYPE_OVERLAY, offsetX, offsetY, zoom, timeout);

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

SCENARIO( "A New ImageRenderPlayerBintr with WindowSinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new PipelineBintr" ) 
    {
        std::string playerName("player");

        std::string filePath("./test/streams/first-person-occurrence-438.jpeg");
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        uint timeout(0);

        DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_IMAGE_BINTR_NEW(playerName.c_str(),
                filePath.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

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
        std::string playerName("player");

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        uint offsetX(400);
        uint offsetY(200);
        uint zoom(50);
        bool repeatEnabled(0);

        DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                filePath.c_str(), DSL_RENDER_TYPE_OVERLAY, offsetX, offsetY, zoom, repeatEnabled);

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

SCENARIO( "A New VideoRenderPlayerBintr with WindowSinkBintr can Play and Stop correctly", "[PlayerBintr]" )
{
    GIVEN( "A new VideoRenderPlayerBintr" ) 
    {
        std::string playerName("player");

        std::string sourceName("file-source");
        std::string filePath = "./test/streams/sample_1080p_h264.mp4";
        uint offsetX(0);
        uint offsetY(0);
        uint zoom(50);
        int timeout(0);

        DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pPlayerBintr = 
            DSL_PLAYER_RENDER_VIDEO_BINTR_NEW(playerName.c_str(),
                filePath.c_str(), DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout);

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
