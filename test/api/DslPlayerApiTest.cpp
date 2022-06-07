/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

static std::wstring player_name(L"player");

static std::wstring source_name(L"file-source");
static std::wstring file_path(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

static std::wstring image_path1(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");
static std::wstring image_path2(L"/opt/nvidia/deepstream/deepstream/samples/streams/yoga.jpg");

static std::wstring sink_name(L"window-sink");
static uint offsetX(0);
static uint offsetY(0);
static uint sinkW(1280);
static uint sinkH(720);


SCENARIO( "A single Player is created and deleted correctly", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_window_new(sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_new(player_name.c_str(),
                source_name.c_str(), sink_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_player_list_size() == 1 );
                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A single Player can Play, Pause, and Stop", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_window_new(sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_new(player_name.c_str(),
                source_name.c_str(), sink_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_pause(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_stop(player_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_stop(player_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A File Render Player can Play, Pause, and Stop", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_render_video_new(player_name.c_str(),file_path.c_str(), 
                DSL_RENDER_TYPE_WINDOW, 10, 10, 75, false) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_pause(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_stop(player_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An Image Render Player can Play, Pause, and Stop", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_render_image_new(player_name.c_str(),image_path1.c_str(), 
                DSL_RENDER_TYPE_WINDOW, 10, 10, 75, 0) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_pause(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_play(player_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_player_stop(player_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An Image Render Player's Attributes are updated correctly'", "[player-api]" )
{
    GIVEN( "A new Image Render Player with Window Sink" ) 
    {
        uint offsetX(123);
        uint offsetY(123);
        uint retOffsetX(0);
        uint retOffsetY(0);
        
        uint zoom(75),  retZoom(57);
        uint timeout(0),  retTimeout(444);
        
        REQUIRE( dsl_player_list_size() == 0 );
        REQUIRE( dsl_player_render_image_new(player_name.c_str(), image_path1.c_str(), 
            DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_player_render_offsets_get(player_name.c_str(), 
            &retOffsetX, &retOffsetY) == DSL_RESULT_SUCCESS );
        REQUIRE( retOffsetX == offsetX );
        REQUIRE( retOffsetY == offsetY );
        
        REQUIRE( dsl_player_render_zoom_get(player_name.c_str(), 
            &retZoom) == DSL_RESULT_SUCCESS );
        REQUIRE( retZoom == zoom );

        REQUIRE( dsl_player_render_image_timeout_get(player_name.c_str(), 
            &retTimeout) == DSL_RESULT_SUCCESS );
        REQUIRE( retTimeout == timeout );

        WHEN( "A the Player's Attributes are Set" ) 
        {
            uint newOffsetX(321), newOffsetY(321);
            REQUIRE( dsl_player_render_offsets_set(player_name.c_str(),
                newOffsetX, newOffsetY) == DSL_RESULT_SUCCESS );

            uint newZoom(543);
            REQUIRE( dsl_player_render_zoom_set(player_name.c_str(),
                newZoom) == DSL_RESULT_SUCCESS );

            uint newTimeout(101);
            REQUIRE( dsl_player_render_image_timeout_set(player_name.c_str(),
                newTimeout) == DSL_RESULT_SUCCESS );

            THEN( "The correct Attribute values are returned on Get" ) 
            {
                REQUIRE( dsl_player_render_offsets_get(player_name.c_str(), 
                    &retOffsetX, &retOffsetY) == DSL_RESULT_SUCCESS );
                REQUIRE( retOffsetX == newOffsetX );
                REQUIRE( retOffsetY == newOffsetY );

                REQUIRE( dsl_player_render_zoom_get(player_name.c_str(), 
                    &retZoom) == DSL_RESULT_SUCCESS );
                REQUIRE( retZoom == newZoom );

                REQUIRE( dsl_player_render_image_timeout_get(player_name.c_str(), 
                    &retTimeout) == DSL_RESULT_SUCCESS );
                REQUIRE( retTimeout == newTimeout );

                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An Video Render Player's Attributes are updated correctly'", "[mmm]" )
{
    GIVEN( "A new Video Render Player with Window Sink" ) 
    {
        uint offsetX(123);
        uint offsetY(123);
        uint retOffsetX(0);
        uint retOffsetY(0);
        
        uint zoom(75),  retZoom(57);
        boolean repeatEnabled(false),  retRepeatEnabled(true);
        
        REQUIRE( dsl_player_list_size() == 0 );
        REQUIRE( dsl_player_render_video_new(player_name.c_str(), file_path.c_str(), 
            DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, repeatEnabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_player_render_offsets_get(player_name.c_str(), 
            &retOffsetX, &retOffsetY) == DSL_RESULT_SUCCESS );
        REQUIRE( retOffsetX == offsetX );
        REQUIRE( retOffsetY == offsetY );
        
        REQUIRE( dsl_player_render_zoom_get(player_name.c_str(), 
            &retZoom) == DSL_RESULT_SUCCESS );
        REQUIRE( retZoom == zoom );

        REQUIRE( dsl_player_render_video_repeat_enabled_get(player_name.c_str(), 
            &retRepeatEnabled) == DSL_RESULT_SUCCESS );
        REQUIRE( retRepeatEnabled == repeatEnabled );

        WHEN( "A the Player's Attributes are Set" ) 
        {
            REQUIRE( dsl_player_render_file_path_set(player_name.c_str(),
                image_path2.c_str()) == DSL_RESULT_SUCCESS );
            
            uint newOffsetX(321), newOffsetY(321);
            REQUIRE( dsl_player_render_offsets_set(player_name.c_str(),
                newOffsetX, newOffsetY) == DSL_RESULT_SUCCESS );

            uint newZoom(543);
            REQUIRE( dsl_player_render_zoom_set(player_name.c_str(),
                newZoom) == DSL_RESULT_SUCCESS );

            boolean newRepeatEnabled(true);
            REQUIRE( dsl_player_render_video_repeat_enabled_set(player_name.c_str(),
                newRepeatEnabled) == DSL_RESULT_SUCCESS );

            THEN( "The correct Attribute values are returned on Get" ) 
            {
                REQUIRE( dsl_player_render_offsets_get(player_name.c_str(), 
                    &retOffsetX, &retOffsetY) == DSL_RESULT_SUCCESS );
                REQUIRE( retOffsetX == newOffsetX );
                REQUIRE( retOffsetY == newOffsetY );

                REQUIRE( dsl_player_render_zoom_get(player_name.c_str(), 
                    &retZoom) == DSL_RESULT_SUCCESS );
                REQUIRE( retZoom == newZoom );

                REQUIRE( dsl_player_render_video_repeat_enabled_get(player_name.c_str(), 
                    &retRepeatEnabled) == DSL_RESULT_SUCCESS );
                REQUIRE( retRepeatEnabled == newRepeatEnabled );

                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Players's XWindow Handle can be Set/Get", "[player-api]" )
{
    GIVEN( "A new Player" ) 
    {
        uint zoom(75);        
        boolean repeatEnabled(false);
        
        uint64_t handle(0);
        
        REQUIRE( dsl_player_render_video_new(player_name.c_str(), file_path.c_str(), 
            DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, repeatEnabled) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_player_xwindow_handle_get(player_name.c_str(), 
            &handle) == DSL_RESULT_SUCCESS );
            
        // must be initialized false
        REQUIRE( handle == 0 );

        WHEN( "When the Player's XWindow Handle is updated" ) 
        {
            handle = 0x1234567812345678;
            REQUIRE( dsl_player_xwindow_handle_set(player_name.c_str(), 
                handle) == DSL_RESULT_SUCCESS );
                
            THEN( "The new handle value is returned on get" )
            {
                uint64_t newHandle(0);
                REQUIRE( dsl_player_xwindow_handle_get(player_name.c_str(), 
                    &newHandle) == DSL_RESULT_SUCCESS );
                REQUIRE( handle == newHandle );

                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}


SCENARIO( "The Player API checks for NULL input parameters", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        uint timeout(0);
        uint zoom(100);
        boolean repeat_enabled(0);

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_window_new(sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_player_new(NULL,
                    source_name.c_str(), sink_name.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_new(player_name.c_str(),
                    NULL, sink_name.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_new(player_name.c_str(),
                    source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_render_reset(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_render_video_new(NULL, NULL, 
                    DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, repeat_enabled) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_render_image_new(NULL, NULL, 
                    DSL_RENDER_TYPE_WINDOW, offsetX, offsetY, zoom, timeout) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_sink_render_reset(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_play(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_pause(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_stop(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_state_get(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_list_size() == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}
