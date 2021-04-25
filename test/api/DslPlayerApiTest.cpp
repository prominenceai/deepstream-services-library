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

SCENARIO( "A single Player is created and deleted correctly", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        std::wstring player_name  = L"player";

        std::wstring source_name = L"file-source";
        std::wstring file_path = L"./test/streams/sample_1080p_h264.mp4";

        std::wstring sinkName = L"overlay-sink";
        uint displayId(0);
        uint depth(0);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_overlay_new(sinkName.c_str(), displayId, depth, 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_new(player_name.c_str(),
                source_name.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );

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
        std::wstring player_name  = L"player";

        std::wstring source_name = L"file-source";
        std::wstring file_path = L"./test/streams/sample_1080p_h264.mp4";

        std::wstring sinkName = L"overlay-sink";
        uint displayId(0);
        uint depth(0);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_overlay_new(sinkName.c_str(), displayId, depth, 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_new(player_name.c_str(),
                source_name.c_str(), sinkName.c_str()) == DSL_RESULT_SUCCESS );

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

SCENARIO( "A File Render Player can Play, Pause, and Stop", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        std::wstring player_name  = L"player";

        std::wstring file_path = L"./test/streams/sample_1080p_h264.mp4";

        uint offsetX(0);
        uint offsetY(0);
        
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_render_video_new(player_name.c_str(),file_path.c_str(), 
                DSL_RENDER_TYPE_OVERLAY, 10, 10, 75, false) == DSL_RESULT_SUCCESS );

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

SCENARIO( "An Image  Render Player can Play, Pause, and Stop", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        std::wstring player_name  = L"player";

        std::wstring file_path = L"./test/streams/first-person-occurrence-438.jpeg";

        uint offsetX(0);
        uint offsetY(0);
        
        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "A new Player is created" ) 
        {
            REQUIRE( dsl_player_render_image_new(player_name.c_str(),file_path.c_str(), 
                DSL_RENDER_TYPE_OVERLAY, 10, 10, 75, 0) == DSL_RESULT_SUCCESS );

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



SCENARIO( "The Player API checks for NULL input parameters", "[player-api]" )
{
    GIVEN( "An empty list of Players" ) 
    {
        std::wstring player_name  = L"player";

        std::wstring source_name = L"file-source";
        std::wstring file_path = L"./test/streams/sample_1080p_h264.mp4";

        std::wstring sinkName = L"overlay-sink";
        uint displayId(0);
        uint depth(0);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_source_file_new(source_name.c_str(), file_path.c_str(), 
            false) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_overlay_new(sinkName.c_str(), displayId, depth, 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_player_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_player_new(NULL,
                    source_name.c_str(), sinkName.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_new(player_name.c_str(),
                    NULL, sinkName.c_str()) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_new(player_name.c_str(),
                    source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_play(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_pause(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_player_stop(NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_player_list_size() == 0 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}
