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

static std::wstring record_tap_name(L"record-tap");
static std::wstring outdir(L"./");
static uint container(DSL_CONTAINER_MP4);

static dsl_record_client_listener_cb client_listener;

static std::wstring file_path(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");


SCENARIO( "The Components container is updated correctly on new Record Tap", "[tap-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Record Tap is created" ) 
        {
            REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
                container, client_listener) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint ret_cache_size(0);
                uint ret_width(0), ret_height(0);
                REQUIRE( dsl_tap_record_cache_size_get(record_tap_name.c_str(), &ret_cache_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_cache_size == DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC );
                REQUIRE( dsl_tap_record_dimensions_get(record_tap_name.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == 0 );
                REQUIRE( ret_height == 0 );
                REQUIRE( dsl_component_list_size() == 1 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "The Components container is updated correctly on Record Tap delete", "[tap-api]" )
{
    GIVEN( "A Record Tap Component" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new Record Tap is deleted" ) 
        {
            REQUIRE( dsl_component_delete(record_tap_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Record Tap's Init Parameters can be Set/Get ",  "[tap-api]" )
{
    GIVEN( "A new DSL_CONTAINER_MKV RecordTapBintr" ) 
    {
        REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        WHEN( "The Video Cache Size is set" )
        {
            uint new_cache_size(20), ret_cache_size(0);
            REQUIRE( dsl_tap_record_cache_size_set(record_tap_name.c_str(), new_cache_size) == DSL_RESULT_SUCCESS );

            THEN( "The correct cache size value is returned" )
            {
                REQUIRE( dsl_tap_record_cache_size_get(record_tap_name.c_str(), &ret_cache_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_cache_size == new_cache_size );
                REQUIRE( dsl_component_delete(record_tap_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }

        WHEN( "The Video Recording Dimensions are set" )
        {
            uint new_width(1024), new_height(780), ret_width(99), ret_height(99);
            REQUIRE( dsl_tap_record_dimensions_set(record_tap_name.c_str(), new_width, new_height) == DSL_RESULT_SUCCESS );

            THEN( "The correct cache size value is returned" )
            {
                REQUIRE( dsl_tap_record_dimensions_get(record_tap_name.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == new_width );
                REQUIRE( ret_height == ret_height );
                REQUIRE( dsl_component_delete(record_tap_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An invalid New parameters are checked on Record Tap create", "[tap-api]" )
{
    GIVEN( "An attributes for a new Record Tap" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "An invalid Output Directory is specified" ) 
        {
            std::wstring outdir(L"/this/is/a/bad/path");
            uint container(DSL_CONTAINER_MKV);

            THEN( "The New Record Tap fails to create" )
            {
                REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
                    container, client_listener) == DSL_RESULT_TAP_FILE_PATH_NOT_FOUND );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "An invalid Container Type is specified" ) 
        {
            std::wstring outdir(L"./");
            uint container(DSL_CONTAINER_MKV+1);

            THEN( "The New Record Tap fails to create" )
            {
                REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
                    container, client_listener) == DSL_RESULT_TAP_CONTAINER_VALUE_INVALID );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An invalid Component is checked on Record Tap Get/Set", "[tap-api]" )
{
    GIVEN( "An invalid container parameter" ) 
    {
        std::wstring nonRecordTapName(L"record-tap");

        WHEN( "An invalid component is used to Get/Set Cache Size" ) 
        {
            // use a fake sink as our invalid component
            REQUIRE( dsl_sink_fake_new(nonRecordTapName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The New Record Tap fails to create" )
            {
                uint cache_size(0);
                REQUIRE( dsl_tap_record_cache_size_get(nonRecordTapName.c_str(), &cache_size) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_tap_record_cache_size_set(nonRecordTapName.c_str(), cache_size) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                    
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "An invalid component is used to Get/Set Dimensions" ) 
        {
            // use a fake sink as our invalid component
            REQUIRE( dsl_sink_fake_new(nonRecordTapName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The New Record Tap fails to create" )
            {
                uint width, height(0);
                REQUIRE( dsl_tap_record_dimensions_get(nonRecordTapName.c_str(), &width, &height) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                REQUIRE( dsl_tap_record_dimensions_set(nonRecordTapName.c_str(), width, height) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Player can be added to and removed from a Record Tap", "[tap-api]" )
{
    GIVEN( "A new Record Tap and Image Player" )
    {
        REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        std::wstring player_name(L"player");
        
        REQUIRE( dsl_player_render_video_new(player_name.c_str(),file_path.c_str(), 
            DSL_RENDER_TYPE_OVERLAY, 10, 10, 75, 0) == DSL_RESULT_SUCCESS );

        WHEN( "A Image Player is added" )
        {
            REQUIRE( dsl_tap_record_video_player_add(record_tap_name.c_str(),
                player_name.c_str()) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_tap_record_video_player_add(record_tap_name.c_str(),
                player_name.c_str()) == DSL_RESULT_TAP_PLAYER_ADD_FAILED );

            THEN( "The same Image Player can be remove" ) 
            {
                REQUIRE( dsl_tap_record_video_player_remove(record_tap_name.c_str(),
                    player_name.c_str()) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_tap_record_video_player_remove(record_tap_name.c_str(),
                    player_name.c_str()) == DSL_RESULT_TAP_PLAYER_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete(record_tap_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_player_delete(player_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}    

SCENARIO( "A Mailer can be added to and removed from a Record Tap", "[tap-api]" )
{
    GIVEN( "A new Record Tap and Mailer" )
    {
        REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        std::wstring mailer_name(L"mailer");
        
        std::wstring subject(L"Subject line");
        
        REQUIRE( dsl_mailer_new(mailer_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Mailer is added" )
        {
            REQUIRE( dsl_tap_record_mailer_add(record_tap_name.c_str(),
                mailer_name.c_str(), subject.c_str()) == DSL_RESULT_SUCCESS );

            // ensure the same listener twice fails
            REQUIRE( dsl_tap_record_mailer_add(record_tap_name.c_str(),
                mailer_name.c_str(), subject.c_str()) == DSL_RESULT_TAP_MAILER_ADD_FAILED );

            THEN( "The Mailer can be removed" ) 
            {
                REQUIRE( dsl_tap_record_mailer_remove(record_tap_name.c_str(),
                    mailer_name.c_str()) == DSL_RESULT_SUCCESS );

                // calling a second time must fail
                REQUIRE( dsl_tap_record_mailer_remove(record_tap_name.c_str(),
                    mailer_name.c_str()) == DSL_RESULT_TAP_MAILER_REMOVE_FAILED );
                    
                REQUIRE( dsl_component_delete(record_tap_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
                REQUIRE( dsl_mailer_delete(mailer_name.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_mailer_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "The Tap API checks for NULL input parameters", "[tap-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        uint cache_size(0), width(0), height(0);
        boolean is_on(0), reset_done(0);

       std::wstring mailerName(L"mailer");        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                
                REQUIRE( dsl_tap_record_new(NULL, NULL,  0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_new(record_tap_name.c_str(), NULL, 0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_session_start(NULL, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_session_stop(NULL, false) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_cache_size_get(NULL, &cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_cache_size_set(NULL, cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_dimensions_get(NULL, &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_dimensions_set(NULL, width, height) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_is_on_get(NULL, &is_on) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_reset_done_get(NULL, &reset_done) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_video_player_add(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_video_player_add(record_tap_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_video_player_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_video_player_remove(record_tap_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_mailer_add(NULL, NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_mailer_add(record_tap_name.c_str(), NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_mailer_add(record_tap_name.c_str(), mailerName.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_mailer_remove(NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_mailer_remove(record_tap_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
