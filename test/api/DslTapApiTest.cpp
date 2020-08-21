/*
The MIT License

Copyright (c) 2019-2020, ROBERT HOWELL

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
SCENARIO( "The Components container is updated correctly on new Record Tap", "[tap-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring recordTapName(L"record-tap");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);

        dsl_record_client_listner_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Record Tap is created" ) 
        {
            REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
                container, client_listener) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                uint ret_cache_size(0);
                uint ret_width(0), ret_height(0);
                REQUIRE( dsl_tap_record_cache_size_get(recordTapName.c_str(), &ret_cache_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_cache_size == DSL_DEFAULT_VIDEO_RECORD_CACHE_IN_SEC );
                REQUIRE( dsl_tap_record_dimensions_get(recordTapName.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
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
        std::wstring recordTapName(L"record-tap");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);

        dsl_record_client_listner_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "A new Record Tap is deleted" ) 
        {
            REQUIRE( dsl_component_delete(recordTapName.c_str()) == DSL_RESULT_SUCCESS );
            
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
        std::wstring recordTapName(L"record-tap");
        std::wstring outdir(L"./");
        uint container(DSL_CONTAINER_MP4);

        dsl_record_client_listner_cb client_listener;
        
        REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
            container, client_listener) == DSL_RESULT_SUCCESS );

        WHEN( "The Video Cache Size is set" )
        {
            uint new_cache_size(20), ret_cache_size(0);
            REQUIRE( dsl_tap_record_cache_size_set(recordTapName.c_str(), new_cache_size) == DSL_RESULT_SUCCESS );

            THEN( "The correct cache size value is returned" )
            {
                REQUIRE( dsl_tap_record_cache_size_get(recordTapName.c_str(), &ret_cache_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_cache_size == new_cache_size );
                REQUIRE( dsl_component_delete(recordTapName.c_str()) == DSL_RESULT_SUCCESS );
            }
        }

        WHEN( "The Video Recording Dimensions are set" )
        {
            uint new_width(1024), new_height(780), ret_width(99), ret_height(99);
            REQUIRE( dsl_tap_record_dimensions_set(recordTapName.c_str(), new_width, new_height) == DSL_RESULT_SUCCESS );

            THEN( "The correct cache size value is returned" )
            {
                REQUIRE( dsl_tap_record_dimensions_get(recordTapName.c_str(), &ret_width, &ret_height) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_width == new_width );
                REQUIRE( ret_height == ret_height );
                REQUIRE( dsl_component_delete(recordTapName.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An invalid New parameters are checked on Record Tap create", "[tap-api]" )
{
    GIVEN( "An attributes for a new Record Tap" ) 
    {
        std::wstring recordTapName(L"record-tap");

        dsl_record_client_listner_cb client_listener;

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "An invalid Output Directory is specified" ) 
        {
            std::wstring outdir(L"/this/is/a/bad/path");
            uint container(DSL_CONTAINER_MKV);

            THEN( "The New Record Tap fails to create" )
            {
                REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
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
                REQUIRE( dsl_tap_record_new(recordTapName.c_str(), outdir.c_str(),
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

SCENARIO( "The Tap API checks for NULL input parameters", "[tap-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring tapName  = L"test-tap";
        std::wstring otherName  = L"other";
        
        uint cache_size(0), width(0), height(0);
        boolean is_on(0), reset_done(0), sync(0), async(0);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                
                REQUIRE( dsl_tap_record_new(NULL, NULL,  0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_new(tapName.c_str(), NULL, 0, NULL ) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_session_start(NULL, 0, 0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_session_stop(NULL, 0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_cache_size_get(NULL, &cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_cache_size_set(NULL, cache_size) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_dimensions_get(NULL, &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tap_record_dimensions_set(NULL, width, height) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_is_on_get(NULL, &is_on) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tap_record_reset_done_get(NULL, &reset_done) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
