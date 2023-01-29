/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"uri-source-1");
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
static const uint skip_frames(0);
static const uint drop_frame_interval(0); 

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"window-sink");

SCENARIO( "A URI File Source can play with buffer-out-format = RGBA]",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"window-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-format is set to RGBA" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_format_set(source_name1.c_str(),
                DSL_VIDEO_FORMAT_RGBA) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A URI File Source can play with buffer-out-format = NV12]",
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"window-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-format is set to RGBA" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_format_set(source_name1.c_str(),
                DSL_VIDEO_FORMAT_NV12) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A URI File Source can play with buffer-out-orientation = \
dsl_source_buffer_out_orientation_set]", "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"window-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-orientation is set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_orientation_set(source_name1.c_str(),
                DSL_VIDEO_ORIENTATION_FLIP_UPPER_LEFT_TO_LOWER_RIGHT) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A URI File Source can play with buffer-out-crop-pre set]", 
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"window-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-crop settings are set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_crop_rectangle_set(source_name1.c_str(),
                DSL_VIDEO_CROP_PRE_CONVERSION, 10, 10, 200, 200) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A URI File Source can play with buffer-out-crop-post set]", 
    "[buffer-out-behavior]")
{
    GIVEN( "A Pipeline, URI source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source-1", L"window-sink", NULL};
        
        REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the buffer-out-crop settings are set" ) 
        {
            REQUIRE( dsl_source_video_buffer_out_crop_rectangle_set(source_name1.c_str(),
                DSL_VIDEO_CROP_POST_CONVERSION, 1000, 1000, 200, 200) == DSL_RESULT_SUCCESS );
                
            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}
