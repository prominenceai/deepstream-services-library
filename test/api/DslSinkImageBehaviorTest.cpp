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
#include "Dsl.h"
#include "DslApi.h"

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring source_name(L"file-source");
static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring window_sink_name(L"egl-sink");
static const uint offsetX(0);
static const uint offsetY(0);
static const uint sinkW(1280);
static const uint sinkH(720);

static const std::wstring image_sink_name(L"multi-image-sink");
static const std::wstring output_file(L"./frame-%05d.jpg");

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(5000)

SCENARIO( "A New Pipeline with a File Source, Window Sink, and a Multi-Object Sink can \
Play and Stop correctly", "[ImageSinkBehavior]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        
        REQUIRE( dsl_source_file_new(source_name.c_str(), uri.c_str(),
            true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_image_multi_new(image_sink_name.c_str(),
            output_file.c_str(), 0, 0, 1, 1) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", 
            L"multi-image-sink", L"egl-sink", NULL};

        WHEN( "The new Pipeline is asembled" )
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline can link-all and play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A Multi-Object Sink can scale the frame capture correctly", "[ImageSinkBehavior]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        
        REQUIRE( dsl_source_file_new(source_name.c_str(), uri.c_str(),
            true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_image_multi_new(image_sink_name.c_str(),
            output_file.c_str(), 640, 360, 1, 1) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", 
            L"multi-image-sink", L"egl-sink", NULL};

        WHEN( "The new Pipeline is asembled" )
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline can link-all and play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A Multi-Object Sink can limit its file count correctly", "[ImageSinkBehavior]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        
        REQUIRE( dsl_source_file_new(source_name.c_str(), uri.c_str(),
            true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_image_multi_new(image_sink_name.c_str(),
            output_file.c_str(), 0, 0, 1, 2) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_image_multi_file_max_set(image_sink_name.c_str(), 
            1) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"file-source", 
            L"multi-image-sink", L"egl-sink", NULL};

        WHEN( "The new Pipeline is asembled" )
        {
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline can link-all and play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                dsl_delete_all();
            }
        }
    }
}
