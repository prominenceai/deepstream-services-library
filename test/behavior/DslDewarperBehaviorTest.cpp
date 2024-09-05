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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(3000)

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"uri-source-1");
static const uint skip_frames(0);
static const uint drop_frame_interval(0); 

static std::wstring dewarper_name(L"dewarper");

static const std::wstring tiler_name1(L"tiler");
static const uint tiler_width(1280);
static const uint tiler_height(720);

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"egl-sink");

static const std::wstring pipeline_graph_name(L"dewarper-behavior");

static uint camera_id(6);  // matches sample_cam6.mp4


SCENARIO( "A URI File Source with a Dewarper -- 360 camera multi-surface \
use-case -- can play]", "[dewarper-behavior]")
{
    GIVEN( "A Pipeline, URI source, Dewarper, and Window Sink" ) 
    {
        std::wstring uri(
            L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_cam6.mp4");
        std::wstring dewarper_config_file(
            L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper.txt");
            
        uint camera_id(6); // for sample_cam6.mp4

        uint sink_width(960);
        uint sink_height(752);
        uint muxer_batch_timeout_usec(33000);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_dewarper_new(dewarper_name.c_str(), 
            dewarper_config_file.c_str(), camera_id) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_video_dewarper_add(source_name1.c_str(), 
            dewarper_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            sink_width, sink_height*4) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width/2, sink_height*2) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {L"uri-source-1", 
                L"tiler", L"egl-sink", NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            uint num_surfaces(0);
            REQUIRE( dsl_dewarper_num_batch_buffers_get(dewarper_name.c_str(), 
                &num_surfaces) == DSL_RESULT_SUCCESS );
                
            if (dsl_info_use_new_nvstreammux_get())
            {
                REQUIRE( dsl_pipeline_streammux_batch_size_set(
                    pipeline_name.c_str(), num_surfaces) 
                        == DSL_RESULT_SUCCESS );
            }
            else
            {
                REQUIRE( dsl_pipeline_streammux_batch_properties_set(pipeline_name.c_str(), 
                    num_surfaces, -1) == DSL_RESULT_SUCCESS );
            }
            REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_set(
                pipeline_name.c_str(), num_surfaces) == DSL_RESULT_SUCCESS );
                
            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                    const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*2);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

SCENARIO( "A URI File Source with a Dewarper -- single-surface Perspective \
Projection use-case -- can play]", "[dewarper-behavior]")
{
    GIVEN( "A Pipeline, URI source, Dewarper, and Window Sink" ) 
    {
        std::wstring uri(
            L"/opt/nvidia/deepstream/deepstream/samples/streams/yoga.mp4");
        std::wstring dewarper_config_file(
            L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper_perspective.txt");
            
        uint camera_id(0); // csv files are not used 

        uint muxer_width(3680);
        uint muxer_height(2428);        
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_dewarper_new(dewarper_name.c_str(), 
            dewarper_config_file.c_str(), camera_id) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_dewarper_num_batch_buffers_set(dewarper_name.c_str(), 1) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_video_dewarper_add(source_name1.c_str(), 
            dewarper_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {L"uri-source-1", 
                L"egl-sink", NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            if (!dsl_info_use_new_nvstreammux_get())
            {
                REQUIRE( dsl_pipeline_streammux_dimensions_set(
                    pipeline_name.c_str(), muxer_width, muxer_height) 
                        == DSL_RESULT_SUCCESS );
            }
            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                    const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}

