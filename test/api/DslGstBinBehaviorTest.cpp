
/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"uri-source-1");
static std::wstring uri(
    L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
        
static const uint skip_frames(0);
static const uint drop_frame_interval(0); 

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"egl-sink");

static const std::wstring gst_bin_name(L"gst-bin");
static const std::wstring gst_element_name1(L"gst-element-1");
static const std::wstring gst_element_name2(L"gst-element-2");

static const std::wstring factory_name1(L"queue");
static const std::wstring factory_name2(L"identity");

static const std::wstring pipeline_graph_name(L"gst-bin-behavior");

GThread* main_loop_thread(NULL);
// ---------------------------------------------------------------------------
//
// Thread function to start and wait on the main-loop
//
void dsl_error_message_handler(const wchar_t* source, 
    const wchar_t* message, void* client_data)
{

    dsl_pipeline_stop(pipeline_name.c_str());
    dsl_main_loop_quit();
}


SCENARIO( "A Pipeline with a Custom Component can play]", "[gst-bin-behavior]")
{
    GIVEN( "A Pipeline, URI source, GST Bin, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), uri.c_str(), 
            false, skip_frames, drop_frame_interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_gst_element_new(gst_element_name1.c_str(), 
            factory_name1.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gst_element_new(gst_element_name2.c_str(), 
            factory_name2.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gst_bin_new(gst_bin_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_gst_bin_element_add(gst_bin_name.c_str(),
            gst_element_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_bin_element_add(gst_bin_name.c_str(),
            gst_element_name2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {source_name1.c_str(), 
                gst_bin_name.c_str(), window_sink_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_error_message_handler_add(pipeline_name.c_str(), 
                dsl_error_message_handler, NULL) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_gst_element_property_uint_set(gst_element_name2.c_str(),
                    L"error-after", 50) == DSL_RESULT_SUCCESS );

                dsl_pipeline_play(pipeline_name.c_str());              
                
                dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                    const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                uint curValue(99);
                REQUIRE( dsl_gst_element_property_uint_get(gst_element_name1.c_str(),
                    L"current-level-buffers", &curValue) == DSL_RESULT_SUCCESS );
                std::cout << "current-level-buffers = " << curValue << std::endl;

                dsl_main_loop_run();

                dsl_delete_all();
            }
        }
    }
}