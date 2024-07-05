
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

static const std::wstring custom_source_name(L"custom-source");
static const std::wstring gst_element_name1(L"gst-element-1");
static const std::wstring gst_element_name2(L"gst-element-2");

static const std::wstring factory_name1(L"videotestsrc");
static const std::wstring factory_name2(L"capsfilter");

static const std::wstring caps_name(L"caps");
static const std::wstring caps_string(L"video/x-raw,width=1280,height=720");

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(1280);
static const uint sink_height(720);

static const std::wstring window_sink_name(L"egl-sink");

static const std::wstring pipeline_graph_name(L"custom-component-behavior");


SCENARIO( "A Pipeline with a Custom Source can play]", "[custom-source-behavior]")
{
    GIVEN( "A Pipeline, Custom Source, and Window Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_sink_window_egl_new(window_sink_name.c_str(), 
            offest_x, offest_y, sink_width, sink_height) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_gst_element_new(gst_element_name1.c_str(), 
            factory_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_element_property_uint_set(gst_element_name1.c_str(),
            L"pattern", 13) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_gst_element_new(gst_element_name2.c_str(), 
            factory_name2.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_gst_caps_new(caps_name.c_str(),
            caps_string.c_str()) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_gst_element_property_caps_set(gst_element_name2.c_str(),
            L"caps", caps_name.c_str()) == DSL_RESULT_SUCCESS );
        

        REQUIRE( dsl_source_custom_new(custom_source_name.c_str(), false) 
            == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_custom_element_add(custom_source_name.c_str(),
            gst_element_name1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_custom_element_add(custom_source_name.c_str(),
            gst_element_name2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {custom_source_name.c_str(), 
                window_sink_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                 dsl_pipeline_play(pipeline_name.c_str());              
                
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