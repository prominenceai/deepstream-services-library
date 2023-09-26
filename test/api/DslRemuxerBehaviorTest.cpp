/*
The MIT License

Copyright (c) 2023, Prominence AI, Inc.

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

static const std::wstring uri(L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4");

static const std::wstring source_name1(L"file-source-1");
static const std::wstring source_name2(L"file-source-2");
static const std::wstring source_name3(L"file-source-3");
static const std::wstring source_name4(L"file-source-4");

static const std::wstring remuxer_name(L"remuxer");

static const std::wstring branch_name1(L"branch-1");
static const std::wstring branch_name2(L"branch-2");

static const std::wstring tiler_name1(L"tiler-1");
static const std::wstring tiler_name2(L"tiler-2");
static const std::wstring tiler_name3(L"tiler-3");
static const std::wstring tiler_name4(L"tiler-4");


static const std::wstring sink_name1(L"sink-1");
static const std::wstring sink_name2(L"sink-2");
static const std::wstring sink_name3(L"sink-3");
static const std::wstring sink_name4(L"sink-4");

static const uint offest_x(0);
static const uint offest_y(0);
static const uint sink_width(640);
static const uint sink_height(360);

// -----------------------------------------------------------------------------------

SCENARIO( "Two File Sources, Remuxer, and two Window Sinks can play", 
    "[remuxer-behavior]")
{
    GIVEN( "A Pipeline, two File sources, Dewarper, and two Window-Sinks" ) 
    {
        static const uint width(1280);
        static const uint height(360);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_file_new(source_name1.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_source_file_new(source_name2.c_str(), uri.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name1.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tiler_name2.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(sink_name1.c_str(),
            offest_x, offest_y, width, height) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name1.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(sink_name2.c_str(),
            offest_x+300, offest_y+300, width, height) 
            == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_sync_enabled_set(sink_name2.c_str(), 
            false) == DSL_RESULT_SUCCESS );

        const wchar_t* branch1_components[] = {
            tiler_name1.c_str(), sink_name1.c_str(), NULL};

        const wchar_t* branch2_components[] = {
            tiler_name2.c_str(), sink_name2.c_str(), NULL};

        REQUIRE( dsl_branch_new_component_add_many(branch_name1.c_str(), 
            branch1_components) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_branch_new_component_add_many(branch_name2.c_str(), 
            branch2_components) == DSL_RESULT_SUCCESS );
        
        const wchar_t* remuxer_branches[] = {
            branch_name1.c_str(), branch_name2.c_str(), NULL};

        REQUIRE( dsl_tee_remuxer_new(remuxer_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_tee_branch_add(remuxer_name.c_str(), 
            branch_name1.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tee_branch_add(remuxer_name.c_str(), 
            branch_name2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is assembled" ) 
        {
            const wchar_t* components[] = {
                source_name1.c_str(), source_name2.c_str(), 
                remuxer_name.c_str(), NULL};
//            const wchar_t* components[] = {
//                source_name1.c_str(), 
//                remuxer_name.c_str(), NULL};
            
            REQUIRE( dsl_pipeline_new_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline is able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);

                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );

                dsl_delete_all();
            }
        }
    }
}
