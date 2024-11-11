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

// ---------------------------------------------------------------------------
// Shared Test Inputs 

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(10000)

static const std::wstring source_uri = L"http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4";

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name1(L"file-source-1");

static const std::wstring sink_name(L"fake-sink");


SCENARIO( "A new Pipeline with a URI Source and Fake Sink can play", "[error]" )
{
    GIVEN( "A Pipeline, URI Source, Fake Sink" ) 
    {
        
        boolean inference_interval(4);
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name1.c_str(), source_uri.c_str(), 
            false, false, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_fake_new(sink_name.c_str()) == DSL_RESULT_SUCCESS );          

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_audiomux_enabled_set(pipeline_name.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            const wchar_t* components[] = {source_name1.c_str(), sink_name.c_str(), NULL};
    
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
