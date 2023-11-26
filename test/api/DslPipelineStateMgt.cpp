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
#include "Dsl.h"
#include "DslApi.h"

SCENARIO( "A new Pipeline with minimal components can Play", "[PipelineStateMgt]" )
{
    GIVEN( "A Pipeline with minimal components" ) 
    {
        std::wstring sourceName = L"test-uri-source";
        std::wstring uri = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";
        uint intrDecode(false);
        uint dropFrameInterval(2);

        std::wstring tilerName = L"tiler";
        uint width(1280);
        uint height(720);

        std::wstring windowSinkName = L"egl-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(sourceName.c_str(), uri.c_str(),
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
    
        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {L"test-uri-source", L"tiler", L"egl-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled and Played" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline can be Stopped and Disassembled" )
            {
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                REQUIRE( dsl_pipeline_pause(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
                REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }

    }
}

