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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(500)

SCENARIO( "The Batch Size for a Pipeline can be set greater than sources", "[pipeline-streammux]" )
{
    GIVEN( "A Pipeline with three sources and minimal components" ) 
    {
        std::wstring sourceName1 = L"test-uri-source-1";
        std::wstring sourceName2 = L"test-uri-source-2";
        std::wstring sourceName3 = L"test-uri-source-3";
        std::wstring uri = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName = L"tiler";
        uint width(1920);
        uint height(720);

        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1920);
        uint sinkH(720);

        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName3.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {L"test-uri-source-1", L"test-uri-source-2", L"test-uri-source-3", 
            L"tiler", L"window-sink", NULL};

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        uint batch_size(0), batch_timeout(0);
        
        dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), &batch_size, &batch_timeout);
        REQUIRE( batch_size == 0 );
        REQUIRE( batch_timeout == DSL_STREAMMUX_DEFAULT_BATCH_TIMEOUT );
        
        WHEN( "The Pipeline's Stream Muxer Batch Size is set to more than the number of sources" ) 
        {
            uint new_batch_size(6), new_batch_timeout(50000);
            REQUIRE( dsl_pipeline_streammux_batch_properties_set(pipelineName.c_str(), new_batch_size, new_batch_timeout) == DSL_RESULT_SUCCESS );
            dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), &batch_size, &batch_timeout);
            REQUIRE( batch_size == new_batch_size );
            REQUIRE( batch_timeout == new_batch_timeout );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The updated Stream Muxer Batch Size is used" )
            {
                dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), &batch_size, &batch_timeout);
                REQUIRE( batch_size == new_batch_size );
                REQUIRE( batch_timeout == new_batch_timeout );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Batch Size for a Pipeline can be set less than sources", "[pipeline-streammux]" )
{
    GIVEN( "A Pipeline with three sources and minimal components" ) 
    {
        std::wstring sourceName1 = L"test-uri-source-1";
        std::wstring sourceName2 = L"test-uri-source-2";
        std::wstring sourceName3 = L"test-uri-source-3";
        std::wstring uri = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";
        uint intrDecode(false);
        uint dropFrameInterval(0);

        std::wstring tilerName = L"tiler";
        uint width(1920);
        uint height(720);

        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1920);
        uint sinkH(720);

        std::wstring pipelineName  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName3.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_new(windowSinkName.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {L"test-uri-source-1", L"test-uri-source-2", L"test-uri-source-3", 
            L"tiler", L"window-sink", NULL};

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        uint batch_size(0), batch_timeout(0);
        
        dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), &batch_size, &batch_timeout);
        REQUIRE( batch_size == 0 );
        REQUIRE( batch_timeout == DSL_STREAMMUX_DEFAULT_BATCH_TIMEOUT );
        
        WHEN( "The Pipeline's Stream Muxer Batch Size is set to more than the number of sources" ) 
        {
            uint new_batch_size(1), new_batch_timeout(50000);
            REQUIRE( dsl_pipeline_streammux_batch_properties_set(pipelineName.c_str(), 
                new_batch_size, new_batch_timeout) == DSL_RESULT_SUCCESS );
            dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), &batch_size, &batch_timeout);
            REQUIRE( batch_size == new_batch_size );
            REQUIRE( batch_timeout == new_batch_timeout );
        
            REQUIRE( dsl_pipeline_component_add_many(pipelineName.c_str(), components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The updated Stream Muxer Batch Size is used" )
            {
                dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), &batch_size, &batch_timeout);
                REQUIRE( batch_size == new_batch_size );
                REQUIRE( batch_timeout == new_batch_timeout );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The NVIDIA buffer memory type for a Pipeline's Streammuxer can be read and updated", "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        uint nvbuf_mem_type(99);
        
        REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_get(pipelineName.c_str(), 
            &nvbuf_mem_type)  == DSL_RESULT_SUCCESS );
        REQUIRE( nvbuf_mem_type == DSL_NVBUF_MEM_TYPE_DEFAULT );
        
        WHEN( "The Pipeline's Streammuxer's NVIDIA buffer memory type is updated" ) 
        {
            uint new_nvbuf_mem_type(DSL_NVBUF_MEM_TYPE_UNIFIED);

            REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_set(pipelineName.c_str(), 
                new_nvbuf_mem_type) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer NVIDIA buffer memory type  is returned" )
            {
                REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_get(pipelineName.c_str(), 
                    &nvbuf_mem_type) == DSL_RESULT_SUCCESS );
                REQUIRE( nvbuf_mem_type == new_nvbuf_mem_type );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
        WHEN( "An invalid NVIDIA buffer memory type is used on set" ) 
        {
            uint new_nvbuf_mem_type(99);

            REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_set(pipelineName.c_str(), 
                new_nvbuf_mem_type) == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );

            THEN( "The Streammuxer NVIDIA buffer memory type is unchanged" )
            {
                REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_get(pipelineName.c_str(), 
                    &nvbuf_mem_type) == DSL_RESULT_SUCCESS );
                REQUIRE( nvbuf_mem_type == DSL_NVBUF_MEM_TYPE_DEFAULT );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tiler can be added to and removed from a Pipeline's Streammuxer output", 
    "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer and new Tiler" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        std::wstring tilerName  = L"tiler";

        REQUIRE( dsl_tiler_new(tilerName.c_str(), DSL_STREAMMUX_DEFAULT_WIDTH,
            DSL_STREAMMUX_DEFAULT_HEIGHT) == DSL_RESULT_SUCCESS );

        // chech that a removal attempt after pipeline creation fails
        REQUIRE( dsl_pipeline_streammux_tiler_remove(pipelineName.c_str()) 
            == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );
        
        WHEN( "A Tiler is added to a Pipeline's Streammuxer output" ) 
        {
            REQUIRE( dsl_pipeline_streammux_tiler_add(pipelineName.c_str(), 
                tilerName.c_str()) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_pipeline_streammux_tiler_add(pipelineName.c_str(), 
                tilerName.c_str()) == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );

            THEN( "The Tiler can be successfully removed" )
            {
                REQUIRE( dsl_pipeline_streammux_tiler_remove(pipelineName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                // second call must fail
                REQUIRE( dsl_pipeline_streammux_tiler_remove(pipelineName.c_str()) 
                    == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "An a non-tiler component is passed as input" ) 
        {
            std::wstring fakeSinkName(L"fake-sink");
            
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Streammuxer NVIDIA buffer memory type is unchanged" )
            {
                REQUIRE( dsl_pipeline_streammux_tiler_add(pipelineName.c_str(), 
                    fakeSinkName.c_str()) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Pipeline Streammux API checks for NULL input parameters", "[pipeline-streammux]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        
        uint batch_size(0);
        uint width(0);

        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        
        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_get(pipelineName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_nvbuf_mem_type_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_batch_properties_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_batch_properties_get(pipelineName.c_str(), 
                    &batch_size, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_batch_properties_set(NULL, 
                    1, 1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_dimensions_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_dimensions_get(pipelineName.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_dimensions_get(pipelineName.c_str(), 
                    &batch_size, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_dimensions_set(NULL, 
                    1, 1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_padding_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_padding_get(pipelineName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_padding_set(NULL, 
                    true) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_get(pipelineName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_tiler_add(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_tiler_add(pipelineName.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_tiler_remove(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );


                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}