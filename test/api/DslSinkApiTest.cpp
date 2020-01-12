/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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

SCENARIO( "The Components container is updated correctly on new Overlay Sink", "[overlay-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Overlay Sink is created" ) 
        {

            REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Overlay Sink delete", "[overlay-sink-api]" )
{
    GIVEN( "An Overlay Sink Component" ) 
    {
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A new Overlay Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be deleted", "[overlay-sink-api]" )
{
    GIVEN( "A new Overlay Sink and new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring overlaySinkName = L"overlay-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Overlay Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Overlay Sink can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Overlay Sink, once removed from a Pipeline, can be deleted", "[overlay-sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring overlaySinkName = L"overlay-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Overlay Sink is removed the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Overlay Sink can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Overlay Sink in use can't be added to a second Pipeline", "[overlay-sink-api]" )
{
    GIVEN( "A new Overlay Sink and two new Pipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring overlaySinkName = L"overlay-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(overlaySinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Overlay Sink is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                overlaySinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Overlay Sink can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    overlaySinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Window Sink", "[window-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Window Sink is created" ) 
        {

            REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Window Sink delete", "[window-sink-api]" )
{
    GIVEN( "An Window Sink Component" ) 
    {
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        WHEN( "A new Window Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink in use can't be deleted", "[window-sink-api]" )
{
    GIVEN( "A new Window Sink and new Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";

        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Window Sink is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink, once removed from a Pipeline, can be deleted", "[window-sink-api]" )
{
    GIVEN( "A new Sink owned by a new pPipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink is removed the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Window Sink in use can't be added to a second Pipeline", "[window-sink-api]" )
{
    GIVEN( "A new Window Sink and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring windowSinkName = L"window-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1280);
        uint sinkH(720);

        REQUIRE( dsl_sink_overlay_new(windowSinkName.c_str(), 
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Window Sink is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                windowSinkName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Window Sink can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    windowSinkName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new File Sink", "[file-sink-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint muxer(DSL_MUXER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new File Sink is created" ) 
        {
            REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
                codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size is updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on File Sink delete", "[file-sink-api]" )
{
    GIVEN( "An Window Sink Component" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint muxer(DSL_MUXER_MPEG4);
        uint bitrate(2000000);
        uint interval(0);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            codec, muxer, bitrate, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new File Sink is deleted" ) 
        {
            REQUIRE( dsl_component_delete(fileSinkName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An File Sink's Encoder settings can be updated", "[file-sink-api]" )
{
    GIVEN( "A new File Sink" ) 
    {
        std::wstring fileSinkName(L"file-sink");
        std::wstring filePath(L"./output.mp4");
        uint codec(DSL_CODEC_H265);
        uint muxer(DSL_MUXER_MPEG4);
        uint initBitrate(2000000);
        uint initInterval(0);

        REQUIRE( dsl_sink_file_new(fileSinkName.c_str(), filePath.c_str(),
            codec, muxer, initBitrate, initInterval) == DSL_RESULT_SUCCESS );
            
        uint currBitRate(0);
        uint currInterval(0);
    
        REQUIRE( dsl_sink_file_encoder_settings_get(fileSinkName.c_str(), &currBitRate, &currInterval) == DSL_RESULT_SUCCESS);
        REQUIRE( currBitRate == initBitrate );
        REQUIRE( currInterval == initInterval );

        WHEN( "The FileSinkBintr's Encoder settings are Set" )
        {
            uint newBitRate(2500000);
            uint newInterval(10);
            
            REQUIRE( dsl_sink_file_encoder_settings_set(fileSinkName.c_str(), newBitRate, newInterval) == DSL_RESULT_SUCCESS);

            THEN( "The FileSinkBintr's new Encoder settings are returned on Get")
            {
                REQUIRE( dsl_sink_file_encoder_settings_get(fileSinkName.c_str(), &currBitRate, &currInterval) == DSL_RESULT_SUCCESS);
                REQUIRE( currBitRate == newBitRate );
                REQUIRE( currInterval == newInterval );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
