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

// ---------------------------------------------------------------------------
// Shared Test Inputs 

static const std::wstring source_name(L"uri-source");
static const std::wstring uri(L"//opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov");
static const uint intr_decode(false);
static const uint drop_frame_interval(0);

static const std::wstring preproc_name(L"preprocessor");

static const std::wstring preproc_config1(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-action-recognition/config_preprocess_2d_custom.txt");

static const std::wstring preproc_config2(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-action-recognition/config_preprocess_3d_custom.txt");

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-action-recognition/config_infer_primary_3d_action.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static const std::wstring osd_name(L"osd");
static const boolean text_enabled(true);
static const boolean clock_enabled(false);
static const boolean bbox_enabled(true);
static const boolean mask_enabled(false);

static const std::wstring window_sink_name(L"window-sink");
static const uint offsetX(100);
static const uint offsetY(140);
static const uint sinkW(1280);
static const uint sinkH(720);

static const std::wstring pipeline_name(L"test-pipeline");

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(2000)

// ---------------------------------------------------------------------------


SCENARIO( "A new Pipeline with a URI source, Preprocessor, Primary GIE, OSD, and Window Sink can Play",
    "[action-recognition-behavior]" )
{
    GIVEN( "A Pipeline, URI source, KTL Tracker, Primary GIE, Tiled Display, ODE Hander, and Overlay Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name.c_str(), uri.c_str(), 
            false, intr_decode, drop_frame_interval) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_preproc_new(preproc_name.c_str(), 
            preproc_config1.c_str(), true) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), 0) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_sink_window_new(window_sink_name.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        const wchar_t* components[] = {L"uri-source", L"preprocessor", L"primary-gie", L"window-sink", NULL};
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), components) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR*10);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

