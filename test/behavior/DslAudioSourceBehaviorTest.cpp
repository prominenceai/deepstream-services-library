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

static const std::wstring source_uri_1 = L"http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4";

static const std::wstring source_uri_2 = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring source_name_1(L"file-source-1");
static const std::wstring source_name_2(L"file-source-2");

static const std::wstring demuxer_name_1(L"demuxer-1");
static const std::wstring demuxer_name_2(L"demuxer-2");

static const std::wstring fake_sink_name(L"fake-sink");

static const std::wstring alsa_sink_name_1(L"alsa-sink-1");
static const std::wstring alsa_sink_name_2(L"alsa-sink-2");

static const std::wstring device_location(L"default");

// Window Sink name and attributes.
static const std::wstring window_sink_name_1(L"egl-sink-1");
static const std::wstring window_sink_name_2(L"egl-sink-2");
static const uint offsetX(0);
static const uint offsetY(0);
static const uint sinkW(DSL_1K_HD_WIDTH/2);
static const uint sinkH(DSL_1K_HD_HEIGHT/2);

static const std::wstring pipeline_graph_name(L"audio-behavior");


SCENARIO( "A new Pipeline with a URI Source, Window Sink, and Audio Fake Sink can play", "[audio-behavior]" )
{
    GIVEN( "A Pipeline, URI Source, Window Sink, and Audio Fake Sink" ) 
    {
        if (dsl_info_use_new_nvstreammux_get())
        {
            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_1.c_str(), 
                false, false, 0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
                DSL_MEDIA_TYPE_AUDIO_VIDEO) == DSL_RESULT_SUCCESS );          

            // New Window Sink to render the video stream. 
            REQUIRE( dsl_sink_window_egl_new(window_sink_name_1.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_fake_new(fake_sink_name.c_str()) == DSL_RESULT_SUCCESS );    
                
            REQUIRE( dsl_component_media_type_set(fake_sink_name.c_str(), 
                DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

            REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
                device_location.c_str()) == DSL_RESULT_SUCCESS );    
                
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_audiomux_enabled_set(pipeline_name.c_str(), 
                true) == DSL_RESULT_SUCCESS );
            
            WHEN( "When the Pipeline is Assembled" ) 
            {
                const wchar_t* components[] = {
                    source_name_1.c_str(), window_sink_name_1.c_str(), 
                    fake_sink_name.c_str(), alsa_sink_name_1.c_str(), NULL};
        
                REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                    components) == DSL_RESULT_SUCCESS );

                THEN( "Pipeline is Able to LinkAll and Play" )
                {
                    // REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                    dsl_pipeline_play(pipeline_name.c_str());

                    dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                        const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                    dsl_delete_all();
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "A new Pipeline with 2 URI Sources, 2 Window Sinks, and 2 ALSA Sinks can play",
    "[audio-behavior]" )
{
    GIVEN( "A Pipeline, URI Source, Window Sink, and Audio Fake Sink" ) 
    {
        if (dsl_info_use_new_nvstreammux_get())
        {
            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_1.c_str(), 
                false, false, 0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
                DSL_MEDIA_TYPE_AUDIO_VIDEO) == DSL_RESULT_SUCCESS );          

            REQUIRE( dsl_source_uri_new(source_name_2.c_str(), source_uri_2.c_str(), 
                false, false, 0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_media_type_set(source_name_2.c_str(), 
                DSL_MEDIA_TYPE_AUDIO_VIDEO) == DSL_RESULT_SUCCESS );          

            // New Window Sinks to render the video stream.s 
            REQUIRE( dsl_sink_window_egl_new(window_sink_name_1.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_window_egl_new(window_sink_name_2.c_str(),
                offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
                device_location.c_str()) == DSL_RESULT_SUCCESS );    
                
            REQUIRE( dsl_sink_alsa_new(alsa_sink_name_2.c_str(),
                device_location.c_str()) == DSL_RESULT_SUCCESS );    
                
            REQUIRE( dsl_tee_demuxer_new(demuxer_name_1.c_str(),
                2) == DSL_RESULT_SUCCESS );    
                
            REQUIRE( dsl_tee_demuxer_new(demuxer_name_2.c_str(),
                2) == DSL_RESULT_SUCCESS );    
            REQUIRE( dsl_component_media_type_set(demuxer_name_2.c_str(), 
                DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          
                
            REQUIRE( dsl_tee_branch_add(demuxer_name_1.c_str(), 
                window_sink_name_1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxer_name_1.c_str(), 
                window_sink_name_2.c_str()) == DSL_RESULT_SUCCESS );
                  
            REQUIRE( dsl_tee_branch_add(demuxer_name_2.c_str(), 
                alsa_sink_name_1.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_tee_branch_add(demuxer_name_2.c_str(), 
                alsa_sink_name_2.c_str()) == DSL_RESULT_SUCCESS );
                  
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_audiomux_enabled_set(pipeline_name.c_str(), 
                true) == DSL_RESULT_SUCCESS );
            
            WHEN( "When the Pipeline is Assembled" ) 
            {
                const wchar_t* components[] = {
                    source_name_1.c_str(), source_name_2.c_str(), 
                    demuxer_name_1.c_str(), demuxer_name_2.c_str(), NULL};
        
                REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                    components) == DSL_RESULT_SUCCESS );

                THEN( "Pipeline is Able to LinkAll and Play" )
                {
                    // REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                    dsl_pipeline_play(pipeline_name.c_str());

                    dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                        const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                    dsl_delete_all();
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}

SCENARIO( "A new Pipeline with a URI Source and ALSA Sink can play - Audio Only",
    "[error]" )
{
    GIVEN( "A Pipeline, URI Source, Window Sink, and Audio Fake Sink" ) 
    {
        if (dsl_info_use_new_nvstreammux_get())
        {
            REQUIRE( dsl_component_list_size() == 0 );

            REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_2.c_str(), 
                false, false, 0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
                DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

            REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
                device_location.c_str()) == DSL_RESULT_SUCCESS );    
                
            REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_audiomux_enabled_set(pipeline_name.c_str(), 
                true) == DSL_RESULT_SUCCESS );
            
            REQUIRE( dsl_pipeline_videomux_enabled_set(pipeline_name.c_str(), 
                false) == DSL_RESULT_SUCCESS );
            
            WHEN( "When the Pipeline is Assembled" ) 
            {
                const wchar_t* components[] = {
                    source_name_1.c_str(), alsa_sink_name_1.c_str(), NULL};
        
                REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                    components) == DSL_RESULT_SUCCESS );

                THEN( "Pipeline is Able to LinkAll and Play" )
                {
                    // REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
                    dsl_pipeline_play(pipeline_name.c_str());

                    dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                        const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                    std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                    REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                    dsl_delete_all();
                    REQUIRE( dsl_pipeline_list_size() == 0 );
                    REQUIRE( dsl_component_list_size() == 0 );
                }
            }
        }
    }
}
