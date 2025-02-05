/*
The MIT License

Copyright (c) 2025, Prominence AI, Inc.

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

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(2000)

static const std::wstring source_uri_1 = L"http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4";

static const std::wstring source_uri_2 = L"/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4";

static const std::wstring pipeline_name(L"test-pipeline");

static const std::wstring alsa_source_name(L"alsa-source");

static const std::wstring source_name_1(L"file-source-1");
static const std::wstring source_name_2(L"file-source-2");

static const std::wstring alsa_sink_name_1(L"alsa-sink-1");
static const std::wstring alsa_sink_name_2(L"alsa-sink-2");

static const std::wstring device_location(L"default");


SCENARIO( "A new Pipeline with a URI Source, Audiomixer, and ALSA Sink can play - Audio Only",
    "[audiomix-behaviour]" )
{
    GIVEN( "A Pipeline with Audiomixer, URI Source, and Alsa Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_2.c_str(), 
            false, false, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
            DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

        REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
            device_location.c_str()) == DSL_RESULT_SUCCESS );    

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_audiomix_enabled_set(pipeline_name.c_str(), 
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
                REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                // dsl_pipeline_dump_to_dot(pipeline_name.c_str(), 
                //     const_cast<wchar_t*>(pipeline_graph_name.c_str()));

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an Audiomixer can mute and unmute a stream - Audio Only",
    "[audiomix-behaviour]" )
{
    GIVEN( "A Pipeline with Audiomixer, URI Source, and Alsa Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_2.c_str(), 
            false, false, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
            DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

        REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
            device_location.c_str()) == DSL_RESULT_SUCCESS );    

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_audiomix_enabled_set(pipeline_name.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_videomux_enabled_set(pipeline_name.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            const wchar_t* components[] = {
                source_name_1.c_str(), alsa_sink_name_1.c_str(), NULL};
    
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    TRUE) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    FALSE) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    TRUE) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an Audiomixer can mute and unmute multiple streams - Audio Only",
    "[audio]" )
{
    GIVEN( "A Pipeline with Audiomixer, URI Source, and Alsa Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_1.c_str(), 
            false, false, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
            DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

        REQUIRE( dsl_source_uri_new(source_name_2.c_str(), source_uri_2.c_str(), 
            false, false, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_media_type_set(source_name_2.c_str(), 
            DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

        REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
            device_location.c_str()) == DSL_RESULT_SUCCESS );    

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_audiomix_enabled_set(pipeline_name.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_videomux_enabled_set(pipeline_name.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            const wchar_t* components[] = {
                source_name_1.c_str(), source_name_2.c_str(), 
                alsa_sink_name_1.c_str(), NULL};
    
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            // pre-mute source 1
            REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                pipeline_name.c_str(), source_name_1.c_str(), 
                TRUE) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );


            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_2.c_str(), 
                    TRUE) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    FALSE) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    TRUE) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_2.c_str(), 
                    FALSE) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_2.c_str(), 
                    TRUE) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    FALSE) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A new Pipeline with an Audiomixer can change the volume of a stream - Audio Only",
    "[audiomix-behaviour]" )
{
    GIVEN( "A Pipeline with Audiomixer, URI Source, and Alsa Sink" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_source_uri_new(source_name_1.c_str(), source_uri_2.c_str(), 
            false, false, 0) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_component_media_type_set(source_name_1.c_str(), 
            DSL_MEDIA_TYPE_AUDIO_ONLY) == DSL_RESULT_SUCCESS );          

        REQUIRE( dsl_sink_alsa_new(alsa_sink_name_1.c_str(),
            device_location.c_str()) == DSL_RESULT_SUCCESS );    

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_audiomix_enabled_set(pipeline_name.c_str(), 
            true) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_pipeline_videomux_enabled_set(pipeline_name.c_str(), 
            false) == DSL_RESULT_SUCCESS );
        
        WHEN( "When the Pipeline is Assembled" ) 
        {
            const wchar_t* components[] = {
                source_name_1.c_str(), alsa_sink_name_1.c_str(), NULL};
    
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "Pipeline is Able to LinkAll and Play" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_volume_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    3.0) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_volume_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    5.0) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_audiomix_volume_set(
                    pipeline_name.c_str(), source_name_1.c_str(), 
                    7.0) == DSL_RESULT_SUCCESS );

                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                dsl_delete_all();
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

