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

SCENARIO( "The Pipeline's Audiomixer can be enabled and disabled", 
    "[pipeline-audiomix]" )
{
    GIVEN( "A new Pipeline with its Audiomixer disabled by default" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        boolean enabled(TRUE);
        
        REQUIRE( dsl_pipeline_audiomix_enabled_get(pipeline_name.c_str(), 
            &enabled)  == DSL_RESULT_SUCCESS );
        REQUIRE( enabled == FALSE );
        
        WHEN( "The Pipeline's Audiomixer is enabled" ) 
        {
            REQUIRE( dsl_pipeline_audiomix_enabled_set(
                pipeline_name.c_str(), TRUE) == DSL_RESULT_SUCCESS );

            THEN( "The correct enabled setting is returned" )
            {
                enabled = FALSE;
                REQUIRE( dsl_pipeline_audiomix_enabled_get(pipeline_name.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == TRUE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
        WHEN( "The Pipeline's Audiomuxer is enabled first" ) 
        {
            REQUIRE( dsl_pipeline_audiomux_enabled_set(
                pipeline_name.c_str(), TRUE) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline's Audiomixer will fail to enable" )
            {
                REQUIRE( dsl_pipeline_audiomix_enabled_set(
                    pipeline_name.c_str(), TRUE) == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );

                // Must return enabled == FALSE
                REQUIRE( dsl_pipeline_audiomix_enabled_get(pipeline_name.c_str(), 
                    &enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( enabled == FALSE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Pipeline's Audiomixer can set and get its mute-enabled settings correctly", 
    "[pipeline-audiomix]" )
{
    GIVEN( "A new Pipeline with multiple Audio Sources" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        std::wstring alsa_source_name_1  = L"test-source-1";
        std::wstring alsa_source_name_2  = L"test-source-2";
        std::wstring device_location(L"default");

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_1.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_2.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_audiomix_enabled_set(
            pipeline_name.c_str(), TRUE) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {
            alsa_source_name_1.c_str(), alsa_source_name_2.c_str(), NULL};

        REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        boolean mute(TRUE);
        REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(
            pipeline_name.c_str(), alsa_source_name_1.c_str(), 
            &mute) == DSL_RESULT_SUCCESS );
        REQUIRE( mute == FALSE );

        mute = TRUE;
        REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(
            pipeline_name.c_str(), alsa_source_name_2.c_str(), 
            &mute) == DSL_RESULT_SUCCESS );
        REQUIRE( mute == FALSE );

        WHEN( "The Pipeline Audiomixer's mute-enabled settings are updated" ) 
        {
            REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                pipeline_name.c_str(), alsa_source_name_1.c_str(), 
                TRUE) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(
                pipeline_name.c_str(), alsa_source_name_2.c_str(), 
                TRUE) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer sync-inputs  is returned" )
            {
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(
                    pipeline_name.c_str(), alsa_source_name_1.c_str(), 
                    &mute) == DSL_RESULT_SUCCESS );
                REQUIRE( mute == TRUE );

                mute = TRUE;
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(
                    pipeline_name.c_str(), alsa_source_name_2.c_str(), 
                    &mute) == DSL_RESULT_SUCCESS );
                REQUIRE( mute == TRUE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Pipeline's Audiomixer can set many mute-enabled settings correctly", 
    "[pipeline-audiomix]" )
{
    GIVEN( "A new Pipeline with multiple Audio Sources" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        std::wstring alsa_source_name_1  = L"test-source-1";
        std::wstring alsa_source_name_2  = L"test-source-2";
        std::wstring device_location(L"default");

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_1.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_2.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_audiomix_enabled_set(
            pipeline_name.c_str(), TRUE) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {
            alsa_source_name_1.c_str(), alsa_source_name_2.c_str(), NULL};

        REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "The Pipeline Audiomixer's mute-enabled settings are updated" ) 
        {
            REQUIRE( dsl_pipeline_audiomix_mute_enabled_set_many(
                pipeline_name.c_str(), components, 
                TRUE) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer sync-inputs  is returned" )
            {
                boolean mute(TRUE);

                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(
                    pipeline_name.c_str(), alsa_source_name_1.c_str(), 
                    &mute) == DSL_RESULT_SUCCESS );
                REQUIRE( mute == TRUE );

                mute = TRUE;
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(
                    pipeline_name.c_str(), alsa_source_name_2.c_str(), 
                    &mute) == DSL_RESULT_SUCCESS );
                REQUIRE( mute == TRUE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Pipeline's Audiomixer can set and get its volume settings correctly", 
    "[pipeline-audiomix]" )
{
    GIVEN( "A new Pipeline with multiple Audio Sources" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        std::wstring alsa_source_name_1  = L"test-source-1";
        std::wstring alsa_source_name_2  = L"test-source-2";
        std::wstring device_location(L"default");

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_1.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_2.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_audiomix_enabled_set(
            pipeline_name.c_str(), TRUE) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {
            alsa_source_name_1.c_str(), alsa_source_name_2.c_str(), NULL};

        REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        double volume(0.0);
        REQUIRE( dsl_pipeline_audiomix_volume_get(
            pipeline_name.c_str(), alsa_source_name_1.c_str(), 
            &volume) == DSL_RESULT_SUCCESS );
        REQUIRE( volume == 1.0 );

        volume = 0.0;
        REQUIRE( dsl_pipeline_audiomix_volume_get(
            pipeline_name.c_str(), alsa_source_name_2.c_str(), 
            &volume) == DSL_RESULT_SUCCESS );
        REQUIRE( volume == 1.0 );

        WHEN( "The Pipeline Audiomixer's volume settings are updated" ) 
        {
            REQUIRE( dsl_pipeline_audiomix_volume_set(
                pipeline_name.c_str(), alsa_source_name_1.c_str(), 
                5.0) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_audiomix_volume_set(
                pipeline_name.c_str(), alsa_source_name_2.c_str(), 
                5.0) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer sync-inputs  is returned" )
            {
                REQUIRE( dsl_pipeline_audiomix_volume_get(
                    pipeline_name.c_str(), alsa_source_name_1.c_str(), 
                    &volume) == DSL_RESULT_SUCCESS );
                REQUIRE( volume == 5.0 );

                volume = TRUE;
                REQUIRE( dsl_pipeline_audiomix_volume_get(
                    pipeline_name.c_str(), alsa_source_name_2.c_str(), 
                    &volume) == DSL_RESULT_SUCCESS );
                REQUIRE( volume == 5.0 );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Pipeline's Audiomixer can set many volume settings correctly", 
    "[pipeline-audiomix]" )
{
    GIVEN( "A new Pipeline with multiple Audio Sources" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        std::wstring alsa_source_name_1  = L"test-source-1";
        std::wstring alsa_source_name_2  = L"test-source-2";
        std::wstring device_location(L"default");

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_1.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );
        
        REQUIRE( dsl_source_alsa_new(alsa_source_name_2.c_str(), 
            device_location.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_audiomix_enabled_set(
            pipeline_name.c_str(), TRUE) == DSL_RESULT_SUCCESS );
        
        const wchar_t* components[] = {
            alsa_source_name_1.c_str(), alsa_source_name_2.c_str(), NULL};

        REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
            components) == DSL_RESULT_SUCCESS );
        
        WHEN( "The Pipeline Audiomixer's volume settings are updated" ) 
        {
            REQUIRE( dsl_pipeline_audiomix_volume_set_many(
                pipeline_name.c_str(), components, 
                5.0) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer sync-inputs  is returned" )
            {
                double volume(0.0);

                REQUIRE( dsl_pipeline_audiomix_volume_get(
                    pipeline_name.c_str(), alsa_source_name_1.c_str(), 
                    &volume) == DSL_RESULT_SUCCESS );
                REQUIRE( volume == 5.0 );

                volume = TRUE;
                REQUIRE( dsl_pipeline_audiomix_volume_get(
                    pipeline_name.c_str(), alsa_source_name_2.c_str(), 
                    &volume) == DSL_RESULT_SUCCESS );
                REQUIRE( volume == 5.0 );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Pipeline Audiomixer API checks for NULL input parameters", 
    "[pipeline-audiomix]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        std::wstring alsa_source_name  = L"test-source";
        
        uint batch_size(0);
        uint width(0);

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_pipeline_audiomix_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_enabled_get(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_enabled_set(NULL, 
                    TRUE) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(pipeline_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_get(pipeline_name.c_str(), 
                    alsa_source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(NULL, 
                    NULL, FALSE) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set(pipeline_name.c_str(), 
                    NULL, FALSE) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set_many(NULL, 
                    NULL, FALSE) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_mute_enabled_set_many(pipeline_name.c_str(), 
                    NULL, FALSE) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_pipeline_audiomix_volume_get(NULL, 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_volume_get(pipeline_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_volume_get(pipeline_name.c_str(), 
                    alsa_source_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_volume_set(NULL, 
                    NULL, 0.0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_volume_set(pipeline_name.c_str(), 
                    NULL, 0.0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_volume_set_many(NULL, 
                    NULL, 0.0) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_audiomix_volume_set_many(pipeline_name.c_str(), 
                    NULL, 0.0) == DSL_RESULT_INVALID_INPUT_PARAM );
            }
        }
    }
}
