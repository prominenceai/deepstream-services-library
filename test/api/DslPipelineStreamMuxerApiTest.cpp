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

SCENARIO( "A Pipeline's Streammuxer can set its config-file correctly", "[pipeline-streammux]" )
{
    GIVEN( "A Pipeline with its Streammuxer" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        // Ensure default == empty string
        const wchar_t* c_ret_config_file;
        REQUIRE( dsl_pipeline_streammux_config_file_get(pipeline_name.c_str(), 
            &c_ret_config_file) == DSL_RESULT_SUCCESS );
            
        std::wstring ret_config_file(c_ret_config_file);
        REQUIRE( ret_config_file == L"" );

        WHEN( "The Pipeline's Streammuxer is called to update its config-file" ) 
        {
            std::wstring new_config_file(L"./test/config/all_sources_30fps.txt");
            
            REQUIRE( dsl_pipeline_streammux_config_file_set(pipeline_name.c_str(), 
                new_config_file.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct config-file is returned on get" ) 
            {
                REQUIRE( dsl_pipeline_streammux_config_file_get(pipeline_name.c_str(), 
                    &c_ret_config_file) == DSL_RESULT_SUCCESS );
                    
                ret_config_file = c_ret_config_file;
                REQUIRE( ret_config_file == new_config_file );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "The Streammuxer service is called with an invalid config-file" ) 
        {
            std::wstring new_config_file(L"./bad/path/config.txt");
            
            REQUIRE( dsl_pipeline_streammux_config_file_set(pipeline_name.c_str(), 
                new_config_file.c_str()) == 
                DSL_RESULT_PIPELINE_STREAMMUX_CONFIG_FILE_NOT_FOUND );

            THEN( "The unset config-file is returned on get" ) 
            {
                REQUIRE( dsl_pipeline_streammux_config_file_get(pipeline_name.c_str(), 
                    &c_ret_config_file) == DSL_RESULT_SUCCESS );
                    
                ret_config_file = c_ret_config_file;
                REQUIRE( ret_config_file == L"" );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

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

        std::wstring windowSinkName = L"egl-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1920);
        uint sinkH(720);

        std::wstring pipeline_name  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName3.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {L"test-uri-source-1", L"test-uri-source-2", L"test-uri-source-3", 
            L"tiler", L"egl-sink", NULL};

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        uint batch_size(0); 
        int batch_timeout(0);
        
        REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
            &batch_size) == DSL_RESULT_SUCCESS );
        REQUIRE( batch_size == 0 );
        
        WHEN( "The Pipeline's Stream Muxer Batch Size is set to more than the number of sources" ) 
        {
            uint new_batch_size(6); 
            REQUIRE( dsl_pipeline_streammux_batch_size_set(pipeline_name.c_str(), 
                new_batch_size) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
                &batch_size) == DSL_RESULT_SUCCESS );
            REQUIRE( batch_size == new_batch_size );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The updated Stream Muxer Batch Size is used" )
            {
                REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
                    &batch_size) == DSL_RESULT_SUCCESS );
                REQUIRE( batch_size == new_batch_size );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

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

        std::wstring windowSinkName = L"egl-sink";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(1920);
        uint sinkH(720);

        std::wstring pipeline_name  = L"test-pipeline";
        
        REQUIRE( dsl_component_list_size() == 0 );

        // create for of the same types of source
        REQUIRE( dsl_source_uri_new(sourceName1.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName2.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_uri_new(sourceName3.c_str(), uri.c_str(), 
            false, intrDecode, dropFrameInterval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_sink_window_egl_new(windowSinkName.c_str(),
            offsetX, offsetY, sinkW, sinkH) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_tiler_new(tilerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
            
        const wchar_t* components[] = {L"test-uri-source-1", 
            L"test-uri-source-2", L"test-uri-source-3", 
            L"tiler", L"egl-sink", NULL};

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        uint batch_size(0); 
        
        REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
            &batch_size) == DSL_RESULT_SUCCESS );
        REQUIRE( batch_size == 0 );
        
        WHEN( "The Pipeline's Stream Muxer Batch Size is set to less than the number of sources" ) 
        {
            uint new_batch_size(1);
            REQUIRE( dsl_pipeline_streammux_batch_size_set(pipeline_name.c_str(), 
                new_batch_size) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
                &batch_size) == DSL_RESULT_SUCCESS );
            REQUIRE( batch_size == new_batch_size );
        
            REQUIRE( dsl_pipeline_component_add_many(pipeline_name.c_str(), 
                components) == DSL_RESULT_SUCCESS );

            REQUIRE( dsl_pipeline_play(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The updated Stream Muxer Batch Size is used" )
            {
                REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
                    &batch_size) == DSL_RESULT_SUCCESS );
                REQUIRE( batch_size == new_batch_size );
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( dsl_pipeline_stop(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The attach-sys-ts property for a Pipeline's Streammuxer can be read and updated", 
    "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        boolean ret_attach_sys_ts(FALSE);
        
        REQUIRE( dsl_pipeline_streammux_attach_sys_ts_enabled_get(pipeline_name.c_str(), 
            &ret_attach_sys_ts)  == DSL_RESULT_SUCCESS );
        REQUIRE( ret_attach_sys_ts == TRUE );
        
        WHEN( "The Pipeline's Streammuxer's attach-sys-ts is updated" ) 
        {
            boolean new_attach_sys_ts(FALSE);

            REQUIRE( dsl_pipeline_streammux_attach_sys_ts_enabled_set(
                pipeline_name.c_str(), new_attach_sys_ts) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer attach-sys-ts  is returned" )
            {
                REQUIRE( dsl_pipeline_streammux_attach_sys_ts_enabled_get(pipeline_name.c_str(), 
                    &ret_attach_sys_ts) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_attach_sys_ts == new_attach_sys_ts );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The sync-inputs property for a Pipeline's Streammuxer can be read and updated", 
    "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        boolean ret_sync_inputs(TRUE);
        
        REQUIRE( dsl_pipeline_streammux_sync_inputs_enabled_get(pipeline_name.c_str(), 
            &ret_sync_inputs)  == DSL_RESULT_SUCCESS );
        REQUIRE( ret_sync_inputs == FALSE );
        
        WHEN( "The Pipeline's Streammuxer's sync-inputs is updated" ) 
        {
            boolean new_sync_inputs(TRUE);

            REQUIRE( dsl_pipeline_streammux_sync_inputs_enabled_set(
                pipeline_name.c_str(), new_sync_inputs) == DSL_RESULT_SUCCESS );

            THEN( "The updated Streammuxer sync-inputs  is returned" )
            {
                REQUIRE( dsl_pipeline_streammux_sync_inputs_enabled_get(pipeline_name.c_str(), 
                    &ret_sync_inputs) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_sync_inputs == new_sync_inputs );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The num-surfaces-per-frame setting for a Pipeline's Streammuxer can be read and updated", 
    "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        uint ret_num_surfaces(99);
        
        REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_get(
            pipeline_name.c_str(), &ret_num_surfaces)  == DSL_RESULT_SUCCESS );
        REQUIRE( ret_num_surfaces == 1 );
        
        WHEN( "The Pipeline's Streammuxer's num-surfaces-per-frame setting is updated" ) 
        {
            uint new_num_surfaces(4);

            REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_set(
                pipeline_name.c_str(), new_num_surfaces) == DSL_RESULT_SUCCESS );

            THEN( "The updated value is returned on get" )
            {
                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_get(
                    pipeline_name.c_str(), &ret_num_surfaces) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_num_surfaces == new_num_surfaces );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The max-latency setting for a Pipeline's Streammuxer can be read and updated", 
    "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        uint ret_max_latency(99);
        
        REQUIRE( dsl_pipeline_streammux_max_latency_get(
            pipeline_name.c_str(), &ret_max_latency)  == DSL_RESULT_SUCCESS );
        REQUIRE( ret_max_latency == 0 );
        
        WHEN( "The Pipeline's Streammuxer's max-latency setting is updated" ) 
        {
            uint new_max_latency(12345678);

            REQUIRE( dsl_pipeline_streammux_max_latency_set(
                pipeline_name.c_str(), new_max_latency) == DSL_RESULT_SUCCESS );

            THEN( "The updated value is returned on get" )
            {
                REQUIRE( dsl_pipeline_streammux_max_latency_get(
                    pipeline_name.c_str(), &ret_max_latency) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_max_latency == new_max_latency );
                
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
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        std::wstring tilerName  = L"tiler";

        REQUIRE( dsl_tiler_new(tilerName.c_str(), DSL_1K_HD_WIDTH,
            DSL_1K_HD_HEIGHT) == DSL_RESULT_SUCCESS );

        // chech that a removal attempt after pipeline creation fails
        REQUIRE( dsl_pipeline_streammux_tiler_remove(pipeline_name.c_str()) 
            == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );
        
        WHEN( "A Tiler is added to a Pipeline's Streammuxer output" ) 
        {
            REQUIRE( dsl_pipeline_streammux_tiler_add(pipeline_name.c_str(), 
                tilerName.c_str()) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_pipeline_streammux_tiler_add(pipeline_name.c_str(), 
                tilerName.c_str()) == DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED );

            THEN( "The Tiler can be successfully removed" )
            {
                REQUIRE( dsl_pipeline_streammux_tiler_remove(pipeline_name.c_str()) 
                    == DSL_RESULT_SUCCESS );
                
                // second call must fail
                REQUIRE( dsl_pipeline_streammux_tiler_remove(pipeline_name.c_str()) 
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
                REQUIRE( dsl_pipeline_streammux_tiler_add(pipeline_name.c_str(), 
                    fakeSinkName.c_str()) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
                
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    return true;
}

SCENARIO( "A Source Pad Probe Handler can be added and removed from a Pipeline's Streammuxer", 
    "[pipeline-streammux]" )
{
    GIVEN( "A new Pipeline with its built-in streammuxer and Custom PPH" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), pad_probe_handler_cb1, 
            NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Probe Handler is added to the Pipeline's Streammuxer" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_pipeline_streammux_pph_remove(pipeline_name.c_str(), 
                customPpmName.c_str()) == 
                    DSL_RESULT_PIPELINE_STREAMMUX_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_pipeline_streammux_pph_add(pipeline_name.c_str(), 
                customPpmName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_pipeline_streammux_pph_remove(pipeline_name.c_str(), 
                    customPpmName.c_str()) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Source Pad Probe Handler is added to the Pipeline's Streammuxer" ) 
        {
            REQUIRE( dsl_pipeline_streammux_pph_add(pipeline_name.c_str(), 
                customPpmName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Source Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_pipeline_streammux_pph_add(pipeline_name.c_str(), 
                    customPpmName.c_str()) == 
                        DSL_RESULT_PIPELINE_STREAMMUX_HANDLER_ADD_FAILED );
                REQUIRE( dsl_pipeline_streammux_pph_remove(pipeline_name.c_str(), 
                    customPpmName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Pipeline Streammux API checks for NULL input parameters", "[pipeline-streammux]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring pipeline_name  = L"test-pipeline";
        
        uint batch_size(0);
        uint width(0);

        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );
        
        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_pipeline_streammux_config_file_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_config_file_get(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_config_file_set(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_config_file_set(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_batch_size_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_batch_size_get(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_batch_size_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_get(
                    pipeline_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_num_surfaces_per_frame_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_attach_sys_ts_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_attach_sys_ts_enabled_get(
                    pipeline_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_attach_sys_ts_enabled_set(NULL, 
                    true) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_sync_inputs_enabled_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_sync_inputs_enabled_get(
                    pipeline_name.c_str(), NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_sync_inputs_enabled_set(NULL, 
                    true) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_max_latency_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_max_latency_get(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_max_latency_set(NULL, 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_tiler_add(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_tiler_add(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_tiler_remove(NULL) 
                    == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_streammux_pph_add(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_pph_add(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_pph_remove(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_pipeline_streammux_pph_remove(pipeline_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}