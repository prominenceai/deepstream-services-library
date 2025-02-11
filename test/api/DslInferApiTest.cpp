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
#include "DslApi.h"

static std::wstring pipeline_name(L"test-pipeline");

static std::wstring primary_aie_name(L"primary-aie");

static std::wstring primary_gie_name(L"primary-gie");
static std::wstring primary_gie_name2(L"primary-gie-2");

static const uint frame_size(441000);
static const uint hop_size(110250);
static const std::wstring transform(L"melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,sample_rate=44100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,p2db_top_db=(float)80.0");
static const std::wstring audio_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-audio/configs/config_infer_audio_sonyc.txt");
static const std::wstring audio_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/SONYC_Audio_Classifier/sonyc_audio_classify.onnx_b2_gpu0_fp32.engine");

static std::wstring infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
static std::wstring model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine");

static std::wstring secondary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
static std::wstring secondary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18_vehicletypenet.etlt_b8_gpu0_int8.engine");
        
static uint interval(1);

static std::wstring secondary_gie_name(L"secondary-gie");
static std::wstring custom_ppm_name(L"custom-ppm");

SCENARIO( "The Components container is updated correctly on new and delete Primary AIE", "[infer-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Primary AIE is created" ) 
        {
            REQUIRE( dsl_infer_aie_primary_new(primary_aie_name.c_str(), 
                audio_infer_config_file.c_str(), 
                audio_model_engine_file.c_str(), 
                frame_size, hop_size, transform.c_str()) == DSL_RESULT_SUCCESS );
            REQUIRE( dsl_component_list_size() == 1 );

            THEN( "The same Primary AIE can be deleted correctly" ) 
            {
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}    

SCENARIO( "A Primary AIE can Set and Get its property settings correctly",  "[infer-api]" )
{
    GIVEN( "A new Primary AIE in memory" ) 
    {
        REQUIRE( dsl_infer_aie_primary_new(primary_aie_name.c_str(), 
            audio_infer_config_file.c_str(), 
            audio_model_engine_file.c_str(), 
            frame_size, hop_size, transform.c_str()) == DSL_RESULT_SUCCESS );

        uint ret_frame_size(99);
        REQUIRE( dsl_infer_aie_frame_size_get(primary_aie_name.c_str(), 
            &ret_frame_size) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_frame_size == frame_size );
        
        uint ret_hop_size(99);
        REQUIRE( dsl_infer_aie_hop_size_get(primary_aie_name.c_str(), 
            &ret_hop_size) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_hop_size == hop_size );

        const wchar_t* c_ret_transform;
        REQUIRE( dsl_infer_aie_transform_get(primary_aie_name.c_str(), 
            &c_ret_transform) == DSL_RESULT_SUCCESS );
        std::wstring ret_transform(c_ret_transform );
        REQUIRE( ret_transform == transform );
        
        WHEN( "The PrimaryGieBintr's batch-size is set" )
        {
            uint new_frame_size(99);
            REQUIRE( dsl_infer_aie_frame_size_set(primary_aie_name.c_str(), 
                new_frame_size) == DSL_RESULT_SUCCESS );

            uint new_hop_size(99);
            REQUIRE( dsl_infer_aie_hop_size_set(primary_aie_name.c_str(), 
                new_hop_size) == DSL_RESULT_SUCCESS );

            static const std::wstring new_transform(L"melsdb,fft_length=2560,hop_size=692");
            REQUIRE( dsl_infer_aie_transform_set(primary_aie_name.c_str(), 
                new_transform.c_str()) == DSL_RESULT_SUCCESS );
                
            THEN( "The correct Files are returned on get" )
            {
                REQUIRE( dsl_infer_aie_frame_size_get(primary_aie_name.c_str(), 
                    &ret_frame_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_frame_size == new_frame_size );
                
                REQUIRE( dsl_infer_aie_hop_size_get(primary_aie_name.c_str(), 
                    &ret_hop_size) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_hop_size == new_hop_size );

                REQUIRE( dsl_infer_aie_transform_get(primary_aie_name.c_str(), 
                    &c_ret_transform) == DSL_RESULT_SUCCESS );
                ret_transform = c_ret_transform;
                REQUIRE( ret_transform == new_transform );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new Primary GIE", "[infer-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Primary GIE is created" ) 
        {

            REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
                model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Primary GIE delete", "[infer-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        REQUIRE( dsl_component_list_size() == 0 );

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary GIE is deleted" ) 
        {
            REQUIRE( dsl_component_delete(primary_gie_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list and contents are updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "Only one Primary GIE can be added to a Pipeline", "[infer-api]" )
{
    GIVEN( "A two Primary GIEs and a new pPipeline" ) 
    {
        uint interval(1);

        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name2.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );
            
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A new Primary GIE is add to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                primary_gie_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "Adding a second Primary GIE to the same Pipeline fails" )
            {
                // TODO why is exception not caught????
//                REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
//                    primary_gie_name2.c_str()) == DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "A Primary GIE in use can't be deleted", "[infer-api]" )
{
    GIVEN( "A new Primary GIE and new pPipeline" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary GIE is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
                primary_gie_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary GIE can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(primary_gie_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

SCENARIO( "A Primary GIE, once removed from a Pipeline, can be deleted", "[infer-api]" )
{
    GIVEN( "A new Primary GIE owned by a new pPipeline" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipeline_name.c_str()) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pipeline_component_add(pipeline_name.c_str(), 
            primary_gie_name.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Primary GIE is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipeline_name.c_str(), 
                primary_gie_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Primary GIE can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(primary_gie_name.c_str()) == DSL_RESULT_SUCCESS );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
        REQUIRE( dsl_component_list_size() == 0 );
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    return DSL_PAD_PROBE_OK;
}
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
    return DSL_PAD_PROBE_OK;
}
    
SCENARIO( "A Sink Pad Probe Handler can be added and removed from a Primary GIE", 
    "[infer-api]" )
{
    GIVEN( "A new Primary GIE and Custom PPH" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(custom_ppm_name.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the Primary GIE" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SINK) == 
                DSL_RESULT_INFER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Primary GIE" ) 
        {
            REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_INFER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Probe Handler can be added and removed froma a Primary GIE", 
    "[infer-api]" )
{
    GIVEN( "A new Primary GIE and Custom PPH" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(custom_ppm_name.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Probe Handler is added to the Primary GIE" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SRC) == 
                DSL_RESULT_INFER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Source Pad Probe Handler is added to the Primary GIE" ) 
        {
            REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Source Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SRC) == 
                    DSL_RESULT_INFER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Sink Pad Probe Handler can be added and removed from a Secondary GIE", 
    "[infer-api]" )
{
    GIVEN( "A new Primary GIE and Custom PPH" ) 
    {
        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            primary_gie_name.c_str(), interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(custom_ppm_name.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the Primary GIE" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_infer_pph_remove(secondary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SINK) == 
                DSL_RESULT_INFER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_infer_pph_add(secondary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_infer_pph_remove(secondary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Primary GIE" ) 
        {
            REQUIRE( dsl_infer_pph_add(secondary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_infer_pph_add(secondary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SINK) == 
                    DSL_RESULT_INFER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_infer_pph_remove(secondary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Probe Handler can be added and removed froma a Secondary GIE", 
    "[infer-api]" )
{
    GIVEN( "A new Secondary GIE and Custom PPH" ) 
    {
        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            primary_gie_name.c_str(), interval) == DSL_RESULT_SUCCESS );

        REQUIRE( dsl_pph_custom_new(custom_ppm_name.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Probe Handler is added to the Secondary GIE" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_infer_pph_remove(secondary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SRC) == 
                DSL_RESULT_INFER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_infer_pph_add(secondary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE( dsl_infer_pph_remove(secondary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Source Pad Probe Handler is added to the Secondary GIE" ) 
        {
            REQUIRE( dsl_infer_pph_add(secondary_gie_name.c_str(), 
                custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Source Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_infer_pph_add(secondary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SRC) == 
                    DSL_RESULT_INFER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_infer_pph_remove(secondary_gie_name.c_str(), 
                    custom_ppm_name.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE can Enable and Disable raw layer info output",
    "[infer-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), model_engine_file.c_str(), 
            interval) == DSL_RESULT_SUCCESS );
        
        WHEN( "The Primary GIE's raw output is enabled" )
        {
            REQUIRE( dsl_infer_raw_output_enabled_set(primary_gie_name.c_str(), 
                true, L"./") == DSL_RESULT_SUCCESS );

            THEN( "The raw output can then be disabled" )
            {
                REQUIRE( dsl_infer_raw_output_enabled_set(primary_gie_name.c_str(), 
                    false, L"") == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE fails to Enable raw layer info output given a bad path",  "[infer-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );
        
        WHEN( "A bad path is constructed" )
        {
            std::wstring badPath(L"this/is/an/invalid/path");
            
            THEN( "The raw output will fail to enable" )
            {
                REQUIRE( dsl_infer_raw_output_enabled_set(primary_gie_name.c_str(), 
                    true, badPath.c_str()) == DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Secondary GIE can Set and Get its Infer Config and Model Engine Files",  "[infer-api]" )
{
    GIVEN( "A new Secondary GIE in memory" ) 
    {
        REQUIRE( dsl_infer_gie_secondary_new(secondary_gie_name.c_str(), 
            secondary_infer_config_file.c_str(), secondary_model_engine_file.c_str(), 
            primary_gie_name.c_str(), interval) == DSL_RESULT_SUCCESS );

        const wchar_t* pRetInferConfigFile;
        REQUIRE( dsl_infer_config_file_get(secondary_gie_name.c_str(), 
            &pRetInferConfigFile) == DSL_RESULT_SUCCESS );
        std::wstring retInferConfigFile(pRetInferConfigFile);
        REQUIRE( retInferConfigFile == secondary_infer_config_file );
        
        const wchar_t* pRetModelEngineFile;
        REQUIRE( dsl_infer_engine_model_engine_file_get(secondary_gie_name.c_str(), 
            &pRetModelEngineFile) == DSL_RESULT_SUCCESS );
        std::wstring retModelEngineFile(pRetModelEngineFile);
        REQUIRE( retModelEngineFile == secondary_model_engine_file );
        
        WHEN( "The SecondaryGieBintr's Infer Config File and Model Engine are set" )
        {
            std::wstring newInferConfigFile = 
                L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehiclemake.txt";
            REQUIRE( dsl_infer_config_file_set(secondary_gie_name.c_str(), 
                newInferConfigFile.c_str()) == DSL_RESULT_SUCCESS );

            std::wstring newModelEngineFile = 
                L"/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleMake/resnet18_vehiclemakenet.etlt_b8_gpu0_int8.engine";
            REQUIRE( dsl_infer_engine_model_engine_file_set(secondary_gie_name.c_str(), 
                newModelEngineFile.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The correct Files are returned on get" )
            {
                REQUIRE( dsl_infer_config_file_get(secondary_gie_name.c_str(), 
                    &pRetInferConfigFile) == DSL_RESULT_SUCCESS);
                retInferConfigFile.assign(pRetInferConfigFile);
                REQUIRE( retInferConfigFile == newInferConfigFile );
                REQUIRE( dsl_infer_engine_model_engine_file_get(secondary_gie_name.c_str(), 
                    &pRetModelEngineFile) == DSL_RESULT_SUCCESS);
                retModelEngineFile.assign(pRetModelEngineFile);
                REQUIRE( retModelEngineFile == newModelEngineFile );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE can Set and Get its tensor-meta settings correctly",  "[infer-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );

        boolean input_enabled(true), output_enabled(false);
        REQUIRE( dsl_infer_engine_tensor_meta_settings_get(primary_gie_name.c_str(), 
            &input_enabled, &output_enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( input_enabled == false );
        REQUIRE( output_enabled == false );
        
        WHEN( "The PrimaryGieBintr's tensor-meta settings are set" )
        {
            
            REQUIRE( dsl_infer_engine_tensor_meta_settings_set(primary_gie_name.c_str(), 
                true, true) == DSL_RESULT_SUCCESS );

            THEN( "The correct Files are returned on get" )
            {
                REQUIRE( dsl_infer_engine_tensor_meta_settings_get(primary_gie_name.c_str(), 
                    &input_enabled, &output_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( input_enabled == true );
                REQUIRE( output_enabled == true );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE can Set and Get its batch-size settings correctly",  "[infer-api]" )
{
    GIVEN( "A new Primary GIE in memory" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
            model_engine_file.c_str(), interval) == DSL_RESULT_SUCCESS );

        uint batch_size(99);
        REQUIRE( dsl_infer_batch_size_get(primary_gie_name.c_str(), 
            &batch_size) == DSL_RESULT_SUCCESS );
        REQUIRE( batch_size == 0 );
        
        WHEN( "The PrimaryGieBintr's batch-size is set" )
        {
            REQUIRE( dsl_infer_batch_size_set(primary_gie_name.c_str(), 
                5) == DSL_RESULT_SUCCESS );

            THEN( "The correct Files are returned on get" )
            {
                REQUIRE( dsl_infer_batch_size_get(primary_gie_name.c_str(), 
                    &batch_size) == DSL_RESULT_SUCCESS );
                REQUIRE( batch_size == 5 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Primary GIE returns its unique id correctly",  "[infer-api]" )
{
    GIVEN( "Attributes for a new Primary GIE" ) 
    {
        // Defined at the top of the file
        
        WHEN( "When a new The Primary GIE is created" )
        {
            REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), infer_config_file.c_str(), 
                NULL, interval) == DSL_RESULT_SUCCESS );


            THEN( "The correct unique id is returned" )
            {
                uint retId(0);
                REQUIRE( dsl_infer_unique_id_get(primary_gie_name.c_str(), &retId) == DSL_RESULT_SUCCESS );

                REQUIRE( retId == 1 );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static void model_update_listener_cb(const wchar_t* name,
    const wchar_t* model_engine_file, void* client_data)
{
}

SCENARIO( "A model-update-listener can be added and removed", "[infer-api]" )
{
    std::wstring pipelineName = L"test-pipeline";

    GIVEN( "A PGIE in memory" ) 
    {
        REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
            infer_config_file.c_str(), NULL, interval) == DSL_RESULT_SUCCESS );

        WHEN( "A model-update-listener is added" )
        {
            REQUIRE( dsl_infer_engine_model_update_listener_add(primary_gie_name.c_str(),
                model_update_listener_cb, (void*)0x12345678) == DSL_RESULT_SUCCESS );

            // second call must fail
            REQUIRE( dsl_infer_engine_model_update_listener_add(primary_gie_name.c_str(),
                model_update_listener_cb, (void*)0x12345678) == 
                    DSL_RESULT_INFER_CALLBACK_ADD_FAILED );

            THEN( "The same listener can be removed again" ) 
            {
                REQUIRE( dsl_infer_engine_model_update_listener_remove(
                    primary_gie_name.c_str(), model_update_listener_cb) == 
                        DSL_RESULT_SUCCESS );

                // second call must fail
                REQUIRE( dsl_infer_engine_model_update_listener_remove(
                    primary_gie_name.c_str(),model_update_listener_cb) == 
                        DSL_RESULT_INFER_CALLBACK_REMOVE_FAILED );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}    


SCENARIO( "The GIE API checks for NULL input parameters", "[infer-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        uint retId(0);
        uint batch_size(1);
        uint frame_size(1);
        uint hop_size(1);
        const wchar_t* ret_transform;

        boolean input(false), output(false);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                REQUIRE( dsl_infer_aie_primary_new(NULL, audio_infer_config_file.c_str(), 
                    audio_model_engine_file.c_str(), 
                    0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_aie_primary_new(primary_gie_name.c_str(), 
                    NULL, audio_model_engine_file.c_str(), 
                    0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_aie_primary_new(primary_gie_name.c_str(), 
                    audio_infer_config_file.c_str(), audio_model_engine_file.c_str(), 
                    0, 0, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                    
                REQUIRE( dsl_infer_aie_frame_size_get(NULL, &frame_size) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_aie_frame_size_get(primary_gie_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_aie_frame_size_set(NULL, frame_size) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );    
                REQUIRE( dsl_infer_aie_hop_size_get(NULL, &hop_size) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_aie_hop_size_get(primary_gie_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_aie_hop_size_set(NULL, hop_size) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );    
                REQUIRE( dsl_infer_aie_transform_get(NULL, &ret_transform) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_aie_transform_get(primary_gie_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_aie_transform_set(NULL, transform.c_str()) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );    

                REQUIRE( dsl_infer_gie_primary_new(NULL, infer_config_file.c_str(), 
                    model_engine_file.c_str(), 1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
                    NULL, model_engine_file.c_str(), 
                    1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_gie_secondary_new(NULL, infer_config_file.c_str(), 
                    model_engine_file.c_str(), primary_gie_name.c_str(), 1) == 
                        DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_gie_secondary_new(primary_gie_name.c_str(), 
                    NULL, model_engine_file.c_str(), 
                    primary_gie_name.c_str(), 1) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_gie_secondary_new(primary_gie_name.c_str(), 
                    infer_config_file.c_str(), 
                    model_engine_file.c_str(), NULL, 1) == 
                        DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_pph_add(NULL, NULL, 
                    DSL_PAD_SRC) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_pph_add(primary_gie_name.c_str(), NULL, 
                    DSL_PAD_SRC) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_pph_remove(NULL, NULL, 
                    DSL_PAD_SRC) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_pph_remove(primary_gie_name.c_str(), NULL, 
                    DSL_PAD_SRC) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_infer_config_file_get(NULL, 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_config_file_get(primary_gie_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_config_file_set(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_config_file_set(primary_gie_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_engine_file_get(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_engine_file_get(primary_gie_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_engine_file_set(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_engine_file_set(primary_gie_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_video_interval_get(NULL, &interval) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_video_interval_get(primary_gie_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_video_interval_set(NULL, interval) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );    

                REQUIRE( dsl_infer_batch_size_get(NULL, &batch_size) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_batch_size_get(primary_gie_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_batch_size_set(NULL, batch_size) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );    

                REQUIRE( dsl_infer_engine_tensor_meta_settings_get(NULL, &input, &output) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_tensor_meta_settings_get(primary_gie_name.c_str(), 
                    NULL, &output) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_tensor_meta_settings_get(primary_gie_name.c_str(), 
                    &input, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_tensor_meta_settings_set(NULL, input, output) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                

                REQUIRE( dsl_infer_unique_id_get(NULL, &retId) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                

                REQUIRE( dsl_infer_engine_model_update_listener_add(NULL, NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_update_listener_add(primary_gie_name.c_str(), 
                    NULL, NULL) ==  DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_update_listener_remove(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );                
                REQUIRE( dsl_infer_engine_model_update_listener_remove(primary_gie_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );                


                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

