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

static const std::wstring primary_gie_name(L"primary-gie");
static std::wstring primary_infer_config_file(
    L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt");
static std::wstring primary_model_engine_file(
    L"/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/resnet10.caffemodel_b8_gpu0_int8.engine");

SCENARIO( "The Components container is updated correctly on new Tracker", "[tracker-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new IOU Tracker is created" ) 
        {
            REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
                width, height) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on Tracker delete", "[tracker-api]" )
{
    GIVEN( "A new IOU Tracker in memory" ) 
    {
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new IOU Tracker is created" ) 
        {
            REQUIRE( dsl_component_delete(tracker_name.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker in use can't be deleted", "[tracker-api]" )
{
    GIVEN( "A new IOU Tracker and new Pipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Tracker is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                tracker_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tracker can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(tracker_name.c_str()) == 
                    DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker, once removed from a Pipeline, can be deleted", "[tracker-api]" )
{
    GIVEN( "A new pipeline with a child IOU Tracker" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            tracker_name.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Tracker is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                tracker_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tracker can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(tracker_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker in use can't be added to a second Pipeline", "[tracker-api]" )
{
    GIVEN( "A new IOU Tracker and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Tracker is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                tracker_name.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tracker can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    tracker_name.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Trackers Max Dimensions can be queried and updated", "[tracker-api]" )
{
    GIVEN( "A new IOU Tracker in memory" ) 
    {
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        uint currWidth(0);
        uint currHeight(0);

        REQUIRE( dsl_tracker_dimensions_get(tracker_name.c_str(), 
            &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
        REQUIRE( currWidth == width );
        REQUIRE( currHeight == height );

        WHEN( "The Tracker's Max Dimensions are updated" ) 
        {
            uint newWidth(300);
            uint newHeight(150);
            REQUIRE( dsl_tracker_dimensions_set(tracker_name.c_str(), 
                newWidth, newHeight) == DSL_RESULT_SUCCESS );

            THEN( "The correct dimensios are returned on get" ) 
            {
                REQUIRE( dsl_tracker_dimensions_get(tracker_name.c_str(), 
                    &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Trackers id-display-enabled can be queried and updated", "[tracker-api]" )
{
    GIVEN( "A new IOU Tracker in memory" ) 
    {
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        boolean ret_id_display_enabled(0);

        REQUIRE( dsl_tracker_id_display_enabled_get(tracker_name.c_str(), 
            &ret_id_display_enabled) == DSL_RESULT_SUCCESS );
        REQUIRE( ret_id_display_enabled == 1 );

        WHEN( "The Tracker's id-display-enabled setting is updated" ) 
        {
            boolean new_id_display_enabled(false);
            REQUIRE( dsl_tracker_id_display_enabled_set(tracker_name.c_str(), 
                new_id_display_enabled) == DSL_RESULT_SUCCESS );

            THEN( "The correct value is returned on get" ) 
            {
                REQUIRE( dsl_tracker_id_display_enabled_get(tracker_name.c_str(), 
                    &ret_id_display_enabled) == DSL_RESULT_SUCCESS );
                REQUIRE( ret_id_display_enabled == new_id_display_enabled );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A  Tracker can update its tensor-meta-settings correctly", "[tracker-api]" )
{
    GIVEN( "A new Tracker in memory" ) 
    {
        std::wstring tracker_name(L"tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        // check defaults first.
        
        boolean ret_input_enabled(true);
        const wchar_t* c_ret_track_on_gie;
        
        REQUIRE( dsl_tracker_tensor_meta_settings_get(tracker_name.c_str(), 
            &ret_input_enabled, &c_ret_track_on_gie) == DSL_RESULT_SUCCESS );
            
        REQUIRE( ret_input_enabled == 0 );

        std::wstring ret_track_on_gie(c_ret_track_on_gie);
        REQUIRE( ret_track_on_gie == L"" );

        WHEN( "Using an invalid GIE name for track-on-gie" ) 
        {
            boolean new_input_enabled(true);
            std::wstring new_track_on_gie(L"invalid-gie-name");
            
            THEN( "The call to set the tensor-settings will fail" )
            {
                REQUIRE( dsl_tracker_tensor_meta_settings_set(tracker_name.c_str(),
                    new_input_enabled, new_track_on_gie.c_str()) == 
                        DSL_RESULT_TRACKER_SET_FAILED );
                
                REQUIRE( dsl_component_delete(tracker_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
        WHEN( "Using a valid GIE name for track-on-gie" ) 
        {
            REQUIRE( dsl_infer_gie_primary_new(primary_gie_name.c_str(), 
                primary_infer_config_file.c_str(), primary_model_engine_file.c_str(), 
                0) == DSL_RESULT_SUCCESS );
                
            boolean new_input_enabled(true);

            REQUIRE( dsl_tracker_tensor_meta_settings_set(tracker_name.c_str(),
                new_input_enabled, primary_gie_name.c_str()) == 
                    DSL_RESULT_SUCCESS );
            
            THEN( "The correct values are returned on get" )
            {
                REQUIRE( dsl_tracker_tensor_meta_settings_get(tracker_name.c_str(), 
                    &ret_input_enabled, &c_ret_track_on_gie) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( ret_input_enabled == new_input_enabled );

                ret_track_on_gie = c_ret_track_on_gie;
                REQUIRE( ret_track_on_gie == primary_gie_name );
                
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
static boolean pad_probe_handler_cb2(void* buffer, void* user_data)
{
    return true;
}
    
SCENARIO( "A Sink Pad Probe Handler can be added and removed from a Tracker", "[tracker-api]" )
{
    GIVEN( "A new Tracker and Custom PPH" ) 
    {
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the Tracker" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), 
                customPpmName.c_str(), DSL_PAD_SINK) == 
                    DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), 
                customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Tracker" ) 
        {
            REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), 
                customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == 
                        DSL_RESULT_TRACKER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SINK) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Probe Handler can be added and removed from a Tracker", "[tracker-api]" )
{
    GIVEN( "A new Tracker and Custom PPH" ) 
    {
        std::wstring tracker_name(L"iou-tracker");
        std::wstring configFile(
            L"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_new(tracker_name.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );

        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the Tracker" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), 
                customPpmName.c_str(), DSL_PAD_SRC) == 
                    DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), 
                customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Tracker" ) 
        {
            REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), 
                customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == 
                        DSL_RESULT_TRACKER_HANDLER_ADD_FAILED );
                REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), 
                    customPpmName.c_str(), DSL_PAD_SRC) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}


SCENARIO( "An invalid Tracker is caught by all Set and Get API calls", "[tracker-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Tracker Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");

        WHEN( "The Tracker Get-Set API called with a Fake sink" )
        {
            
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The Trakcer Get-Set APIs fail correctly")
            {
                uint width(0), height(0);
                const wchar_t* config;
                
                REQUIRE( dsl_tracker_dimensions_get(fakeSinkName.c_str(), &width, &height) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);
                REQUIRE( dsl_tracker_dimensions_set(fakeSinkName.c_str(), 500, 300) == 
                    DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Tracker API checks for NULL input parameters", "[tracker-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring tracker_name  = L"test-tracker";
        std::wstring otherName  = L"other";
        
        uint width(0), height(0);
        boolean enabled;
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                
                REQUIRE( dsl_tracker_new(NULL, NULL, 0,  0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tracker_lib_file_get(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_lib_file_get(tracker_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_lib_file_set(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_lib_file_set(tracker_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tracker_config_file_get(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_config_file_get(tracker_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_config_file_set(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_config_file_set(tracker_name.c_str(), NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tracker_dimensions_get(NULL, &width, &height) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_dimensions_get(tracker_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_dimensions_get(tracker_name.c_str(), 
                    &width, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_dimensions_set(NULL, width, height) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tracker_id_display_enabled_get(NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_id_display_enabled_get(tracker_name.c_str(), 
                    NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_id_display_enabled_set(NULL, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tracker_tensor_meta_settings_get(NULL, NULL, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_tensor_meta_settings_get(tracker_name.c_str(), 
                    NULL, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_tensor_meta_settings_get(tracker_name.c_str(), 
                    &enabled, NULL) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_tensor_meta_settings_set(NULL, 0, NULL) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_tracker_pph_add( NULL, NULL, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_pph_add(tracker_name.c_str(), NULL, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_pph_remove( NULL, NULL, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_tracker_pph_remove(tracker_name.c_str(), NULL, 0) == 
                    DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
