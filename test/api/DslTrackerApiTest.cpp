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

SCENARIO( "The Components container is updated correctly on new KTL Tracker", "[tracker-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new KTL Tracker is created" ) 
        {

            REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on KTL Tracker delete", "[tracker-api]" )
{
    GIVEN( "A new KTL Tracker in memory" ) 
    {
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new KTL Tracker is created" ) 
        {
            REQUIRE( dsl_component_delete(trackerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "The Components container is updated correctly on new IOU Tracker", "[tracker-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring trackerName(L"iou-tracker");
        std::wstring configFile(L"./test/configs/iou_config.txt");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new IOU Tracker is created" ) 
        {
            REQUIRE( dsl_tracker_iou_new(trackerName.c_str(), configFile.c_str(), 
                width, height) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}    

SCENARIO( "The Components container is updated correctly on IOU Tracker delete", "[tracker-api]" )
{
    GIVEN( "A new IOU Tracker in memory" ) 
    {
        std::wstring trackerName(L"ktl-tracker");
        std::wstring configFile(L"./test/configs/iou_config.txt");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_tracker_iou_new(trackerName.c_str(), configFile.c_str(), 
            width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new KTL Tracker is created" ) 
        {
            REQUIRE( dsl_component_delete(trackerName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker in use can't be deleted", "[tracker-api]" )
{
    GIVEN( "A new KTL Tracker and new pPipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Tracker is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                trackerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tracker can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(trackerName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
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
    GIVEN( "A new pPipeline with a child KTL Tracker" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            trackerName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Tracker is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                trackerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tracker can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(trackerName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker in use can't be added to a second Pipeline", "[tracker-api]" )
{
    GIVEN( "A new KTL Tracker and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Tracker is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                trackerName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Tracker can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    trackerName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Trackers Max Dimensions can be queried and updated", "[tracker-api]" )
{
    GIVEN( "A new KTL Tracker in memory" ) 
    {
        std::wstring trackerName(L"ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), initWidth, initHeight) == DSL_RESULT_SUCCESS );

        uint currWidth(0);
        uint currHeight(0);

        REQUIRE( dsl_tracker_max_dimensions_get(trackerName.c_str(), &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
        REQUIRE( currWidth == initWidth );
        REQUIRE( currHeight == initHeight );

        WHEN( "A the KTL Tracker's Max Dimensions are updated" ) 
        {
            uint newWidth(300);
            uint newHeight(150);
            REQUIRE( dsl_tracker_max_dimensions_set(trackerName.c_str(), newWidth, newHeight) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_tracker_max_dimensions_get(trackerName.c_str(), &currWidth, &currHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static boolean batch_meta_handler_cb1(void* batch_meta, void* user_data)
{
}
static boolean batch_meta_handler_cb2(void* batch_meta, void* user_data)
{
}
    
SCENARIO( "A Sink Pad Batch Meta Handler can be added and removed from a Tracker", "[tracker-api]" )
{
    GIVEN( "A new pPipeline with a new Tracker" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            trackerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Batch Meta Handler is added to the Tracker" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE ( dsl_tracker_batch_meta_handler_remove(trackerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED );

            REQUIRE ( dsl_tracker_batch_meta_handler_add(trackerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE ( dsl_tracker_batch_meta_handler_remove(trackerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Batch Meta Handler can be added and removed froma a Tracker", "[tracker-api]" )
{
    GIVEN( "A new pPipeline with a new Tracker" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            trackerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Batch Meta Handler is added to the Tracker" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE ( dsl_tracker_batch_meta_handler_remove(trackerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED );

            REQUIRE ( dsl_tracker_batch_meta_handler_add(trackerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Meta Batch Handler can then be removed" ) 
            {
                REQUIRE ( dsl_tracker_batch_meta_handler_remove(trackerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A second Sink Pad Meta Batch Handler can not be added to a Tracker", "[tracker-api]" )
{
    GIVEN( "A new pPipeline with a new Tracker" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            trackerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Meta Batch Handler is added to the Tracker " ) 
        {
            REQUIRE ( dsl_tracker_batch_meta_handler_add(trackerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The same Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE ( dsl_tracker_batch_meta_handler_add(trackerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_TRACKER_HANDLER_ADD_FAILED );
                
                REQUIRE ( dsl_tracker_batch_meta_handler_remove(trackerName.c_str(), DSL_PAD_SINK, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "A Source Pad Meta Batch Handler can not be added to a Tracker twice", "[tracker-api]" )
{
    GIVEN( "A new pPipeline with a new Tracker" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            trackerName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "A Source Pad Meta Batch Handler is added to the Tracker " ) 
        {
            REQUIRE ( dsl_tracker_batch_meta_handler_add(trackerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_SUCCESS );
            
            THEN( "The Sink Pad Meta Batch Handler can not be added again" ) 
            {
                REQUIRE ( dsl_tracker_batch_meta_handler_add(trackerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL)
                    == DSL_RESULT_TRACKER_HANDLER_ADD_FAILED );
                
                REQUIRE ( dsl_tracker_batch_meta_handler_remove(trackerName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
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
                
                REQUIRE( dsl_tracker_max_dimensions_get(fakeSinkName.c_str(), &width, &height) == DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER);
                REQUIRE( dsl_tracker_max_dimensions_set(fakeSinkName.c_str(), 500, 300) == DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER);

                REQUIRE ( dsl_tracker_batch_meta_handler_add(fakeSinkName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1, NULL) == DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER );
                REQUIRE ( dsl_tracker_batch_meta_handler_remove(fakeSinkName.c_str(), DSL_PAD_SRC, batch_meta_handler_cb1) == DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER );
                
//                REQUIRE ( dsl_tracker_iou_config_file_get(fakeSinkName.c_str(), &config) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );
//                REQUIRE ( dsl_tracker_iou_config_file_set(fakeSinkName.c_str(), config) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker can Enable and Disable Kitti output", "[gie-api]" )
{
    GIVEN( "A new Tracker in memory" ) 
    {
        std::wstring trackerName(L"ktl-tracker");
        uint width(480);
        uint height(272);

        REQUIRE( dsl_tracker_ktl_new(trackerName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        
        WHEN( "The Tracker's Kitti output is enabled" )
        {
            REQUIRE( dsl_tracker_kitti_output_enabled_set(trackerName.c_str(), true, L"./") == DSL_RESULT_SUCCESS );

            THEN( "The Kitti output can then be disabled" )
            {
                REQUIRE( dsl_tracker_kitti_output_enabled_set(trackerName.c_str(), false, L"") == DSL_RESULT_SUCCESS );

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

