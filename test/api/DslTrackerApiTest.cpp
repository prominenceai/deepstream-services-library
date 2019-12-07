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

