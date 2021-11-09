
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
#include "Dsl.h"

SCENARIO( "The Components container is updated correctly on new Segmentation Visualizer", "[segvisual-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new Segmentation Visualizer is created" ) 
        {

            REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on Segmentation Visualizer delete", "[segvisual-api]" )
{
    GIVEN( "A new Segmentation Visualizer in memory" ) 
    {
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new Segmentation Visualizer is deleted" ) 
        {
            REQUIRE( dsl_component_delete(segvisualName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Segmentation Visualizer in use can't be deleted", "[segvisual-api]" )
{
    GIVEN( "A new Segmentation Visualizer and new Pipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The Segmentation Visualizer is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                segvisualName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Segmentation Visualizer can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(segvisualName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Segmentation Visualizer, once removed from a Pipeline, can be deleted", "[segvisual-api]" )
{
    GIVEN( "A new pPipeline with a child Segmentation Visualizer" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            segvisualName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The Segmentation Visualizer is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                segvisualName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Segmentation Visualizer can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(segvisualName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "An Segmentation Visualizer in use can't be added to a second Pipeline", "[segvisual-api]" )
{
    GIVEN( "A new Segmentation Visualizer and two new pPipelines" ) 
    {
        std::wstring pipelineName1(L"test-pipeline-1");
        std::wstring pipelineName2(L"test-pipeline-2");
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName1.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_new(pipelineName2.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "The Segmentation Visualizer is added to the first Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName1.c_str(), 
                segvisualName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Segmentation Visualizer can't be added to the second Pipeline" ) 
            {
                REQUIRE( dsl_pipeline_component_add(pipelineName2.c_str(), 
                    segvisualName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "An invalid Segmentation Visualizer is caught by all Set and Get API calls", "[segvisual-api]" )
{
    GIVEN( "A new Fake Sink as incorrect Segmentation Visualizer Type" ) 
    {
        std::wstring fakeSinkName(L"fake-sink");
            

        WHEN( "The Segmentation Visualizer Get-Set APIs are called with a Fake sink" )
        {
            REQUIRE( dsl_sink_fake_new(fakeSinkName.c_str()) == DSL_RESULT_SUCCESS);

            THEN( "The Segmentation Visualizer Get-Set APIs fail correctly")
            {
                uint width(0), height(0);
                uint rows(0), cols(0);
                const wchar_t* config;
                
                REQUIRE( dsl_segvisual_dimensions_get(fakeSinkName.c_str(), &width, &height) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);
                REQUIRE( dsl_segvisual_dimensions_set(fakeSinkName.c_str(), 500, 300) == DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE);

                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Segmentation Visualizer can Set and Get all properties", "[segvisual-api]" )
{
    GIVEN( "A new Segmentation Visualizer" ) 
    {
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );

        WHEN( "A Segmentation Visualizer's Dimensions are Set " ) 
        {
            uint newWidth(640), newHeight(360), retWidth(0), retHeight(0);
            REQUIRE( dsl_segvisual_dimensions_set(segvisualName.c_str(), newWidth, newHeight) == DSL_RESULT_SUCCESS);
            
            THEN( "The correct values are returned on Get" ) 
            {
                REQUIRE( dsl_segvisual_dimensions_get(segvisualName.c_str(), &retWidth, &retHeight) == DSL_RESULT_SUCCESS);
                REQUIRE( retWidth == newWidth );
                REQUIRE( retHeight == newHeight );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

static boolean pad_probe_handler_cb1(void* buffer, void* user_data)
{
    return true;
}

SCENARIO( "A Source Pad Probe Handler can be added and removed froma a Segmentation Visualizer", "[segvisual-api]" )
{
    GIVEN( "A new Segmentation Visualizer and Custom PPH" ) 
    {
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);

        std::wstring customPpmName(L"custom-ppm");

        REQUIRE( dsl_segvisual_new(segvisualName.c_str(), width, height) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pph_custom_new(customPpmName.c_str(), 
            pad_probe_handler_cb1, NULL) == DSL_RESULT_SUCCESS );

        WHEN( "A Sink Pad Probe Handler is added to the Segmentation Visulizer" ) 
        {
            // Test the remove failure case first, prior to adding the handler
            REQUIRE( dsl_segvisual_pph_remove(segvisualName.c_str(), customPpmName.c_str()) 
                == DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED );

            REQUIRE( dsl_segvisual_pph_add(segvisualName.c_str(), customPpmName.c_str()) 
                == DSL_RESULT_SUCCESS );
            
            THEN( "The Padd Probe Handler can then be removed" ) 
            {
                REQUIRE( dsl_segvisual_pph_remove(segvisualName.c_str(), customPpmName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
        WHEN( "A Sink Pad Probe Handler is added to the Segmentation Visulize" ) 
        {
            REQUIRE( dsl_segvisual_pph_add(segvisualName.c_str(), customPpmName.c_str()) 
                == DSL_RESULT_SUCCESS );
            
            THEN( "Attempting to add the same Sink Pad Probe Handler twice failes" ) 
            {
                REQUIRE( dsl_segvisual_pph_add(segvisualName.c_str(), customPpmName.c_str()) 
                    == DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED );
                REQUIRE( dsl_segvisual_pph_remove(segvisualName.c_str(), customPpmName.c_str()) 
                    == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pph_delete_all() == DSL_RESULT_SUCCESS );
            }
        }
    }
}

SCENARIO( "The Segmentation Visualizer API checks for NULL input parameters", "[segvisual-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring segvisualName(L"segvisual");
        uint width(1280);
        uint height(720);
        
        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "When NULL pointers are used as input" ) 
        {
            THEN( "The API returns DSL_RESULT_INVALID_INPUT_PARAM in all cases" ) 
            {
                
                REQUIRE( dsl_segvisual_new(NULL, 0,  0) == DSL_RESULT_INVALID_INPUT_PARAM );

                REQUIRE( dsl_segvisual_dimensions_get(NULL, &width, &height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_segvisual_dimensions_set(NULL, width, height) == DSL_RESULT_INVALID_INPUT_PARAM );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}
