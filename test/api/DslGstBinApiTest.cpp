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


SCENARIO( "The Components container is updated correctly on new GST Pin", "[gst-bin-api]" )
{
    GIVEN( "An empty list of Components" ) 
    {
        std::wstring binName(L"test-bin");

        REQUIRE( dsl_component_list_size() == 0 );

        WHEN( "A new GST Thin is created" ) 
        {

            REQUIRE( dsl_gst_bin_new(binName.c_str()) == DSL_RESULT_SUCCESS );
            
            // second call must fail
            REQUIRE( dsl_gst_bin_new(binName.c_str()) ==
                DSL_RESULT_GST_BIN_NAME_NOT_UNIQUE );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_component_list_size() == 1 );
            }
        }
        REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
    }
}

SCENARIO( "The Components container is updated correctly on GST Bin delete", "[gst-bin-api]" )
{
    GIVEN( "A new GST Bin in memory" ) 
    {
        std::wstring binName(L"test-bin");

        REQUIRE( dsl_component_list_size() == 0 );
        REQUIRE( dsl_gst_bin_new(binName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );

        WHEN( "The new GST Bin is deleted" ) 
        {
            REQUIRE( dsl_component_delete(binName.c_str()) == DSL_RESULT_SUCCESS );
            
            THEN( "The list size is updated correctly" )
            {
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A GST Been in use can't be deleted", "[gst-bin-api]" )
{
    GIVEN( "A new GST Display and new Pipeline" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring binName(L"test-bin");

        REQUIRE( dsl_gst_bin_new(binName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        WHEN( "The GST Been is added to the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
                binName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The GST Bin can't be deleted" ) 
            {
                REQUIRE( dsl_component_delete(binName.c_str()) == DSL_RESULT_COMPONENT_IN_USE );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
                REQUIRE( dsl_component_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A GST Been, once removed from a Pipeline, can be deleted", "[gst-bin-api]" )
{
    GIVEN( "A new Pipeline with a child GST Bin" ) 
    {
        std::wstring pipelineName(L"test-pipeline");
        std::wstring binName(L"test-bin");

        REQUIRE( dsl_gst_bin_new(binName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_list_size() == 1 );
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 1 );

        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            binName.c_str()) == DSL_RESULT_SUCCESS );
            
        WHEN( "The GST Been is from the Pipeline" ) 
        {
            REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
                binName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The GST Bin can be deleted" ) 
            {
                REQUIRE( dsl_component_delete(binName.c_str()) == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_component_list_size() == 0 );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}