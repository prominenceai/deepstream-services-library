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

SCENARIO( "A single Pipeline is created and deleted correctly", "[PipelineMgt]" )
{
    GIVEN( "An empty list of Pipelines" ) 
    {
        std::wstring actualName  = L"test-pipeline";
        
        REQUIRE( dsl_pipeline_list_size() == 0 );

        WHEN( "A new Pipeline is created" ) 
        {

            REQUIRE( dsl_pipeline_new(actualName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pipeline_list_size() == 1 );
            }
        }
        REQUIRE( dsl_pipeline_delete(actualName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
    }
}

SCENARIO( "Multiple Pipelines are created and deleted correctly", "[pipeline]" )
{
    GIVEN( "A map of multiple Pipelines" ) 
    {
        int sampleSize = 6;
        
        for(int i = 0; i < sampleSize; i++)
        {
            REQUIRE( dsl_pipeline_new(std::to_wstring(i).c_str()) == DSL_RESULT_SUCCESS );
        }
        REQUIRE( dsl_pipeline_list_size() == sampleSize );

        WHEN( "Multiple Pipelines are deleted" ) 
        {
            const wchar_t* pipelineList[] = {L"1",L"3", NULL};
            
            REQUIRE( dsl_pipeline_delete_many(pipelineList) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pipeline_list_size() == sampleSize - 2 );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
    }
}

SCENARIO( "Many Pipelines are created correctly", "[PipelineMgt]" )
{
    const wchar_t* pipelineNames[] = {L"PipelineA", L"PipelineB", L"PipelineC", NULL};
    
    GIVEN( "An empty container of Pipelines" ) 
    {
        REQUIRE( dsl_pipeline_list_size() == 0 );

        WHEN( "Many Pipelines are created at once" ) 
        {

            REQUIRE( dsl_pipeline_new_many(pipelineNames) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pipeline_list_size() == 3 );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
    }
}

SCENARIO( "Many Pipelines are deleted correctly", "[PipelineMgt]" )
{
    const wchar_t* pipelineNames[] = {L"PipelineA", L"PipelineB", L"PipelineC", NULL};
    
    GIVEN( "Many Pipelines created with an array of names" ) 
    {
        REQUIRE( dsl_pipeline_new_many(pipelineNames) == DSL_RESULT_SUCCESS );

        WHEN( "Many Pipelines are deleted with the same array of names" ) 
        {
            REQUIRE( dsl_pipeline_delete_many(pipelineNames) == DSL_RESULT_SUCCESS );

            THEN( "The list size and contents are updated correctly" ) 
            {
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A non-unique Pipeline name fails when creating Many Pipelines", "[PipelineMgt]" )
{
    const wchar_t* pipelineNames[] = {L"PipelineA", L"PipelineB", L"PipelineA", NULL};
    
    GIVEN( "An empty container of Pipelines" ) 
    {
        REQUIRE( dsl_pipeline_list_size() == 0 );

        WHEN( "A non-unique Pipeline Name is used when creating many Pipelines" ) 
        {

            REQUIRE( dsl_pipeline_new_many(pipelineNames) == 
                DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE );

            THEN( "The list size and contents are updated correctly" ) 
            {
                // Only the first two were added?
                // TODO - checkfor uniqueness first
                REQUIRE( dsl_pipeline_list_size() == 2 );
            }
        }
        REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_list_size() == 0 );
    }
}
