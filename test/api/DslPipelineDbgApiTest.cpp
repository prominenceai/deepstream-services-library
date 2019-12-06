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

SCENARIO( "A Pipeline's graph can be dumped to .dot with and without timestamp", "[pipeline-dbg-api]" )
{
    std::wstring pipelineName  = L"test-pipeline";
    std::wstring sourceName = L"csi-source";
    wchar_t fileName[] = L"test-dot-file-sans-ts";

    GIVEN( "A Pipeline in memory with at least one component" ) 
    {
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_source_csi_new(sourceName.c_str(), 1280, 720, 30, 1) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_component_add(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
            
        // ** NOTE ** This test is incomplete... the Pipeline must transition into 
        // a ready or playing state to generate a graph before one can be dumped to
        // file. This scenario only is only testing the service execution at thsi time.
        
    }
    WHEN( "A Pipeline is called to dump its graph to .dot file with timestamp" )
    {
        REQUIRE( dsl_pipeline_dump_to_dot_with_ts(pipelineName.c_str(),
            fileName) == DSL_RESULT_SUCCESS );

        THEN( "The new file is created successfully" ) 
        {
            
        }
    }
    WHEN( "A Pipeline is called to dump its graph to .dot file without timestamp" )
    {
        REQUIRE( dsl_pipeline_dump_to_dot(pipelineName.c_str(),
            fileName) == DSL_RESULT_SUCCESS );

        THEN( "The new file is created successfully" ) 
        {
            
        }
    }
    WHEN( "The Pipeline and Component are deleted")
    {
        REQUIRE( dsl_pipeline_component_remove(pipelineName.c_str(), 
            sourceName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_pipeline_delete(pipelineName.c_str()) == DSL_RESULT_SUCCESS );
        REQUIRE( dsl_component_delete(sourceName.c_str()) == DSL_RESULT_SUCCESS );
        
        THEN( "the containers are updated correctly")
        {
            REQUIRE( dsl_pipeline_list_size() == 0 );
            REQUIRE( dsl_component_list_size() == 0 );
        }
    }
}
