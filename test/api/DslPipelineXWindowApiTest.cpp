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
#include "DslPipelineBintr.h"

using namespace DSL;

SCENARIO( "A Pipeline's XWindow Dimensions can be queried", "[pipeline-xwindow-api]" )
{
    GIVEN( "A new Pipeline in memeory" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        
        WHEN( "When the Pipeline is created" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline returns the correct default XWindow dimensions" )
            {
                uint xwindowWidth(123), xwindowHeight(456);
                REQUIRE( dsl_pipeline_xwindow_dimensions_get(pipelineName.c_str(), 
                    &xwindowWidth, &xwindowHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( xwindowWidth == 0 );
                REQUIRE( xwindowHeight == 0 );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Pipeline's XWindow Dimensions can be updated", "[pipeline-xwindow-api]" )
{
    GIVEN( "A new Pipeline in memeory" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        uint newXwindowWidth(1280); 
        uint newXwindowHeight(720);
        
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        WHEN( "When the Pipeline is created" ) 
        {
            REQUIRE( dsl_pipeline_xwindow_dimensions_set(pipelineName.c_str(), 
                newXwindowWidth, newXwindowHeight) == DSL_RESULT_SUCCESS );
                
            THEN( "The Pipeline returns the new XWindow dimensions" )
            {
                uint xwindowWidth(0), xwindowHeight(0);
                REQUIRE( dsl_pipeline_xwindow_dimensions_get(pipelineName.c_str(), 
                    &xwindowWidth, &xwindowHeight) == DSL_RESULT_SUCCESS );
                REQUIRE( xwindowWidth == newXwindowWidth );
                REQUIRE( xwindowHeight == newXwindowHeight );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}
