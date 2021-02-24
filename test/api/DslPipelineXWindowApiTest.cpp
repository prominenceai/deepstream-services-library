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
#include "DslPipelineBintr.h"

using namespace DSL;

SCENARIO( "A Pipeline's XWindow Offsets and Dimensions can be queried", "[pipeline-xwindow-api]" )
{
    GIVEN( "A new Pipeline in memeory" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        
        WHEN( "When the Pipeline is created" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline returns the correct default XWindow dimensions" )
            {
                uint xwindowOffsetX(123), xwindowOffsetY(456);
                REQUIRE( dsl_pipeline_xwindow_offsets_get(pipelineName.c_str(), 
                    &xwindowOffsetX, &xwindowOffsetY) == DSL_RESULT_SUCCESS );
                REQUIRE( xwindowOffsetX == 0 );
                REQUIRE( xwindowOffsetY == 0 );

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

SCENARIO( "A Pipeline's XWindow Full-Screen-Enabled be Set/Get", "[pipeline-xwindow-api]" )
{
    GIVEN( "A new Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        boolean newFullScreenEnabled(1);
        
        REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

        boolean fullScreenEnabled(1);
        REQUIRE( dsl_pipeline_xwindow_fullscreen_enabled_get(pipelineName.c_str(), 
            &fullScreenEnabled) == DSL_RESULT_SUCCESS );
            
        // must be initialized false
        REQUIRE( fullScreenEnabled == 0 );

        WHEN( "When the Pipeline's XWindow Offsets are updated" ) 
        {
            REQUIRE( dsl_pipeline_xwindow_fullscreen_enabled_set(pipelineName.c_str(), 
                newFullScreenEnabled) == DSL_RESULT_SUCCESS );
                
            THEN( "The new values are returned on get" )
            {
                REQUIRE( dsl_pipeline_xwindow_fullscreen_enabled_get(pipelineName.c_str(), 
                    &fullScreenEnabled) == DSL_RESULT_SUCCESS );
                    
                REQUIRE( fullScreenEnabled == newFullScreenEnabled );

                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}

SCENARIO( "A Pipeline's XWindow Full-Sreen-Enabled setting can be queried", "[pipeline-xwindow-api]" )
{
    GIVEN( "A name for a new Pipeline" ) 
    {
        std::wstring pipelineName  = L"test-pipeline";
        
        WHEN( "When the Pipeline is created" ) 
        {
            REQUIRE( dsl_pipeline_new(pipelineName.c_str()) == DSL_RESULT_SUCCESS );

            THEN( "The Pipeline returns the correct default XWindow offsets" )
            {
                uint xwindowOffsetX(123), xwindowOffsetY(456);
                REQUIRE( dsl_pipeline_xwindow_offsets_get(pipelineName.c_str(), 
                    &xwindowOffsetX, &xwindowOffsetY) == DSL_RESULT_SUCCESS );
                REQUIRE( xwindowOffsetX == 0 );
                REQUIRE( xwindowOffsetY == 0 );
                REQUIRE( dsl_pipeline_delete_all() == DSL_RESULT_SUCCESS );
                REQUIRE( dsl_pipeline_list_size() == 0 );
            }
        }
    }
}
