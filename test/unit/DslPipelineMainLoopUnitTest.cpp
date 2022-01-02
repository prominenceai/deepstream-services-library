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
#include "DslPipelineStateMgr.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

using namespace DSL;

static std::string pipelineName1("pipeline1");
static std::string pipelineName2("pipeline2");
static std::string pipelineName3("pipeline3");

SCENARIO( "A New PipelineBintr can create and delete its own main-loop", "[PipelineMainLoop]" )
{
    GIVEN( "A new Pipeline in memory" ) 
    {

        DSL_PIPELINE_PTR pPipelineBintr =
            DSL_PIPELINE_NEW(pipelineName1.c_str());

        // calling delete after construction must fail
        REQUIRE( pPipelineBintr->DeleteMainLoop() == false );

        WHEN( "The PipelineBintr creates its own main-loop" )
        {
            REQUIRE( pPipelineBintr->NewMainLoop() == true );
            
            // Second call must fail
            REQUIRE( pPipelineBintr->NewMainLoop() == false );
            
            THEN( "The main-loop can then be destroyed" )
            {
                REQUIRE( pPipelineBintr->DeleteMainLoop() == true );
                
                // Second call must fail
                REQUIRE( pPipelineBintr->DeleteMainLoop() == false );
            }
        }
    }
}

static void* main_loop_thread_func(gpointer data)
{
    std::cout << "Pipeline main-loop thread function has started" << std::endl;

    static_cast<PipelineBintr*>(data)->RunMainLoop();

    std::cout << "Pipeline main-loop thread function is quiting" << std::endl;

    return NULL;
}

SCENARIO( "A New PipelineBintr can run and quit its own main-loop", "[PipelineMainLoop]" )
{
    GIVEN( "A new Pipeline in memory" ) 
    {
        DSL_PIPELINE_PTR pPipelineBintr1 =
            DSL_PIPELINE_NEW(pipelineName1.c_str());

        REQUIRE( pPipelineBintr1->NewMainLoop() == true );

        WHEN( "The PipelineBintr's main-loop is started in a new Thread" )
        {
            GThread* main_loop_thread = g_thread_new("main-loop", 
                main_loop_thread_func, pPipelineBintr1.get());
            
            THEN( "The current thread can stop and join the main-loop thread" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPipelineBintr1->QuitMainLoop() == true );

                // second call must fail
                REQUIRE( pPipelineBintr1->QuitMainLoop() == false );
                
                g_thread_join(main_loop_thread);
                
                REQUIRE( pPipelineBintr1->DeleteMainLoop() == true );
            }
        }
    }
}

SCENARIO( "Multiple PipelineBintr can run and quit their own main-loop", "[PipelineMainLoop]" )
{
    GIVEN( "Multiple Pipelines in memory" ) 
    {
        DSL_PIPELINE_PTR pPipelineBintr1 =
            DSL_PIPELINE_NEW(pipelineName1.c_str());
        DSL_PIPELINE_PTR pPipelineBintr2 =
            DSL_PIPELINE_NEW(pipelineName2.c_str());
        DSL_PIPELINE_PTR pPipelineBintr3 =
            DSL_PIPELINE_NEW(pipelineName3.c_str());

        REQUIRE( pPipelineBintr1->NewMainLoop() == true );
        REQUIRE( pPipelineBintr2->NewMainLoop() == true );
        REQUIRE( pPipelineBintr3->NewMainLoop() == true );

        WHEN( "The PipelineBintr's main-loop is started in a new Thread" )
        {
            GThread* main_loop_thread_1 = g_thread_new("main-loop-1", 
                main_loop_thread_func, pPipelineBintr1.get());
            GThread* main_loop_thread_2 = g_thread_new("main-loop-2", 
                main_loop_thread_func, pPipelineBintr2.get());
            GThread* main_loop_thread_3 = g_thread_new("main-loop-3", 
                main_loop_thread_func, pPipelineBintr3.get());
            
            THEN( "The current thread can stop and join the main-loop thread" )
            {
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPipelineBintr1->QuitMainLoop() == true );
                g_thread_join(main_loop_thread_1);
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPipelineBintr2->QuitMainLoop() == true );
                g_thread_join(main_loop_thread_2);
                
                std::this_thread::sleep_for(TIME_TO_SLEEP_FOR);
                REQUIRE( pPipelineBintr3->QuitMainLoop() == true );
                g_thread_join(main_loop_thread_3);
                
                REQUIRE( pPipelineBintr1->DeleteMainLoop() == true );
                REQUIRE( pPipelineBintr2->DeleteMainLoop() == true );
                REQUIRE( pPipelineBintr3->DeleteMainLoop() == true );
            }
        }
    }
}

