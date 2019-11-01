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
#include "DslPipelineSinksBintr.h"
#include "DslSinkBintr.h"

using namespace DSL;

SCENARIO( "Adding a single Sink to a Pipeline Sinks Bintr is managed correctly" )
{
    GIVEN( "A new Pipelinks Sinks Bintr and new Sink Bintr in memory" ) 
    {
        std::string sinkName = "overlay-sink";
        std::string pipelineSinksName = "pipeline-sinks";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pPipelineSinksBintr->GetNumChildren() == 0 );
            
        WHEN( "The Sink Bintr is added to the Pipeline Sinks Bintr" )
        {
            pPipelineSinksBintr->AddChild(pSinkBintr);
            
            THEN( "The Pipeline Sources Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSinksBintr->GetNumChildren() == 1 );
                REQUIRE( pSinkBintr->IsInUse() == true );
            }
        }
    }
}
