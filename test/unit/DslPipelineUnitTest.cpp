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
#include "DslDisplayBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslPipelineBintr.h"

using namespace DSL;

SCENARIO( "A New PipelineBintr is created correctly", "[PipelineBintr]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        std::string pipelineName = "test-pipeline";

        WHEN( "The new PipelineBintr is created" )
        {
            DSL_PIPELINE_PTR pPipelineBintr = 
                DSL_PIPELINE_NEW(pipelineName.c_str());

            THEN( "All member variables are setup correctly" )
            {
                REQUIRE( pPipelineBintr->m_name == pipelineName );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to asseble with a Tiled Display component ", "[PipelineBintr]" )
{
    GIVEN( "A Pipeline and a Tiled Display only" ) 
    {
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";

        uint displayW(1280);
        uint displayH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName.c_str(), displayW, displayH, fps_n, fps_d));

        std::shared_ptr<DSL::DisplayBintr> pDisplayBintr = 
            std::shared_ptr<DSL::DisplayBintr>(new DSL::DisplayBintr(
            displayName.c_str(), displayW, displayH));

        std::shared_ptr<DSL::OverlaySinkBintr> pSinkBintr = 
            std::shared_ptr<DSL::OverlaySinkBintr>(new DSL::OverlaySinkBintr(
            sinkName.c_str(), offsetX, offsetY, sinkW, sinkH));

        std::shared_ptr<DSL::PipelineBintr> pPipelineBintr = 
            std::shared_ptr<DSL::PipelineBintr>(new DSL::PipelineBintr(pipelineName.c_str()));
            
        WHEN( "The Pipeline is setup with a Tiled Display Component" )
        {
            pSourceBintr->AddToParent(pPipelineBintr);
            pDisplayBintr->AddToParent(pPipelineBintr);
            pSinkBintr->AddToParent(pPipelineBintr);

            THEN( "The Pipeline fails to assemble" )
            {
//                REQUIRE( pPipelineBintr->_assemble() == true );
//                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
    }
}
