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

SCENARIO( "A Pipeline fails to assemble without one Source component ", "[pipeline]" )
{
    GIVEN( "A Pipeline and a Tiled Display only" ) 
    {
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";

        uint initWidth(1280);
        uint initHeight(720);

        std::shared_ptr<DSL::DisplayBintr> pDisplayBintr = 
            std::shared_ptr<DSL::DisplayBintr>(new DSL::DisplayBintr(
            displayName.c_str(), initWidth, initHeight));

        std::shared_ptr<DSL::PipelineBintr> pPipelineBintr = 
            std::shared_ptr<DSL::PipelineBintr>(new DSL::PipelineBintr(pipelineName.c_str()));
            
        WHEN( "The Pipeline is setup without a Source Component" )
        {
            pDisplayBintr->AddToParent(pPipelineBintr);

            THEN( "The Pipeline fails to assemble" )
            {
                REQUIRE( pPipelineBintr->_assemble() == false );
//                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
    }
}

SCENARIO( "A Pipeline is able to asseble with a Tiled Display component ", "[pipeline]" )
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
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
    }
}
