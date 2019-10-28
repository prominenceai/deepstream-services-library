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
#include "DslPipelineBintr.h"

SCENARIO( "A Pipeline assembles a XWindow correctly", "[pipeline]" )
{
    GIVEN( "A Pipeline with a Tiled Display" ) 
    {
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";

        uint initWidth(1280);
        uint initHeight(720);

        std::shared_ptr<DSL::DisplayBintr> pDisplayBintr = 
            std::shared_ptr<DSL::DisplayBintr>(new DSL::DisplayBintr(
            displayName.c_str(), initWidth, initHeight));

        std::shared_ptr<DSL::PipelineBintr> pPipelineBintr = 
            std::shared_ptr<DSL::PipelineBintr>(new DSL::PipelineBintr(displayName.c_str()));
            
        pDisplayBintr->AddToParent(pPipelineBintr);

        WHEN( "The Pipeline is Assembled" )
        {
//            pPipelineBintr->_assemble();

            THEN( "The Display's new demensions are returned on Get")
            {
            }
        }
    }
}
