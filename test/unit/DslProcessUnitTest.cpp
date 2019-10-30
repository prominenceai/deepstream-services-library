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
#include "DslProcessBintr.h"

SCENARIO( "Adding a single Sink to a Process Bintr is managed correctly" )
{
    GIVEN( "A new Process Bintr and new Sink in memory" ) 
    {
        std::string sinkName = "overlay-sink";
//        std::string pipelineSourcesName = "pipeline-sources";
//
//        std::shared_ptr<DSL::PipelineSourcesBintr> pPipelineSourcesBintr = 
//            std::shared_ptr<DSL::PipelineSourcesBintr>(
//            new DSL::PipelineSourcesBintr(pipelineSourcesName.c_str()));
//
//        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );
//
//        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr = 
//            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
//            sourceName.c_str(), 1280, 720, 30, 1));
//            
//        WHEN( "The Source is added to the Pipeline Sources Bintr" )
//        {
//            pPipelineSourcesBintr->AddChild(pSourceBintr);
//            
//            THEN( "The Pipeline Sources Bintr is updated correctly" )
//            {
//                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 1 );
//                REQUIRE( pSourceBintr->IsInUse() == true );
//            }
//        }
    }
}
