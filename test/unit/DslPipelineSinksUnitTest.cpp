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

SCENARIO( "A PipelineSinksBintr is created correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A name for a PipelineSinksBintr" ) 
    {
        std::string pipelineSinksName = "pipeline-sinks";

        WHEN( "The PipelineSinksBintr is created" )
        {
            DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = 
                DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSinksBintr->GetName() == pipelineSinksName );
                REQUIRE( pPipelineSinksBintr->GetNumChildren() == 0 );
                REQUIRE( pPipelineSinksBintr->m_pQueue != nullptr );
                REQUIRE( pPipelineSinksBintr->m_pTee != nullptr );
            }
        }
    }
}

SCENARIO( "Adding a single Sink to a Pipeline Sinks Bintr is managed correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A new Pipeline Sinks Bintr and new Sink Bintr in memory" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);
        std::string sinkName = "overlay-sink";
        std::string pipelineSinksName = "pipeline-sinks";

        DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        WHEN( "The Sink Bintr is added to the Pipeline Sinks Bintr" )
        {
            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );
            
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSinksBintr->GetNumChildren() == 1 );
                REQUIRE( pSinkBintr->IsInUse() == true );
            }
        }
    }
}

SCENARIO( "Removing a single Sink from a Pipeline Sinks Bintr is managed correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A new Pipeline Sinks Bintr with a new Sink Bintr" ) 
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

        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );
            
        WHEN( "The Sink Bintr is removed from the Pipeline Sinks Bintr" )
        {
            REQUIRE( pPipelineSinksBintr->RemoveChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );  
            
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSinksBintr->GetNumChildren() == 0 );
                REQUIRE( pSinkBintr->IsInUse() == false );
            }
        }
    }
}

SCENARIO( "Linking multiple sinks to a Pipeline Sinks Bintr Tee is managed correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A new Pipeline Sinks Bintr with several new Sink Bintrs" ) 
    {
        std::string pipelineSinksName = "pipeline-sinks";

        std::string sinkName1 = "overlay-sink-1";
        std::string sinkName2 = "overlay-sink-2";
        std::string sinkName3 = "overlay-sink-3";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
        DSL_OVERLAY_SINK_PTR pSinkBintr1 = 
            DSL_OVERLAY_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_OVERLAY_SINK_PTR pSinkBintr2 = 
            DSL_OVERLAY_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_OVERLAY_SINK_PTR pSinkBintr3 = 
            DSL_OVERLAY_SINK_NEW(sinkName3.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );
        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr3)) == true );

        REQUIRE( pPipelineSinksBintr->GetNumChildren() == 3 );
            
        WHEN( "The Sink Bintrs are linked to the Pipeline Sinks Bintr" )
        {
            REQUIRE( pPipelineSinksBintr->LinkAll()  == true );
            
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pSinkBintr3->IsInUse() == true );
                REQUIRE( pSinkBintr3->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr1->IsInUse() == true );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr2->IsInUse() == true );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
            }
        }
    }
}

SCENARIO( "Multiple sinks linked to a Pipeline Sinks Bintr Tee can be unlinked correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A new Pipeline Sinks Bintr with several new Sink Bintrs all linked" ) 
    {
        std::string pipelineSinksName = "pipeline-sinks";

        std::string sinkName1 = "overlay-sink-1";
        std::string sinkName2 = "overlay-sink-2";
        std::string sinkName3 = "overlay-sink-3";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
        DSL_OVERLAY_SINK_PTR pSinkBintr1 = 
            DSL_OVERLAY_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_OVERLAY_SINK_PTR pSinkBintr2 = 
            DSL_OVERLAY_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_OVERLAY_SINK_PTR pSinkBintr3 = 
            DSL_OVERLAY_SINK_NEW(sinkName3.c_str(), offsetX, offsetY, sinkW, sinkH);

        pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1));
        pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2));
        pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr3));
        pPipelineSinksBintr->LinkAll();

        WHEN( "The Sink Bintrs are linked to the Pipeline Sinks Bintr" )
        {
            
            pPipelineSinksBintr->UnlinkAll();
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
            }
        }
    }
}
