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

        REQUIRE( pPipelineSinksBintr->GetNumChildren() == 0 );
            
        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
        REQUIRE( pSinkBintr->GetSinkId() == -1 );

        WHEN( "The Sink Bintr is added to the Pipeline Sinks Bintr" )
        {
            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );
            
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSinksBintr->GetNumChildren() == 1 );
                REQUIRE( pSinkBintr->IsInUse() == true );
                REQUIRE( pSinkBintr->GetSinkId() == -1 );
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
        REQUIRE( pSinkBintr->GetSinkId() == -1 );
            
        WHEN( "The Sink Bintr is removed from the Pipeline Sinks Bintr" )
        {
            REQUIRE( pPipelineSinksBintr->RemoveChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );  
            
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSinksBintr->GetNumChildren() == 0 );
                REQUIRE( pSinkBintr->IsInUse() == false );
                REQUIRE( pSinkBintr->GetSinkId() == -1 );
            }
        }
    }
}

SCENARIO( "Linking multiple sinks to a Pipeline Sinks Bintr Tee is managed correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A new Pipeline Sinks Bintr with several new Sink Bintrs" ) 
    {
        std::string pipelineSinksName = "pipeline-sinks";

        std::string sinkName0 = "overlay-sink-0";
        std::string sinkName1 = "overlay-sink-1";
        std::string sinkName2 = "overlay-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
        DSL_OVERLAY_SINK_PTR pSinkBintr0 = 
            DSL_OVERLAY_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr0->GetSinkId() == -1 );

        DSL_OVERLAY_SINK_PTR pSinkBintr1 = 
            DSL_OVERLAY_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr1->GetSinkId() == -1 );

        DSL_OVERLAY_SINK_PTR pSinkBintr2 = 
            DSL_OVERLAY_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr2->GetSinkId() == -1 );

        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );

        REQUIRE( pPipelineSinksBintr->GetNumChildren() == 3 );
            
        WHEN( "The Sink Bintrs are linked to the Pipeline Sinks Bintr" )
        {
            REQUIRE( pPipelineSinksBintr->LinkAll()  == true );
            
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pSinkBintr0->IsInUse() == true );
                REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr0->GetSinkId() == 0 );
                REQUIRE( pSinkBintr1->IsInUse() == true );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr1->GetSinkId() == 1 );
                REQUIRE( pSinkBintr2->IsInUse() == true );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr2->GetSinkId() == 2 );
            }
        }
    }
}

SCENARIO( "Multiple sinks linked to a Pipeline Sinks Bintr Tee can be unlinked correctly", "[PipelineSinksBintr]" )
{
    GIVEN( "A new Pipeline Sinks Bintr with several new Sink Bintrs all linked" ) 
    {
        std::string pipelineSinksName = "pipeline-sinks";

        std::string sinkName0 = "overlay-sink-0";
        std::string sinkName1 = "overlay-sink-1";
        std::string sinkName2 = "overlay-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
            
        DSL_OVERLAY_SINK_PTR pSinkBintr0 = 
            DSL_OVERLAY_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_OVERLAY_SINK_PTR pSinkBintr1 = 
            DSL_OVERLAY_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_OVERLAY_SINK_PTR pSinkBintr2 = 
            DSL_OVERLAY_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
        REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );

        REQUIRE( pPipelineSinksBintr->GetNumChildren() == 3 );
        REQUIRE( pPipelineSinksBintr->LinkAll()  == true );

        REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr0->GetSinkId() == 0 );
        REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr1->GetSinkId() == 1 );
        REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr2->GetSinkId() == 2 );

        WHEN( "The PipelineSinksBintr and child SinkBintrs are unlinked" )
        {
            pPipelineSinksBintr->UnlinkAll();
            THEN( "The Pipeline Sinks Bintr is updated correctly" )
            {
                REQUIRE( pSinkBintr0->IsLinkedToSource() == false );
                REQUIRE( pSinkBintr0->GetSinkId() == -1 );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == false );
                REQUIRE( pSinkBintr1->GetSinkId() == -1 );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == false );
                REQUIRE( pSinkBintr2->GetSinkId() == -1 );
            }
        }
    }
}

SCENARIO( "All GST Resources are released on PipelineSinksBintr destruction", "[test]" )
{
    GIVEN( "Attributes for a new PipelineSinksBintr and several new SinkBintrs" ) 
    {
        std::string pipelineSinksName = "pipeline-sinks";

        std::string sinkName0 = "overlay-sink-0";
        std::string sinkName1 = "overlay-sink-1";
        std::string sinkName2 = "overlay-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        WHEN( "The Bintrs are created and the Sinks are added as children and linked" )
        {
            DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
                
            DSL_OVERLAY_SINK_PTR pSinkBintr0 = 
                DSL_OVERLAY_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_OVERLAY_SINK_PTR pSinkBintr1 = 
                DSL_OVERLAY_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_OVERLAY_SINK_PTR pSinkBintr2 = 
                DSL_OVERLAY_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );

            REQUIRE( pPipelineSinksBintr->GetNumChildren() == 3 );
            REQUIRE( pPipelineSinksBintr->LinkAll()  == true );

            THEN( "The SinkBintrs are updated correctly" )
            {
                REQUIRE( pSinkBintr2->IsInUse() == true );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr0->IsInUse() == true );
                REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr1->IsInUse() == true );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
            }
        }
        WHEN( "After destruction, all SinkBintrs and Request Pads can be recreated and linked again" )
        {
            DSL_PIPELINE_SINKS_PTR pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW(pipelineSinksName.c_str());
                
            DSL_OVERLAY_SINK_PTR pSinkBintr0 = 
                DSL_OVERLAY_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_OVERLAY_SINK_PTR pSinkBintr1 = 
                DSL_OVERLAY_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_OVERLAY_SINK_PTR pSinkBintr2 = 
                DSL_OVERLAY_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
            REQUIRE( pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );
            pPipelineSinksBintr->LinkAll();

            THEN( "The SinkBintrs are updated correctly" )
            {
                REQUIRE( pSinkBintr2->IsInUse() == true );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr0->IsInUse() == true );
                REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr1->IsInUse() == true );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
            }
        }
    }
}
