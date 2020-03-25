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
#include "DslMultiComponentsBintr.h"
#include "DslBranchBintr.h"

using namespace DSL;

SCENARIO( "A StreamDemuxerBintr is created correctly", "[StreamDemuxerBintr]" )
{
    GIVEN( "A name for a StreamDemuxerBintr" ) 
    {
        std::string demuxerBintrName = "demuxer";

        WHEN( "The StreamDemuxerBintr is created" )
        {
            DSL_STREAM_DEMUXER_PTR pStreamDemuxerBintr = DSL_STREAM_DEMUXER_NEW(demuxerBintrName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pStreamDemuxerBintr->GetName() == demuxerBintrName );
                REQUIRE( pStreamDemuxerBintr->GetNumChildren() == 0 );
            }
        }
    }
}
SCENARIO( "Adding a single Branch to a StreamDemuxerBintr is managed correctly", "[StreamDemuxerBintr]" )
{
    GIVEN( "A new StreamDemuxerBintr and new BranchBintr in memory" ) 
    {
        std::string branchBintrName = "branch";
        std::string demuxerBintrName = "demuxer";

        DSL_STREAM_DEMUXER_PTR pStreamDemuxerBintr = DSL_STREAM_DEMUXER_NEW(demuxerBintrName.c_str());

        REQUIRE( pStreamDemuxerBintr->GetNumChildren() == 0 );
            
        DSL_BRANCH_PTR pBranchBintr = DSL_BRANCH_NEW(branchBintrName.c_str());
            
        REQUIRE( pBranchBintr->GetId() == -1 );

        WHEN( "The BranchBintr is added to the StreamDemuxerBintr" )
        {
            REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr)) == true );
            
            THEN( "The StreamDemuxerBintr and BranchBintr updated correctly" )
            {
                REQUIRE( pStreamDemuxerBintr->GetNumChildren() == 1 );
                REQUIRE( pBranchBintr->IsInUse() == true );
                REQUIRE( pBranchBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Removing a single BranchBintr from a StreamDemuxerBintr is managed correctly", "[StreamDemuxerBintr]" )
{
    GIVEN( "A new StreamDemuxerBintr with a new Sink Bintr" ) 
    {
        std::string branchBintrName = "branch";
        std::string demuxerBintrName = "demuxer";

        DSL_STREAM_DEMUXER_PTR pStreamDemuxerBintr = DSL_STREAM_DEMUXER_NEW(demuxerBintrName.c_str());
        DSL_BRANCH_PTR pBranchBintr = DSL_BRANCH_NEW(branchBintrName.c_str());

        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr)) == true );
            
        WHEN( "The BranchBintr is removed from the StreamDemuxerBintr" )
        {
            REQUIRE( pStreamDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr)) == true );  
            
            THEN( "The StreamDemuxerBintr is updated correctly" )
            {
                REQUIRE( pStreamDemuxerBintr->GetNumChildren() == 0 );
                REQUIRE( pBranchBintr->IsInUse() == false );
                REQUIRE( pBranchBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Linking multiple BranchBintrs to a StreamDemuxerBintr is managed correctly", "[StreamDemuxerBintr]" )
{
    GIVEN( "A new StreamDemuxerBintr with several new BranchBintrs" ) 
    {
        std::string demuxerBintrName("demuxer");
        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");


        DSL_STREAM_DEMUXER_PTR pStreamDemuxerBintr = DSL_STREAM_DEMUXER_NEW(demuxerBintrName.c_str());
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

        REQUIRE( pStreamDemuxerBintr->GetNumChildren() == 3 );
            
        WHEN( "The BranchBintrs are linked to the StreamDemuxerBintr" )
        {
            REQUIRE( pStreamDemuxerBintr->LinkAll()  == true );
            
            THEN( "The BranchBintrs are updated correctly" )
            {
                REQUIRE( pBranchBintr0->IsInUse() == true );
                REQUIRE( pBranchBintr0->IsLinkedToSource() == true );
                REQUIRE( pBranchBintr0->GetId() == 0 );
                REQUIRE( pBranchBintr1->IsInUse() == true );
                REQUIRE( pBranchBintr1->IsLinkedToSource() == true );
                REQUIRE( pBranchBintr1->GetId() == 1 );
                REQUIRE( pBranchBintr2->IsInUse() == true );
                REQUIRE( pBranchBintr2->IsLinkedToSource() == true );
                REQUIRE( pBranchBintr2->GetId() == 2 );
            }
        }
    }
}

SCENARIO( "Multiple Branches linked to a Demuxer can be unlinked correctly", "[StreamDemuxerBintr]" )
{
    GIVEN( "A new StreamDemuxerBintr with several new BranchBintrs" ) 
    {
        std::string demuxerBintrName("demuxer");
        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");


        DSL_STREAM_DEMUXER_PTR pStreamDemuxerBintr = DSL_STREAM_DEMUXER_NEW(demuxerBintrName.c_str());
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pStreamDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

        REQUIRE( pStreamDemuxerBintr->GetNumChildren() == 3 );
        REQUIRE( pStreamDemuxerBintr->LinkAll()  == true );

        REQUIRE( pBranchBintr0->IsInUse() == true );
        REQUIRE( pBranchBintr0->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr0->GetId() == 0 );
        REQUIRE( pBranchBintr1->IsInUse() == true );
        REQUIRE( pBranchBintr1->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr1->GetId() == 1 );
        REQUIRE( pBranchBintr2->IsInUse() == true );
        REQUIRE( pBranchBintr2->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr2->GetId() == 2 );
            
        WHEN( "The BranchBintrs are unlinked and removed from the StreamDemuxerBintr" )
        {
            pStreamDemuxerBintr->UnlinkAll();
            REQUIRE( pStreamDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pStreamDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pStreamDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

            THEN( "The BranchBintrs are updated correctly" )
            {
                REQUIRE( pBranchBintr0->IsInUse() == false );
                REQUIRE( pBranchBintr0->IsLinkedToSource() == false );
                REQUIRE( pBranchBintr0->GetId() == -1 );
                REQUIRE( pBranchBintr1->IsInUse() == false );
                REQUIRE( pBranchBintr1->IsLinkedToSource() == false );
                REQUIRE( pBranchBintr1->GetId() == -1 );
                REQUIRE( pBranchBintr2->IsInUse() == false );
                REQUIRE( pBranchBintr2->IsLinkedToSource() == false );
                REQUIRE( pBranchBintr2->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Multiple Branches linked to a Tee component can be unlinked correctly", "[TeeBintr]" )
{
    GIVEN( "A new StreamDemuxerBintr with several new BranchBintrs" ) 
    {
        std::string teeBintrName("tee");
        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");


        DSL_TEE_PTR pTeeBintr = DSL_TEE_NEW(teeBintrName.c_str());
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pTeeBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pTeeBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pTeeBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

        REQUIRE( pTeeBintr->GetNumChildren() == 3 );
        REQUIRE( pTeeBintr->LinkAll()  == true );

        REQUIRE( pBranchBintr0->IsInUse() == true );
        REQUIRE( pBranchBintr0->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr0->GetId() == 0 );
        REQUIRE( pBranchBintr1->IsInUse() == true );
        REQUIRE( pBranchBintr1->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr1->GetId() == 1 );
        REQUIRE( pBranchBintr2->IsInUse() == true );
        REQUIRE( pBranchBintr2->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr2->GetId() == 2 );
            
        WHEN( "The BranchBintrs are unlinked and removed from the StreamDemuxerBintr" )
        {
            pTeeBintr->UnlinkAll();
            REQUIRE( pTeeBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pTeeBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pTeeBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

            THEN( "The BranchBintrs are updated correctly" )
            {
                REQUIRE( pBranchBintr0->IsInUse() == false );
                REQUIRE( pBranchBintr0->IsLinkedToSource() == false );
                REQUIRE( pBranchBintr0->GetId() == -1 );
                REQUIRE( pBranchBintr1->IsInUse() == false );
                REQUIRE( pBranchBintr1->IsLinkedToSource() == false );
                REQUIRE( pBranchBintr1->GetId() == -1 );
                REQUIRE( pBranchBintr2->IsInUse() == false );
                REQUIRE( pBranchBintr2->IsLinkedToSource() == false );
                REQUIRE( pBranchBintr2->GetId() == -1 );
            }
        }
    }
}
