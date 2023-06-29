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
#include "DslMultiComponentsBintr.h"
#include "DslBranchBintr.h"

using namespace DSL;

// Note: other than the constructor, Demuxers and Tees are of the
// same MultiComponentsBintr and share the same methods. 

SCENARIO( "A DemuxerBintr is created correctly", "[DemuxerBintr]" )
{
    GIVEN( "A name for a DemuxerBintr" ) 
    {
        std::string demuxerBintrName = "demuxer";

        WHEN( "The DemuxerBintr is created" )
        {
            DSL_DEMUXER_PTR pDemuxerBintr = 
                DSL_DEMUXER_NEW(demuxerBintrName.c_str(), 1);
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pDemuxerBintr->GetName() == demuxerBintrName );
                REQUIRE( pDemuxerBintr->GetNumChildren() == 0 );
                REQUIRE( pDemuxerBintr->GetMaxBranches() == 1 );
            }
        }
    }
}

SCENARIO( "Adding a single Branch to a DemuxerBintr is managed correctly", "[DemuxerBintr]" )
{
    GIVEN( "A new DemuxerBintr and new BranchBintr in memory" ) 
    {
        std::string branchBintrName = "branch";
        std::string demuxerBintrName = "demuxer";

        DSL_DEMUXER_PTR pDemuxerBintr = 
            DSL_DEMUXER_NEW(demuxerBintrName.c_str(), 1);

        REQUIRE( pDemuxerBintr->GetNumChildren() == 0 );
            
        DSL_BRANCH_PTR pBranchBintr = DSL_BRANCH_NEW(branchBintrName.c_str());
            
        REQUIRE( pBranchBintr->GetId() == -1 );

        WHEN( "The BranchBintr is added to the DemuxerBintr" )
        {
            REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr)) == true );
            
            THEN( "The DemuxerBintr and BranchBintr updated correctly" )
            {
                REQUIRE( pDemuxerBintr->GetNumChildren() == 1 );
                REQUIRE( pBranchBintr->IsInUse() == true );
                REQUIRE( pBranchBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Removing a single BranchBintr from a DemuxerBintr is managed correctly", "[DemuxerBintr]" )
{
    GIVEN( "A new DemuxerBintr with a new Sink Bintr" ) 
    {
        std::string branchBintrName = "branch";
        std::string demuxerBintrName = "demuxer";

        DSL_DEMUXER_PTR pDemuxerBintr = DSL_DEMUXER_NEW(demuxerBintrName.c_str(), 1);
        DSL_BRANCH_PTR pBranchBintr = DSL_BRANCH_NEW(branchBintrName.c_str());

        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr)) == true );
            
        WHEN( "The BranchBintr is removed from the DemuxerBintr" )
        {
            REQUIRE( pDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr)) == true );  
            
            THEN( "The DemuxerBintr is updated correctly" )
            {
                REQUIRE( pDemuxerBintr->GetNumChildren() == 0 );
                REQUIRE( pBranchBintr->IsInUse() == false );
                REQUIRE( pBranchBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Linking multiple BranchBintrs to a DemuxerBintr is managed correctly", "[DemuxerBintr]" )
{
    GIVEN( "A new DemuxerBintr with several new BranchBintrs" ) 
    {
        std::string demuxerBintrName("demuxer");
        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");


        DSL_DEMUXER_PTR pDemuxerBintr = DSL_DEMUXER_NEW(demuxerBintrName.c_str(), 3);
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

        REQUIRE( pDemuxerBintr->GetNumChildren() == 3 );
            
        WHEN( "The BranchBintrs are linked to the DemuxerBintr" )
        {
            REQUIRE( pDemuxerBintr->LinkAll()  == true );
            
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

SCENARIO( "Multiple Branches linked to a Demuxer can be unlinked correctly", "[DemuxerBintr]" )
{
    GIVEN( "A new DemuxerBintr with several new BranchBintrs" ) 
    {
        std::string demuxerBintrName("demuxer");
        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");


        DSL_DEMUXER_PTR pDemuxerBintr = DSL_DEMUXER_NEW(demuxerBintrName.c_str(), 3);
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

        REQUIRE( pDemuxerBintr->GetNumChildren() == 3 );
        REQUIRE( pDemuxerBintr->LinkAll()  == true );

        REQUIRE( pBranchBintr0->IsInUse() == true );
        REQUIRE( pBranchBintr0->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr0->GetId() == 0 );
        REQUIRE( pBranchBintr1->IsInUse() == true );
        REQUIRE( pBranchBintr1->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr1->GetId() == 1 );
        REQUIRE( pBranchBintr2->IsInUse() == true );
        REQUIRE( pBranchBintr2->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr2->GetId() == 2 );
            
        WHEN( "The BranchBintrs are unlinked and removed from the DemuxerBintr" )
        {
            pDemuxerBintr->UnlinkAll();
            REQUIRE( pDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pDemuxerBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

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

SCENARIO( "Multiple Branches linked to a Splitter component can be unlinked correctly", "[SplitterBintr]" )
{
    GIVEN( "A new DemuxerBintr with several new BranchBintrs" ) 
    {
        std::string splitterBintrName("splitter");
        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");


        DSL_SPLITTER_PTR pSplitterBinter = DSL_SPLITTER_NEW(splitterBintrName.c_str());
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pSplitterBinter->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pSplitterBinter->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pSplitterBinter->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

        REQUIRE( pSplitterBinter->GetNumChildren() == 3 );
        REQUIRE( pSplitterBinter->LinkAll()  == true );

        REQUIRE( pBranchBintr0->IsInUse() == true );
        REQUIRE( pBranchBintr0->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr0->GetId() == 0 );
        REQUIRE( pBranchBintr1->IsInUse() == true );
        REQUIRE( pBranchBintr1->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr1->GetId() == 1 );
        REQUIRE( pBranchBintr2->IsInUse() == true );
        REQUIRE( pBranchBintr2->IsLinkedToSource() == true );
        REQUIRE( pBranchBintr2->GetId() == 2 );
            
        WHEN( "The BranchBintrs are unlinked and removed from the DemuxerBintr" )
        {
            pSplitterBinter->UnlinkAll();
            REQUIRE( pSplitterBinter->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pSplitterBinter->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pSplitterBinter->RemoveChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

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
