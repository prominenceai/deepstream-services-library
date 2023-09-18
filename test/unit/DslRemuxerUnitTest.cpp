/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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
#include "DslRemuxerBintr.h"
#include "DslBranchBintr.h"

using namespace DSL;

static const std::string branchBintrName0("branch0");
static const std::string branchBintrName1("branch1");
static const std::string branchBintrName2("branch2");
static const std::string sinkName0("fake-sink0");
static const std::string sinkName1("fake-sink1");
static const std::string sinkName2("fake-sink2");

SCENARIO( "A RemuxerBintr is created correctly", "[RemuxerBintr]" )
{
    GIVEN( "Attributes for a new RemuxerBintr" ) 
    {
        std::string remuxerBintrName = "remuxer";
        uint maxStreamIds(4);

        WHEN( "The DemuxerBintr is created" )
        {
            DSL_REMUXER_PTR pRemuxerBintr = 
                DSL_REMUXER_NEW(remuxerBintrName.c_str(), maxStreamIds);
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pRemuxerBintr->GetName() == remuxerBintrName );
                REQUIRE( pRemuxerBintr->GetNumChildren() == 0 );
//                REQUIRE( pDemuxerBintr->GetMaxBranches() == 1 );
//                REQUIRE( pDemuxerBintr->GetBlockingTimeout() 
//                    == DSL_TEE_DEFAULT_BLOCKING_TIMEOUT_IN_SEC );
            }
        }
    }
}

SCENARIO( "A RemuxerBintr can Add and Remove child Branches.", "[rjh]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintr in memory" ) 
    {
        std::string remuxerBintrName = "remuxer";
        uint maxStreamIds(4);
        
        std::vector<uint> streamIds{0,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str(), maxStreamIds);

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());

        // prior to adding the child
        REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );

        WHEN( "The BranchBintr is added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                     &streamIds[0], streamIds.size()) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );

            // Second attempt must fail
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                    &streamIds[0], streamIds.size()) == false );
            
            THEN( "The same BranchBintr can be removed" )
            {
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
                
                // Second attempt must fail
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == false );
            }
        }
    }
}

SCENARIO( "Linking multiple BranchBintrs to a RemuxerBintr is managed correctly",
    "[RemuxerBintr]" )
{
    GIVEN( "A new DemuxerBintr with several new BranchBintrs" ) 
    {
        std::string remuxerBintrName = "remuxer";
        uint maxStreamIds(4);

        std::string branchBintrName0("branch0");
        std::string branchBintrName1("branch1");
        std::string branchBintrName2("branch2");
        std::string sinkName0("fake-sink0");
        std::string sinkName1("fake-sink1");
        std::string sinkName2("fake-sink2");

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str(), maxStreamIds);
        
        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());

        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

//        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
//        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
//        REQUIRE( pDemuxerBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );

//        REQUIRE( pDemuxerBintr->GetNumChildren() == 3 );
            
        WHEN( "The BranchBintrs are linked to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->LinkAll() == true );
            
            THEN( "The BranchBintrs are updated correctly" )
            {
//                REQUIRE( pBranchBintr0->IsInUse() == true );
//                REQUIRE( pBranchBintr0->IsLinkedToSource() == true );
//                REQUIRE( pBranchBintr0->GetRequestPadId() == 0 );
//                REQUIRE( pBranchBintr1->IsInUse() == true );
//                REQUIRE( pBranchBintr1->IsLinkedToSource() == true );
//                REQUIRE( pBranchBintr1->GetRequestPadId() == 1 );
//                REQUIRE( pBranchBintr2->IsInUse() == true );
//                REQUIRE( pBranchBintr2->IsLinkedToSource() == true );
//                REQUIRE( pBranchBintr2->GetRequestPadId() == 2 );
            }
        }
    }
}

