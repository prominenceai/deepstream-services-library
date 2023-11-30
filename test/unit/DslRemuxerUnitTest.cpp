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
#include "DslSinkBintr.h"

using namespace DSL;

static const std::string remuxerBintrName("remuxer");
static const std::string remuxerBranchBintrName0("remuxer-branch0");
static const std::string remuxerBranchBintrName1("remuxer-branch1");
static const std::string remuxerBranchBintrName2("remuxer-branch2");

static const std::string branchBintrName0("branch0");
static const std::string branchBintrName1("branch1");
static const std::string branchBintrName2("branch2");
static const std::string sinkName0("fake-sink0");
static const std::string sinkName1("fake-sink1");
static const std::string sinkName2("fake-sink2");

SCENARIO( "A RemuxerBranchBintr with specific stream-ids is created correctly",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, BranchBintr, and attributes for a new RemuxerBranchBintr" )
    {
        std::vector<uint> streamIds{0,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());

        WHEN( "The RemuxerBranchBintr is created" )
        {
            DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
                DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                    pRemuxerBintr->GetGstObject(), pBranchBintr0, 
                    &streamIds[0], streamIds.size());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pRemuxerBranchBintr->GetName() == remuxerBranchBintrName0 );
                REQUIRE( pRemuxerBranchBintr->GetBatchSize() == 0 );
            }
        }
    }
}

SCENARIO( "A RemuxerBranchBintr whith no specific stream-ids is created correctly",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, BranchBintr, and attributes for a new RemuxerBranchBintr" )
    {
        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());

        WHEN( "The RemuxerBranchBintr is created" )
        {
            DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
                DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                    pRemuxerBintr->GetGstObject(), pBranchBintr0, 
                    NULL, 0);
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pRemuxerBranchBintr->GetName() == remuxerBranchBintrName0 );
                REQUIRE( pRemuxerBranchBintr->GetBatchSize() == 0 );
            }
        }
    }
}

SCENARIO( "A RemuxerBranchBintr with specific stream-ids can LinkAll",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, FakeSinkBintr, and a new RemuxerBranchBintr" )
    {
        std::vector<uint> streamIds{0,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_FAKE_SINK_PTR pFakeSink = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
            DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                pRemuxerBintr->GetGstObject(), pFakeSink, 
                &streamIds[0], streamIds.size());

        WHEN( "The RemuxerBranchBintr is linked" )
        {
            pRemuxerBranchBintr->SetBatchSize(4);
            REQUIRE( pRemuxerBranchBintr->LinkAll() == true );
            
            THEN( "The linked state is updated correctly" )
            {
                REQUIRE( pRemuxerBranchBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A RemuxerBranchBintr with specific stream-ids can UnlinkAll",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, FakeSinkBintr, and a new RemuxerBranchBintr" )
    {
        std::vector<uint> streamIds{0,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_FAKE_SINK_PTR pFakeSink = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
            DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                pRemuxerBintr->GetGstObject(), pFakeSink, 
                &streamIds[0], streamIds.size());

        pRemuxerBranchBintr->SetBatchSize(4);
        REQUIRE( pRemuxerBranchBintr->LinkAll() == true );
        REQUIRE( pRemuxerBranchBintr->IsLinked() == true );

        // second attempt must fail
        REQUIRE( pRemuxerBranchBintr->LinkAll() == false );

        WHEN( "The RemuxerBranchBintr is unlinked" )
        {
            pRemuxerBranchBintr->UnlinkAll();
            
            THEN( "The linked state is updated correctly" )
            {
                REQUIRE( pRemuxerBranchBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A RemuxerBranchBintr with no specific stream-ids can LinkAll",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, FakeSinkBintr, and a new RemuxerBranchBintr" )
    {
        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_FAKE_SINK_PTR pFakeSink = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
            DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                pRemuxerBintr->GetGstObject(), pFakeSink, NULL, 0);

        WHEN( "The RemuxerBranchBintr is linked" )
        {
            pRemuxerBranchBintr->SetBatchSize(4);
            REQUIRE( pRemuxerBranchBintr->LinkAll() == true );
            
            THEN( "The linked state is updated correctly" )
            {
                REQUIRE( pRemuxerBranchBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A RemuxerBranchBintr with no specific stream-ids can UnlinkAll",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, FakeSinkBintr, and a new RemuxerBranchBintr" )
    {
        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_FAKE_SINK_PTR pFakeSink = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
            DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                pRemuxerBintr->GetGstObject(), pFakeSink, NULL, 0);

        pRemuxerBranchBintr->SetBatchSize(4);
        REQUIRE( pRemuxerBranchBintr->LinkAll() == true );
        REQUIRE( pRemuxerBranchBintr->IsLinked() == true );

        // second attempt must fail
        REQUIRE( pRemuxerBranchBintr->LinkAll() == false );

        WHEN( "The RemuxerBranchBintr is unlinked" )
        {
            pRemuxerBranchBintr->UnlinkAll();
            
            THEN( "The linked state is updated correctly" )
            {
                REQUIRE( pRemuxerBranchBintr->IsLinked() == false );
            }
        }
    }
}


SCENARIO( "A RemuxerBintr is created correctly", "[RemuxerBintr]" )
{
    GIVEN( "Attributes for a new RemuxerBintr" ) 
    {
        WHEN( "The RemuxerBintr is created" )
        {
            DSL_REMUXER_PTR pRemuxerBintr = 
                DSL_REMUXER_NEW(remuxerBintrName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pRemuxerBintr->GetName() == remuxerBintrName );
                REQUIRE( pRemuxerBintr->GetNumChildren() == 0 );
                
                REQUIRE( pRemuxerBintr->GetBatchSize() == 0 );
            }
        }
    }
}

SCENARIO( "A RemuxerBintr can Add and Remove a child Branch.", 
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintr in memory" ) 
    {
        std::vector<uint> streamIds{0,1,2,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());

        // prior to adding the child
        REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );

        WHEN( "The BranchBintr is added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );

            // Second attempt must fail
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == false );
            
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

SCENARIO( "A RemuxerBintr can Add-To and Remove a child Branch.", 
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintr in memory" ) 
    {
        std::vector<uint> streamIds{0,1,2,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());

        // prior to adding the child
        REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );

        WHEN( "The BranchBintr is added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->AddChildTo(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                     &streamIds[0], streamIds.size()) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );

            // Second attempt must fail
            REQUIRE( pRemuxerBintr->AddChildTo(
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

SCENARIO( "A RemuxerBintr can Add and Remove multiple child Branches", 
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintr in memory" ) 
    {
        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());
        WHEN( "The BranchBintrs are added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pRemuxerBintr->AddChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
            
            THEN( "The same BranchBintrs can be removed" )
            {
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
            }
        }
    }
}


SCENARIO( "A RemuxerBintr can Add-To and Remove multiple child Branches", 
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintr in memory" ) 
    {
        std::vector<uint> streamIds{0,1,2,3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());
        WHEN( "The BranchBintrs are added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->AddChildTo(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                     &streamIds[0], streamIds.size()) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
            REQUIRE( pRemuxerBintr->AddChildTo(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr1),
                     &streamIds[0], streamIds.size()) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
            REQUIRE( pRemuxerBintr->AddChildTo(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr2),
                     &streamIds[0], streamIds.size()) == true );
            REQUIRE( pRemuxerBintr->IsChild(
                std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
            
            THEN( "The same BranchBintrs can be removed" )
            {
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
                REQUIRE( pRemuxerBintr->RemoveChild(
                    std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
                REQUIRE( pRemuxerBintr->IsChild(pBranchBintr0) == false );
            }
        }
    }
}

SCENARIO( "Linking a BranchBintr with select stream-ids to a RemuxerBintr is managed correctly",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintr in memory" ) 
    {
        std::string remuxerBintrName = "remuxer";
        
        std::vector<uint> streamIds0{0,2,3,5};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );

        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                 &streamIds0[0], streamIds0.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        
        WHEN( "The batch-size and Branch stream-ids are valid" )
        {
            REQUIRE( pRemuxerBintr->SetBatchSize(6) == true );
            
            THEN( "The Remuxer can LinkAll Successfully" )
            {
                REQUIRE( pRemuxerBintr->LinkAll() == true );
                REQUIRE( pRemuxerBintr->IsLinked() == true );
            }
        }
        WHEN( "The num-stream-ids is greater than the batch-size" )
        {
            pRemuxerBintr->SetBatchSize(3);
            
            THEN( "The RemuxerBintr must fail to LinkAll" )
            {
                REQUIRE( pRemuxerBintr->LinkAll() == false );
                REQUIRE( pRemuxerBintr->IsLinked() == false );
            }
        }
        WHEN( "An one or more stream-ids are greater in value than the batch-size" )
        {
            pRemuxerBintr->SetBatchSize(5);
            
            THEN( "The RemuxerBintr must fail to LinkAll" )
            {
                REQUIRE( pRemuxerBintr->LinkAll() == false );
                REQUIRE( pRemuxerBintr->IsLinked() == false );
            }
        }
}
}
SCENARIO( "A RemuxerBranchBintr with invalid stream-ids will fail to LinkAll", 
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr, FakeSinkBintr, and a new RemuxerBranchBintr" )
    {
        std::vector<uint> streamIds{0,1,2,5};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_FAKE_SINK_PTR pFakeSink = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        DSL_REMUXER_BRANCH_PTR pRemuxerBranchBintr = 
            DSL_REMUXER_BRANCH_NEW(remuxerBranchBintrName0.c_str(), 
                pRemuxerBintr->GetGstObject(), pFakeSink, 
                &streamIds[0], streamIds.size());
    }
}


SCENARIO( "Linking a BranchBintr with no select stream-ids to a RemuxerBintr is managed correctly",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintrs in memory" ) 
    {
        std::string remuxerBintrName = "remuxer";
        
        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );

        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                 NULL, 0) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        
        WHEN( "The BranchBintrs are added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->SetBatchSize(4) == true );
            REQUIRE( pRemuxerBintr->LinkAll() == true );
            THEN( "The same BranchBintrs can be removed" )
            {
                REQUIRE( pRemuxerBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "Linking multiple BranchBintrs with select stream-ids to a RemuxerBintr is managed correctly",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintrs in memory" ) 
    {
        std::string remuxerBintrName = "remuxer";
        std::vector<uint> streamIds0{0,1,2,3};
        std::vector<uint> streamIds1{0,2};
        std::vector<uint> streamIds2{3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                 &streamIds0[0], streamIds0.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr1),
                 &streamIds1[0], streamIds1.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr2),
                 &streamIds2[0], streamIds2.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
        
        WHEN( "The BranchBintrs are added to the RemuxerBintr" )
        {
            REQUIRE( pRemuxerBintr->SetBatchSize(4) == true );
            REQUIRE( pRemuxerBintr->LinkAll() == true );
            
            THEN( "The same BranchBintrs can be removed" )
            {
                REQUIRE( pRemuxerBintr->IsLinked() == true );
            }
        }
    }
}


SCENARIO( "BranchBintrs with select stream-ids can be linked to a RemuxerBintr multiple times",
    "[RemuxerBintr]" )
{
    GIVEN( "A new RemuxerBintr and BranchBintrs in memory" ) 
    {
        std::string remuxerBintrName = "remuxer";
        std::vector<uint> streamIds0{0,1,2,3};
        std::vector<uint> streamIds1{0,2};
        std::vector<uint> streamIds2{3};

        DSL_REMUXER_PTR pRemuxerBintr = 
            DSL_REMUXER_NEW(remuxerBintrName.c_str());

        DSL_BRANCH_PTR pBranchBintr0 = DSL_BRANCH_NEW(branchBintrName0.c_str());
        DSL_BRANCH_PTR pBranchBintr1 = DSL_BRANCH_NEW(branchBintrName1.c_str());
        DSL_BRANCH_PTR pBranchBintr2 = DSL_BRANCH_NEW(branchBintrName2.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr0 = DSL_FAKE_SINK_NEW(sinkName0.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr1 = DSL_FAKE_SINK_NEW(sinkName1.c_str());
        DSL_FAKE_SINK_PTR pSinkBintr2 = DSL_FAKE_SINK_NEW(sinkName2.c_str());

        REQUIRE( pSinkBintr0->AddToParent(pBranchBintr0) == true );
        REQUIRE( pSinkBintr1->AddToParent(pBranchBintr1) == true );
        REQUIRE( pSinkBintr2->AddToParent(pBranchBintr2) == true );

        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0),
                 &streamIds0[0], streamIds0.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr0)) == true );
        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr1),
                 &streamIds1[0], streamIds1.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr1)) == true );
        REQUIRE( pRemuxerBintr->AddChildTo(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr2),
                 &streamIds2[0], streamIds2.size()) == true );
        REQUIRE( pRemuxerBintr->IsChild(
            std::dynamic_pointer_cast<Bintr>(pBranchBintr2)) == true );
        
        WHEN( "The RemuxerBintr is linked and unlinked multiple times" )
        {
            for (auto i=0; i<10; i++)
            {
                REQUIRE( pRemuxerBintr->SetBatchSize(4) == true );
                REQUIRE( pRemuxerBintr->LinkAll() == true );
                REQUIRE( pRemuxerBintr->IsLinked() == true );
                pRemuxerBintr->UnlinkAll();
                REQUIRE( pRemuxerBintr->IsLinked() == false );
            }
            
            THEN( "The final linked state is correct" )
            {
                REQUIRE( pRemuxerBintr->IsLinked() == false );
            }
        }
    }
}
