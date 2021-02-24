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
#include "DslSinkBintr.h"

using namespace DSL;

SCENARIO( "A MultiSinksBintr is created correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A name for a MultiSinksBintr" ) 
    {
        std::string multiSinksBintrName = "multi-sinks";

        WHEN( "The MultiSinksBintr is created" )
        {
            DSL_MULTI_SINKS_PTR pMultiSinksBintr = 
                DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pMultiSinksBintr->GetName() == multiSinksBintrName );
                REQUIRE( pMultiSinksBintr->GetNumChildren() == 0 );
            }
        }
    }
}

SCENARIO( "Adding a single Sink to a MultiSinksBintr is managed correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new MultiSinksBintr and new Sink Bintr in memory" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);
        std::string sinkName = "window-sink";
        std::string multiSinksBintrName = "multi-sinks";

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());

        REQUIRE( pMultiSinksBintr->GetNumChildren() == 0 );
            
        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
        REQUIRE( pSinkBintr->GetId() == -1 );

        WHEN( "The Sink Bintr is added to the MultiSinksBintr" )
        {
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr)) == true );
            
            THEN( "The MultiSinksBintr is updated correctly" )
            {
                REQUIRE( pMultiSinksBintr->GetNumChildren() == 1 );
                REQUIRE( pSinkBintr->IsInUse() == true );
                REQUIRE( pSinkBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Removing a single Sink from a MultiSinksBintr is managed correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new MultiSinksBintr with a new Sink Bintr" ) 
    {
        std::string sinkName = "window-sink";
        std::string multiSinksBintrName = "multi-sinks";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());
            
        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr)) == true );
        REQUIRE( pSinkBintr->GetId() == -1 );
            
        WHEN( "The Sink Bintr is removed from the MultiSinksBintr" )
        {
            REQUIRE( pMultiSinksBintr->RemoveChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr)) == true );  
            
            THEN( "The MultiSinksBintr is updated correctly" )
            {
                REQUIRE( pMultiSinksBintr->GetNumChildren() == 0 );
                REQUIRE( pSinkBintr->IsInUse() == false );
                REQUIRE( pSinkBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Linking multiple sinks to a MultiSinksBintr Tee is managed correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new MultiSinksBintr with several new Sink Bintrs" ) 
    {
        std::string multiSinksBintrName = "multi-sinks";

        std::string sinkName0 = "window-sink-0";
        std::string sinkName1 = "window-sink-1";
        std::string sinkName2 = "window-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());
            
        DSL_WINDOW_SINK_PTR pSinkBintr0 = 
            DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr0->GetId() == -1 );

        DSL_WINDOW_SINK_PTR pSinkBintr1 = 
            DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr1->GetId() == -1 );

        DSL_WINDOW_SINK_PTR pSinkBintr2 = 
            DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr2->GetId() == -1 );

        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr0)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr1)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr2)) == true );

        REQUIRE( pMultiSinksBintr->GetNumChildren() == 3 );
            
        WHEN( "The Sink Bintrs are linked to the MultiSinksBintr" )
        {
            REQUIRE( pMultiSinksBintr->LinkAll()  == true );
            
            THEN( "The MultiSinksBintr is updated correctly" )
            {
                REQUIRE( pSinkBintr0->IsInUse() == true );
                REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr0->GetId() == 0 );
                REQUIRE( pSinkBintr1->IsInUse() == true );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr1->GetId() == 1 );
                REQUIRE( pSinkBintr2->IsInUse() == true );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
                REQUIRE( pSinkBintr2->GetId() == 2 );
            }
        }
    }
}

SCENARIO( "Multiple sinks linked to a MultiSinksBintr Tee can be unlinked correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new MultiSinksBintr with several new Sink Bintrs all linked" ) 
    {
        std::string multiSinksBintrName = "multi-sinks";

        std::string sinkName0 = "window-sink-0";
        std::string sinkName1 = "window-sink-1";
        std::string sinkName2 = "window-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());
            
        DSL_WINDOW_SINK_PTR pSinkBintr0 = 
            DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_WINDOW_SINK_PTR pSinkBintr1 = 
            DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_WINDOW_SINK_PTR pSinkBintr2 = 
            DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr0)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr1)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr2)) == true );

        REQUIRE( pMultiSinksBintr->GetNumChildren() == 3 );
        REQUIRE( pMultiSinksBintr->LinkAll()  == true );

        REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr0->GetId() == 0 );
        REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr1->GetId() == 1 );
        REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr2->GetId() == 2 );

        WHEN( "The MultiSinksBintr and child SinkBintrs are unlinked" )
        {
            pMultiSinksBintr->UnlinkAll();
            THEN( "The MultiSinksBintr is updated correctly" )
            {
                REQUIRE( pSinkBintr0->IsLinkedToSource() == false );
                REQUIRE( pSinkBintr0->GetId() == -1 );
                REQUIRE( pSinkBintr1->IsLinkedToSource() == false );
                REQUIRE( pSinkBintr1->GetId() == -1 );
                REQUIRE( pSinkBintr2->IsLinkedToSource() == false );
                REQUIRE( pSinkBintr2->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "All GST Resources are released on MultiSinksBintr destruction", "[MultiSinksBintr]" )
{
    GIVEN( "Attributes for a new MultiSinksBintr and several new SinkBintrs" ) 
    {
        std::string multiSinksBintrName = "multi-sinks";

        std::string sinkName0 = "window-sink-0";
        std::string sinkName1 = "window-sink-1";
        std::string sinkName2 = "window-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        WHEN( "The Bintrs are created and the Components are added as children and linked" )
        {
            DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());
                
            DSL_WINDOW_SINK_PTR pSinkBintr0 = 
                DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr1 = 
                DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr2 = 
                DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr0)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr1)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr2)) == true );

            REQUIRE( pMultiSinksBintr->GetNumChildren() == 3 );
            REQUIRE( pMultiSinksBintr->LinkAll()  == true );

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
            DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(multiSinksBintrName.c_str());
                
            DSL_WINDOW_SINK_PTR pSinkBintr0 = 
                DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr1 = 
                DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr2 = 
                DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr0)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr1)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<Bintr>(pSinkBintr2)) == true );
            pMultiSinksBintr->LinkAll();

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
