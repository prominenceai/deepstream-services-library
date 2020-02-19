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
#include "DslMultiSinksBintr.h"
#include "DslSinkBintr.h"

using namespace DSL;

SCENARIO( "A MultiSinksBintr is created correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A name for a MultiSinksBintr" ) 
    {
        std::string MultiSinksBintrName = "multi-sinks";

        WHEN( "The MultiSinksBintr is created" )
        {
            DSL_MULTI_SINKS_PTR pMultiSinksBintr = 
                DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pMultiSinksBintr->GetName() == MultiSinksBintrName );
                REQUIRE( pMultiSinksBintr->GetNumChildren() == 0 );
            }
        }
    }
}

SCENARIO( "Adding a single Sink to a Multi Sinks Bintr is managed correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new Multi Sinks Bintr and new Sink Bintr in memory" ) 
    {
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);
        std::string sinkName = "window-sink";
        std::string MultiSinksBintrName = "multi-sinks";

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());

        REQUIRE( pMultiSinksBintr->GetNumChildren() == 0 );
            
        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);
            
        REQUIRE( pSinkBintr->GetSinkId() == -1 );

        WHEN( "The Sink Bintr is added to the Multi Sinks Bintr" )
        {
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );
            
            THEN( "The Multi Sinks Bintr is updated correctly" )
            {
                REQUIRE( pMultiSinksBintr->GetNumChildren() == 1 );
                REQUIRE( pSinkBintr->IsInUse() == true );
                REQUIRE( pSinkBintr->GetSinkId() == -1 );
            }
        }
    }
}

SCENARIO( "Removing a single Sink from a Multi Sinks Bintr is managed correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new Multi Sinks Bintr with a new Sink Bintr" ) 
    {
        std::string sinkName = "window-sink";
        std::string MultiSinksBintrName = "multi-sinks";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());
            
        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );
        REQUIRE( pSinkBintr->GetSinkId() == -1 );
            
        WHEN( "The Sink Bintr is removed from the Multi Sinks Bintr" )
        {
            REQUIRE( pMultiSinksBintr->RemoveChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)) == true );  
            
            THEN( "The Multi Sinks Bintr is updated correctly" )
            {
                REQUIRE( pMultiSinksBintr->GetNumChildren() == 0 );
                REQUIRE( pSinkBintr->IsInUse() == false );
                REQUIRE( pSinkBintr->GetSinkId() == -1 );
            }
        }
    }
}

SCENARIO( "Linking multiple sinks to a Multi Sinks Bintr Tee is managed correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new Multi Sinks Bintr with several new Sink Bintrs" ) 
    {
        std::string MultiSinksBintrName = "multi-sinks";

        std::string sinkName0 = "window-sink-0";
        std::string sinkName1 = "window-sink-1";
        std::string sinkName2 = "window-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());
            
        DSL_WINDOW_SINK_PTR pSinkBintr0 = 
            DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr0->GetSinkId() == -1 );

        DSL_WINDOW_SINK_PTR pSinkBintr1 = 
            DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr1->GetSinkId() == -1 );

        DSL_WINDOW_SINK_PTR pSinkBintr2 = 
            DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);
        REQUIRE( pSinkBintr2->GetSinkId() == -1 );

        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );

        REQUIRE( pMultiSinksBintr->GetNumChildren() == 3 );
            
        WHEN( "The Sink Bintrs are linked to the Multi Sinks Bintr" )
        {
            REQUIRE( pMultiSinksBintr->LinkAll()  == true );
            
            THEN( "The Multi Sinks Bintr is updated correctly" )
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

SCENARIO( "Multiple sinks linked to a Multi Sinks Bintr Tee can be unlinked correctly", "[MultiSinksBintr]" )
{
    GIVEN( "A new Multi Sinks Bintr with several new Sink Bintrs all linked" ) 
    {
        std::string MultiSinksBintrName = "multi-sinks";

        std::string sinkName0 = "window-sink-0";
        std::string sinkName1 = "window-sink-1";
        std::string sinkName2 = "window-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());
            
        DSL_WINDOW_SINK_PTR pSinkBintr0 = 
            DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_WINDOW_SINK_PTR pSinkBintr1 = 
            DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_WINDOW_SINK_PTR pSinkBintr2 = 
            DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
        REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );

        REQUIRE( pMultiSinksBintr->GetNumChildren() == 3 );
        REQUIRE( pMultiSinksBintr->LinkAll()  == true );

        REQUIRE( pSinkBintr0->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr0->GetSinkId() == 0 );
        REQUIRE( pSinkBintr1->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr1->GetSinkId() == 1 );
        REQUIRE( pSinkBintr2->IsLinkedToSource() == true );
        REQUIRE( pSinkBintr2->GetSinkId() == 2 );

        WHEN( "The MultiSinksBintr and child SinkBintrs are unlinked" )
        {
            pMultiSinksBintr->UnlinkAll();
            THEN( "The Multi Sinks Bintr is updated correctly" )
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

SCENARIO( "All GST Resources are released on MultiSinksBintr destruction", "[MultiSinksBintr]" )
{
    GIVEN( "Attributes for a new MultiSinksBintr and several new SinkBintrs" ) 
    {
        std::string MultiSinksBintrName = "multi-sinks";

        std::string sinkName0 = "window-sink-0";
        std::string sinkName1 = "window-sink-1";
        std::string sinkName2 = "window-sink-2";
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        WHEN( "The Bintrs are created and the Sinks are added as children and linked" )
        {
            DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());
                
            DSL_WINDOW_SINK_PTR pSinkBintr0 = 
                DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr1 = 
                DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr2 = 
                DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );

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
            DSL_MULTI_SINKS_PTR pMultiSinksBintr = DSL_MULTI_SINKS_NEW(MultiSinksBintrName.c_str());
                
            DSL_WINDOW_SINK_PTR pSinkBintr0 = 
                DSL_WINDOW_SINK_NEW(sinkName0.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr1 = 
                DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_WINDOW_SINK_PTR pSinkBintr2 = 
                DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, sinkW, sinkH);

            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr0)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr1)) == true );
            REQUIRE( pMultiSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr2)) == true );
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
