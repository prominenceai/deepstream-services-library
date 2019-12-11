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
#include "DslTrackerBintr.h"

using namespace DSL;

SCENARIO( "A KTL Tracker is created correctly", "[TrackerBintr]" )
{
    GIVEN( "Attributes for a new KTL Tracker" ) 
    {
        std::string trackerName("ktl-tracker");
        uint width(200);
        uint height(100);

        WHEN( "The KTL Tracker is created" )
        {
            DSL_KTL_TRACKER_PTR pTrackerBintr = 
                DSL_KTL_TRACKER_NEW(trackerName.c_str(), width, height);

            THEN( "The KTL Tracker's lib is found, loaded, and returned correctly")
            {
                std::string defPathSpec(NVDS_KLT_LIB);
                std::string retPathSpec(pTrackerBintr->GetLibFile());
                
                REQUIRE( retPathSpec == defPathSpec );
            }
        }
    }
}

SCENARIO( "An IOU Tracker is created correctly", "[TrackerBintr]" )
{
    GIVEN( "Attributes for a new IOU Tracker" ) 
    {
        std::string trackerName("iou-tracker");
        uint width(200);
        uint height(100);
        std::string defConfigFile("./test/configs/iou_config.txt");

        WHEN( "The IOU Tracker is created" )
        {
            DSL_IOU_TRACKER_PTR pTrackerBintr = 
                DSL_IOU_TRACKER_NEW(trackerName.c_str(), defConfigFile.c_str(), width, height);

            THEN( "The IOU Tracker's lib is found, loaded, and returned correctly")
            {
                std::string defLibPathSpec(NVDS_IOU_LIB);
                std::string retLibPathSpec(pTrackerBintr->GetLibFile());
                std::string retConfigPathSpec(pTrackerBintr->GetConfigFile());
                REQUIRE( retLibPathSpec == defLibPathSpec );
                REQUIRE( retConfigPathSpec == defConfigFile );
            }
        }
    }
}

SCENARIO( "A Tracker's dimensions can be updated", "[TrackerBintr]" )
{
    GIVEN( "A new Tracker in memory" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);
            
        uint currWidth(0);
        uint currHeight(0);
    
        pTrackerBintr->GetMaxDimensions(&currWidth, &currHeight);
        REQUIRE( currWidth == initWidth );
        REQUIRE( currHeight == initHeight );

        WHEN( "The Trackers's demensions are Set" )
        {
            uint newWidth(300);
            uint newHeight(150);
            
            pTrackerBintr->SetMaxDimensions(newWidth, newHeight);

            THEN( "The Display's new demensions are returned on Get")
            {
                pTrackerBintr->GetMaxDimensions(&currWidth, &currHeight);
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );
            }
        }
    }
}

SCENARIO( "A Tracker can add a Batch Meta Handler", "[TrackerBintr]" )
{
    GIVEN( "A new Tracker in memory" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        dsl_batch_meta_handler_cb handler;

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        REQUIRE( pTrackerBintr->GetBatchMetaHandler() == NULL );
        
        WHEN( "The Tracker is called to add a Batch Meta Handler" )
        {
            REQUIRE( pTrackerBintr->AddBatchMetaHandler(handler, NULL) == true );

            THEN( "The Tracker is able to return the same Handler on get" )
            {
                REQUIRE( pTrackerBintr->GetBatchMetaHandler() == handler );
            }
        }
    }
}

SCENARIO( "A Tracker can remove a Batch Meta Handler", "[TrackerBintr]" )
{
    GIVEN( "A new Tracker with an existing Batch Meta Handler" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        dsl_batch_meta_handler_cb handler((dsl_batch_meta_handler_cb)0x01);

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        REQUIRE( pTrackerBintr->GetBatchMetaHandler() == NULL );
        REQUIRE( pTrackerBintr->AddBatchMetaHandler(handler, NULL) == true );

        WHEN( "After the Tracker is called to remove the Batch Meta Handler" )
        {
            REQUIRE( pTrackerBintr->RemoveBatchMetaHandler() == true );
            
            THEN( "The Tracker returns NULL when queried" )
            {
                REQUIRE( pTrackerBintr->GetBatchMetaHandler() == NULL );
            }
        }
    }
}

SCENARIO( "A Tracker can have at most one Batch Meta Handler", "[TrackerBintr]" )
{
    GIVEN( "A new Tracker with an existing Batch Meta Handler" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        dsl_batch_meta_handler_cb handler1((dsl_batch_meta_handler_cb)0x01);
        dsl_batch_meta_handler_cb handler2((dsl_batch_meta_handler_cb)0x02);

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        REQUIRE( pTrackerBintr->GetBatchMetaHandler() == NULL );

        WHEN( "A Batch Meta Handler has been added to the Tracker" )
        {
            REQUIRE( pTrackerBintr->AddBatchMetaHandler(handler1, NULL) == true );
            
            THEN( "A second Batch Meta Handler can not be added " )
            {
                REQUIRE( pTrackerBintr->AddBatchMetaHandler(handler2, NULL) == false );
                REQUIRE( pTrackerBintr->GetBatchMetaHandler() == handler1 );
            }
        } 
    }
}
