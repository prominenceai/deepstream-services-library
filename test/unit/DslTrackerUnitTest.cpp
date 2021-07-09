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
                REQUIRE( pTrackerBintr->GetBatchProcessingEnabled() == true );
                REQUIRE( pTrackerBintr->GetPastFrameReportingEnabled() == true );
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
    
        pTrackerBintr->GetDimensions(&currWidth, &currHeight);
        REQUIRE( currWidth == initWidth );
        REQUIRE( currHeight == initHeight );

        WHEN( "The Trackers's demensions are Set" )
        {
            uint newWidth(300);
            uint newHeight(150);
            
            pTrackerBintr->SetDimensions(newWidth, newHeight);

            THEN( "The Display's new demensions are returned on Get")
            {
                pTrackerBintr->GetDimensions(&currWidth, &currHeight);
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );
            }
        }
    }
}

SCENARIO( "A Tracker's enable-patch-processing and enable-past-frame settings can be updated", "[TrackerBintr]" )
{
    GIVEN( "A new Tracker in memory" ) 
    {
        std::string trackerName("ktl-tracker");
        uint width(640);
        uint height(368);

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), width, height);

        WHEN( "The Trackers's demensions are Set" )
        {
            REQUIRE( pTrackerBintr->SetBatchProcessingEnabled(false) == true );
            REQUIRE( pTrackerBintr->SetPastFrameReportingEnabled(false) == true );

            THEN( "The Display's new demensions are returned on Get")
            {
                REQUIRE( pTrackerBintr->GetBatchProcessingEnabled() == false );
                REQUIRE( pTrackerBintr->GetPastFrameReportingEnabled() == false );
            }
        }
    }
}


