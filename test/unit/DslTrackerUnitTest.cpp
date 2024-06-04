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

// Filespec for the NvDCF Tracker config file
static const std::string dcfTrackerConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml");

static const std::string iouTrackerConfigFile("/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.ym");

using namespace DSL;

SCENARIO( "A DCF Tracker is created correctly", "[TrackerBintr]" )
{
    GIVEN( "Attributes for a new DCF Tracker" ) 
    {
        std::string trackerName("dcf-tracker");
        uint width(32);
        uint height(32);
        bool batchProcessingEnabled(true);
        bool pastFrameReportingEnabled(false);

        WHEN( "The DCF Tracker is created" )
        {
            DSL_TRACKER_PTR pTrackerBintr = 
                DSL_TRACKER_NEW(trackerName.c_str(), dcfTrackerConfigFile.c_str(), width, height);

            THEN( "The DCF Tracker's lib is found, loaded, and returned correctly")
            {
                std::string defTrackerLibFile(NVDS_MOT_LIB);
                
                std::string retTrackerLibFile = pTrackerBintr->GetLibFile();
                REQUIRE( retTrackerLibFile == defTrackerLibFile );
                
                std::string retTrackerConfigFile = pTrackerBintr->GetConfigFile();
                REQUIRE( retTrackerConfigFile == dcfTrackerConfigFile );
                
                uint retWidth(0), retHeight(0);
                pTrackerBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == width );
                REQUIRE( retHeight == height );
                
                boolean retInputEnabled(true);
                const char* cRetTrackOnGie;
                
                pTrackerBintr->GetTensorMetaSettings(&retInputEnabled,
                    &cRetTrackOnGie);
                std::string retTrackOnGie(cRetTrackOnGie);
                REQUIRE( retInputEnabled == 0 );
                REQUIRE( retTrackOnGie == "" );
                
                REQUIRE( pTrackerBintr->GetIdDisplayEnabled() == 1 );

                REQUIRE( pTrackerBintr->GetQueueLeaky() == DSL_COMPONENT_QUEUE_LEAKY_NO );
                REQUIRE( pTrackerBintr->GetQueueCurrentLevel(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 0 );
                REQUIRE( pTrackerBintr->GetQueueCurrentLevel(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 0 );
                REQUIRE( pTrackerBintr->GetQueueCurrentLevel(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 200 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 10485760 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 1000000000 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 0 );
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

        WHEN( "The IOU Tracker is created" )
        {
            DSL_TRACKER_PTR pTrackerBintr = 
                DSL_TRACKER_NEW(trackerName.c_str(), iouTrackerConfigFile.c_str(), width, height);

            THEN( "The IOU Tracker's lib is found, loaded, and returned correctly")
            {
                std::string defTrackerLibFile(NVDS_MOT_LIB);
                
                std::string retTrackerLibFile = pTrackerBintr->GetLibFile();
                REQUIRE( retTrackerLibFile == defTrackerLibFile );
                
                std::string retTrackerConfigFile = pTrackerBintr->GetConfigFile();
                REQUIRE( retTrackerConfigFile == iouTrackerConfigFile );

                uint retWidth(0), retHeight(0);
                pTrackerBintr->GetDimensions(&retWidth, &retHeight);
                REQUIRE( retWidth == width );
                REQUIRE( retHeight == height );
                
                REQUIRE( pTrackerBintr->GetQueueLeaky() == DSL_COMPONENT_QUEUE_LEAKY_NO );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 200 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 10485760 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 1000000000 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 0 );
            }
        }
    }
}

SCENARIO( "A Tracker's dimensions can be updated", "[TrackerBintr]" )
{
    GIVEN( "A new Tracker in memory" ) 
    {
        std::string trackerName("tracker");
        uint initWidth(200);
        uint initHeight(100);

        DSL_TRACKER_PTR pTrackerBintr = 
            DSL_TRACKER_NEW(trackerName.c_str(), "", initWidth, initHeight);
            
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

