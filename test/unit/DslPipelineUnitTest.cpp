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
#include "DslTilerBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslPipelineBintr.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

using namespace DSL;

static const std::string sourceName("source");
static const std::string sourceName1("source-1");
static const std::string sourceName2("source-2");
static const std::string sourceName3("source-3");
static const std::string sourceName4("source-4");

static const std::string filePath(
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");

static const std::string imagePath("/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");

static const std::string sinkName("sink");
static const std::string sinkName1("sink-1");
static const std::string sinkName2("sink-2");
static const std::string sinkName3("sink-3");
static const std::string sinkName4("sink-4");

static const std::string tilerName("tiler");
static const std::string pipelineName("pipeline");
static const std::string primaryGieName("primary-gie");
static const std::string primaryInferConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static const std::string primaryModelEngineFile(
    "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");
    
static const std::string demuxerName("demuxer");
static const std::string trackerName("iou-tracker");
static const uint trackerW(300);
static const uint trackerH(150);

static const std::string secondaryGieName("secondary-gie");
static const std::string secondaryInferConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carmake.txt");
static const std::string secondaryModelEngineFile(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarMake/resnet18.caffemodel_b8_gpu0_fp16.engine");

static const std::string osdName("on-screen-tiler");
        
static const uint sourceW(1280);
static const uint sourceH(720);
static const uint interval(1);
static const uint fps_n(1);
static const uint fps_d(30);
static const uint tilerW(1280);
static const uint tilerH(720);
static const uint displayId(0);
static const uint depth(0);
static const uint offsetX(0);
static const uint offsetY(0);
static const uint windowW(1280);
static const uint windowH(720);

SCENARIO( "A New PipelineBintr is created correctly", "[PipelineBintr]" )
{
    GIVEN( "A new name for a PipelineBintr" ) 
    {
        std::string pipelineName = "pipeline";

        WHEN( "The new PipelineBintr is created" )
        {
            DSL_PIPELINE_PTR pPipelineBintr = 
                DSL_PIPELINE_NEW(pipelineName.c_str());

            THEN( "All member variables are setup correctly" )
            {
                GstState state;
                REQUIRE( pPipelineBintr->GetName() == pipelineName );
                pPipelineBintr->GetState(state, 0);
                REQUIRE( state == DSL_STATE_NULL );
                REQUIRE( pPipelineBintr->IsLive() == False );
            }
        }
    }
}

SCENARIO( "A New PipelineBintr will fail to LinkAll with insufficient Components", "[PipelineBintr]" )
{
    GIVEN( "A new CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());

        WHEN( "The PipelineBintr has a OverlaySinkBintr but no CsiSourceBintr" )
        {
            pSinkBintr->AddToParent(pPipelineBintr);

            THEN( "The Pipeline will fail to LinkAll" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == false );
            }
        }
        WHEN( "The PipelineBintr has a CsiSourceBintr but no OverlaySinkBintr" )
        {
            pSourceBintr->AddToParent(pPipelineBintr);

            THEN( "The Pipeline will fail to LinkAll" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == false );
            }
        }
    }
}

SCENARIO( "A PipelineBintr will fail to create an XWindow without setting dimensions first", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr with default XWindow dimensions of 0" ) 
    {
        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        WHEN( "The PipelineBintr is called to create its XWindow" )
        {
            REQUIRE( pPipelineBintr->CreateXWindow() == false );
                
            THEN( "The XWindow handle  is unavailable" )
            {
                REQUIRE( pPipelineBintr->GetXWindow() == 0 );
            }
        }
    }
}

SCENARIO( "A PipelineBintr's' XWindow is created correctly", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr with valid XWindow dimensions" ) 
    {
        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        pPipelineBintr->SetXWindowDimensions(windowW, windowH);

        WHEN( "The new PipelineBintr's XWindow is created" )
        {
            REQUIRE( pPipelineBintr->CreateXWindow() == true );
                
            THEN( "The XWindow handle is available" )
            {
                REQUIRE( pPipelineBintr->GetXWindow() != 0 );
            }
        }
    }
}

SCENARIO( "A PipelineBintr's' XWindow is created correctly in Full-Screen-Mode", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr with valid XWindow dimensions" ) 
    {
        boolean full_screen_enabled(1);

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        pPipelineBintr->SetXWindowDimensions(windowW, windowH);
        pPipelineBintr->SetXWindowFullScreenEnabled(full_screen_enabled);

        WHEN( "The new PipelineBintr's XWindow is created" )
        {
            REQUIRE( pPipelineBintr->CreateXWindow() == true );
                
            THEN( "The XWindow handle is available" )
            {
                REQUIRE( pPipelineBintr->GetXWindow() != 0 );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll with minimum Components ", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline components are Linked correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to UnlinkAll after linking with minimum Components ", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added and the PipelineBintr is Linked" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPipelineBintr->LinkAll() == true );

            THEN( "The Pipeline components are Linked correctly" )
            {
                pPipelineBintr->UnlinkAll();
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll with minimum Components and a PrimaryGieBintr", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, PrimaryGieBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline components are Linked correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
            }
        }
    }
}

SCENARIO( "A Pipeline is unable to LinkAll with a SecondaryGieBintr and no PrimaryGieBintr", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, SecondaryGieBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSecondaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPipelineBintr->IsLinked() == false );

            THEN( "The Pipeline is unable to LinkAll without a PrimaryGieBintr" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == false );
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll and UnlinkAll with a PrimaryGieBintr and OsdBintr", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, PrimaryGieBintr, OverlaySinkBintr, PipelineBintr, and OsdBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_OSD_PTR pOsdBintr = 
            DSL_OSD_NEW(osdName.c_str(), true, true, true, false);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pOsdBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline components are Linked correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
                REQUIRE( pPipelineBintr->IsLinked() == true );
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll and UnlinkAll with a PrimaryGieBintr, OsdBintr, and TrackerBintr", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, PrimaryGieBintr, OverlaySinkBintr, PipelineBintr, TrackerBintr, and OsdBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TRACKER_PTR pTrackerBintr = 
            DSL_TRACKER_NEW(trackerName.c_str(), "", trackerW, trackerH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OSD_PTR pOsdBintr = 
            DSL_OSD_NEW(osdName.c_str(), true, true, true, false);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTrackerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pOsdBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline components are able to Link and Unlink correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
                REQUIRE( pPipelineBintr->IsLinked() == true );
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll and UnlinkAll with all Optional Components", "[PipelineBintr]" )
{
    GIVEN( "A new TilerBintr, CsiSourceBintr, PrimaryGieBintr, SecondaryGieBintr, OsdBintr, OverlaySinkBintr, and PipelineBintr" ) 
    {

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TRACKER_PTR pTrackerBintr = 
            DSL_TRACKER_NEW(trackerName.c_str(), "", trackerW, trackerH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), secondaryInferConfigFile.c_str(), 
            secondaryModelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_OSD_PTR pOsdBintr = 
            DSL_OSD_NEW(osdName.c_str(), true, true, true, false);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTrackerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSecondaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pOsdBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline is able to LinkAll and UnlinkAll correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
                REQUIRE( pPipelineBintr->IsLinked() == true );
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}


SCENARIO( "A Pipeline can have at most one TilerBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new TilerBintrs and PipelineBintr" ) 
    {
        std::string tilerName1("tiler-1");
        std::string tilerName2("tiler-2");

        DSL_TILER_PTR pTilerBintr1 = 
            DSL_TILER_NEW(tilerName1.c_str(), tilerW, tilerH);

        DSL_TILER_PTR pTilerBintr2 = 
            DSL_TILER_NEW(tilerName2.c_str(), tilerW, tilerH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "A TilerBintr is added to the PipelineBintr" )
        {
            REQUIRE( pTilerBintr1->AddToParent(pPipelineBintr) == true );

            THEN( "A second TilerBintr can not be added" )
            {
                REQUIRE( pTilerBintr2->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "A Pipeline can have at most one TrackerBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new TrackerBintrs and PipelineBintr" ) 
    {
        std::string trackerName1("tracker-1");
        std::string trackerName2("tracker-2");

        DSL_TRACKER_PTR pTrackerBintr1 = 
            DSL_TRACKER_NEW(trackerName1.c_str(), "", trackerW, trackerH);

        DSL_TRACKER_PTR pTrackerBintr2 = 
            DSL_TRACKER_NEW(trackerName2.c_str(), "", trackerW, trackerH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "A TrackerBintr is added to the PipelineBintr" )
        {
            REQUIRE( pTrackerBintr1->AddToParent(pPipelineBintr) == true );

            THEN( "A second TrackerBintr can not be added" )
            {
                REQUIRE( pTrackerBintr2->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "A Pipeline can have at most one PrimaryGieBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new PrimaryGieBintrs and PipelineBintr" ) 
    {
        std::string primaryGieName1("primary-gie-1");
        std::string primaryGieName2("primary-gie-2");

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr1 = 
            DSL_PRIMARY_GIE_NEW(primaryGieName1.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr2 = 
            DSL_PRIMARY_GIE_NEW(primaryGieName2.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "A PrimaryGieBintr is added to the PipelineBintr" )
        {
            REQUIRE( pPrimaryGieBintr1->AddToParent(pPipelineBintr) == true );

            THEN( "A second PrimaryGieBintr can not be added" )
            {
                REQUIRE( pPrimaryGieBintr2->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "A Pipeline can have at most one OsdBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new OsdBintrs and PipelineBintr" ) 
    {
        std::string osdName1("on-screen-display-1");
        std::string osdName2("on-screen-display-2");

        DSL_OSD_PTR pOsdBintr1 = 
            DSL_OSD_NEW(osdName1.c_str(), true, true, true, false);

        DSL_OSD_PTR pOsdBintr2 = 
            DSL_OSD_NEW(osdName2.c_str(), true, true, true, false);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "A OsdBintr is added to the PipelineBintr" )
        {
            REQUIRE( pOsdBintr1->AddToParent(pPipelineBintr) == true );

            THEN( "A second OsdBintr can not be added" )
            {
                REQUIRE( pOsdBintr2->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "A Pipeline can have at most one DemuxerBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new DemuxerBintrs and PipelineBintr" ) 
    {
        std::string demuxerName1("demuxer-1");
        std::string demuxerName2("demuxer-2");

        DSL_DEMUXER_PTR pDemuxerBintr1 = 
            DSL_DEMUXER_NEW(demuxerName1.c_str());

        DSL_DEMUXER_PTR pDemuxerBintr2 = 
            DSL_DEMUXER_NEW(demuxerName2.c_str());

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "A DemuxerBintr is added to the PipelineBintr" )
        {
            REQUIRE( pDemuxerBintr1->AddToParent(pPipelineBintr) == true );

            THEN( "A second DemuxerBintr can not be added" )
            {
                REQUIRE( pDemuxerBintr2->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "Adding a DemuxerBintr to a PipelineBintr with a TilerBintr fails", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr, TilerBinter and Demuxer" ) 
    {
        DSL_DEMUXER_PTR pDemuxerBintr = 
            DSL_DEMUXER_NEW(demuxerName.c_str());

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        WHEN( "A TilerBintr is added to the PipelineBintr" )
        {
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            
            THEN( "Adding a DemuxerBintr to the PipelineBintr after fails" )
            {
                REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "Adding a TilerBintr to a PipelineBintr with a DemuxerBintr fails", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr, TilerBinter and Demuxer" ) 
    {
        DSL_DEMUXER_PTR pDemuxerBintr = 
            DSL_DEMUXER_NEW(demuxerName.c_str());

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        WHEN( "A DemuxerBintr is added to the PipelineBintr" )
        {
            REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );
            
            THEN( "Adding a TilerBintr to the PipelineBintr after fails" )
            {
                REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "Adding an OsdBintr to a PipelineBintr with a DemuxerBintr fails", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr, TilerBinter and Demuxer" ) 
    {
        DSL_DEMUXER_PTR pDemuxerBintr = 
            DSL_DEMUXER_NEW(demuxerName.c_str());

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        DSL_OSD_PTR pOsdBintr = DSL_OSD_NEW(osdName.c_str(), 
            false, false, false, false);

        WHEN( "A DemuxerBintr is added to the PipelineBintr" )
        {
            REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );
            
            THEN( "Adding an OsdBintr to the PipelineBintr after fails" )
            {
                REQUIRE( pOsdBintr->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll with a Demuxer and minimum Components", "[PipelineBintr]" )
{
    GIVEN( "A new DemuxerBintr, CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), sourceW, sourceH, fps_n, fps_d);

        // Note: need to use Bintr pointer when calling DemuxerBinter->AddChild() - non-ambiguious
        DSL_BINTR_PTR pDemuxerBintr = std::shared_ptr<Bintr>(new DemuxerBintr(demuxerName.c_str()));

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());

        REQUIRE( pPipelineBintr->IsLinked() == false );
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pDemuxerBintr->AddChild(pSinkBintr) == true );
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline components are Linked correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
                REQUIRE( pPipelineBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to UnlinkAll with a Demuxer and minimum Components", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr with DemuxerBintr, CsiSourceBintr, OverlaySinkBintr in a Linked State" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), sourceW, sourceH, fps_n, fps_d);

        // Note: need to use Bintr pointer when calling DemuxerBinter->AddChild() - non-ambiguious
        DSL_BINTR_PTR pDemuxerBintr = std::shared_ptr<Bintr>(new DemuxerBintr(demuxerName.c_str()));

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth,  offsetX, offsetY, windowW, windowH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr) == true );
        REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );

        REQUIRE( pPipelineBintr->IsLinked() == false );

        WHEN( "The Pipeline is in a Linked State" )
        {
            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );

            THEN( "The Pipeline can be unlinked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll/UnlinkAll with a Demuxer and multiple Sources with Sinks", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr with DemuxerBintr and multiple CsiSourceBintrs and WindowSinkBintrs in a Linked State" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr1 = 
            DSL_CSI_SOURCE_NEW(sourceName1.c_str(), sourceW, sourceH, fps_n, fps_d);
        DSL_WINDOW_SINK_PTR pSinkBintr1 = 
            DSL_WINDOW_SINK_NEW(sinkName1.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_CSI_SOURCE_PTR pSourceBintr2 = 
            DSL_CSI_SOURCE_NEW(sourceName2.c_str(), sourceW, sourceH, fps_n, fps_d);
        DSL_WINDOW_SINK_PTR pSinkBintr2 = 
            DSL_WINDOW_SINK_NEW(sinkName2.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_CSI_SOURCE_PTR pSourceBintr3 = 
            DSL_CSI_SOURCE_NEW(sourceName3.c_str(), sourceW, sourceH, fps_n, fps_d);
        DSL_WINDOW_SINK_PTR pSinkBintr3 = 
            DSL_WINDOW_SINK_NEW(sinkName3.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_CSI_SOURCE_PTR pSourceBintr4 = 
            DSL_CSI_SOURCE_NEW(sourceName4.c_str(), sourceW, sourceH, fps_n, fps_d);
        DSL_WINDOW_SINK_PTR pSinkBintr4 = 
            DSL_WINDOW_SINK_NEW(sinkName4.c_str(), offsetX, offsetY, windowW, windowH);

        // Note: need to use Bintr pointer when calling DemuxerBinter->AddChild() - non-ambiguious
        DSL_BINTR_PTR pDemuxerBintr = std::shared_ptr<Bintr>(new DemuxerBintr(demuxerName.c_str()));

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr1) == true );
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr2) == true );
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr3) == true );
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr4) == true );
        REQUIRE( pSourceBintr1->AddToParent(pPipelineBintr) == true );
        REQUIRE( pSourceBintr2->AddToParent(pPipelineBintr) == true );
        REQUIRE( pSourceBintr3->AddToParent(pPipelineBintr) == true );
        REQUIRE( pSourceBintr4->AddToParent(pPipelineBintr) == true );
        REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );

        REQUIRE( pPipelineBintr->IsLinked() == false );

        WHEN( "The Pipeline is in a Linked State" )
        {
            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );

            THEN( "The Pipeline can be unlinked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll/UnlinkAll with a Demuxer and Primary GIE", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr,  DemuxerBintr, CsiSourceBintr, WindowSinkBintr, Primary GIE" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), sourceW, sourceH, fps_n, fps_d);

        // Note: need to use Bintr pointer when calling DemuxerBinter->AddChild() - non-ambiguious
        DSL_BINTR_PTR pDemuxerBintr = std::shared_ptr<Bintr>(new DemuxerBintr(demuxerName.c_str()));

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr) == true );
        REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );

        REQUIRE( pPipelineBintr->IsLinked() == false );

        WHEN( "The Pipeline is Linked with the Demuxer and Primary GIE" )
        {
            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );

            THEN( "The Pipeline can be unlinked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll/UnlinkAll with a Demuxer, Primary GIE, and Tracker", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr, DemuxerBintr, CsiSourceBintr, WindowSinkBintr, PrimaryGieBintr, TrackerBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), sourceW, sourceH, fps_n, fps_d);

        // Note: need to use Bintr pointer when calling DemuxerBinter->AddChild() - non-ambiguious
        DSL_BINTR_PTR pDemuxerBintr = std::shared_ptr<Bintr>(new DemuxerBintr(demuxerName.c_str()));

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_TRACKER_PTR pTrackerBintr = 
            DSL_TRACKER_NEW(trackerName.c_str(), "", trackerW, trackerH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr) == true );
        REQUIRE( pTrackerBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );

        REQUIRE( pPipelineBintr->IsLinked() == false );

        WHEN( "The Pipeline is Linked with the Demuxer, Primary GIE, and Tracker" )
        {
            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );

            THEN( "The Pipeline can be unlinked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline is able to LinkAll/UnlinkAll with a Demuxer, Primary GIE, Tracker and Secondary GIE", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr, DemuxerBintr, CsiSourceBintr, WindowSinkBintr, PrimaryGieBintr, TrackerBintr, SecondaryGieBintr" ) 
    {
        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), sourceW, sourceH, fps_n, fps_d);

        // Note: need to use Bintr pointer when calling DemuxerBinter->AddChild() - non-ambiguious
        DSL_BINTR_PTR pDemuxerBintr = std::shared_ptr<Bintr>(new DemuxerBintr(demuxerName.c_str()));

        DSL_WINDOW_SINK_PTR pSinkBintr = 
            DSL_WINDOW_SINK_NEW(sinkName.c_str(), offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), secondaryInferConfigFile.c_str(), 
            secondaryModelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_TRACKER_PTR pTrackerBintr = 
            DSL_TRACKER_NEW(trackerName.c_str(), "", trackerW, trackerH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        REQUIRE( pDemuxerBintr->AddChild(pSinkBintr) == true );
        REQUIRE( pTrackerBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pDemuxerBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
        REQUIRE( pSecondaryGieBintr->AddToParent(pPipelineBintr) == true );

        REQUIRE( pPipelineBintr->IsLinked() == false );

        WHEN( "The Pipeline is Linked with the Demuxer, Primary GIE, and Tracker" )
        {
            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );

            THEN( "The Pipeline can be unlinked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline with an ImageStreamSourceBintr is able to Link/UnlinkAll", "[PipelineBintr]" )
{
    GIVEN( "A new ImageStreamSourceBintr, PrimaryGieBintr, TilerBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        bool isLive(true);
        uint timeout(0);

        DSL_IMAGE_STREAM_SOURCE_PTR pSourceBintr = 
            DSL_IMAGE_STREAM_SOURCE_NEW(sourceName.c_str(), imagePath.c_str(), isLive, fps_n, fps_d, timeout);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());

        REQUIRE( pPipelineBintr->IsLinked() == false );
            
        WHEN( "All components are added and PipelineBintr Linked" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );
            THEN( "The Pipeline components are Linked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Pipeline with an FileSourceBintr is able to Link/UnlinkAll", "[PipelineBintr]" )
{
    GIVEN( "A new FileSourceBintr, PrimaryGieBintr, TilerBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        uint repeat(true);

        DSL_FILE_SOURCE_PTR pSourceBintr = 
            DSL_FILE_SOURCE_NEW(sourceName.c_str(), filePath.c_str(), repeat);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), displayId, depth, offsetX, offsetY, windowW, windowH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());

        REQUIRE( pPipelineBintr->IsLinked() == false );
            
        WHEN( "All components are added and PipelineBintr Linked" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pTilerBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            REQUIRE( pPipelineBintr->LinkAll() == true );
            REQUIRE( pPipelineBintr->IsLinked() == true );
            THEN( "The Pipeline components are Linked correctly" )
            {
                pPipelineBintr->UnlinkAll();
                REQUIRE( pPipelineBintr->IsLinked() == false );
            }
        }
    }
}
