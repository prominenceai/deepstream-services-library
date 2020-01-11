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
#include "DslTilerBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslPipelineBintr.h"

#define TIME_TO_SLEEP_FOR std::chrono::milliseconds(1000)

using namespace DSL;

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
                REQUIRE( pPipelineBintr->GetName() == pipelineName );
            }
        }
    }
}

SCENARIO( "A New PipelineBintr will fail to LinkAll with insufficient Components", "[PipelineBintr]" )
{
    GIVEN( "A new CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string pipelineName = "pipeline";

        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

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

SCENARIO( "A PipelineBintr's' XWindow is created correctly", "[PipelineBintr]" )
{
    GIVEN( "A PipelineBintr with a Tiler" ) 
    {
        std::string tilerName = "tiler";
        std::string pipelineName = "pipeline";

        uint tilerW(1280);
        uint tilerH(720);

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        pTilerBintr->AddToParent(pPipelineBintr);

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
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string tilerName = "tiler";
        std::string pipelineName = "pipeline";

        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

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
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string tilerName = "tiler";
        std::string pipelineName = "pipeline";

        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

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
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string tilerName = "tiler";
        std::string pipelineName = "pipeline";
        std::string primaryGieName = "primary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string modelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

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
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string tilerName = "tiler";
        std::string pipelineName = "pipeline";
        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";
        
        uint interval(1);
        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

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
        std::string pipelineName = "pipeline";
        std::string sourceName = "csi-source";
        std::string primaryGieName = "primary-gie";
        std::string tilerName = "tiler";
        std::string sinkName = "overlay-sink";
        std::string osdName = "on-screen-tiler";
        std::string inferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string modelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        DSL_OSD_PTR pOsdBintr = 
            DSL_OSD_NEW(osdName.c_str(), true);

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
        std::string pipelineName = "pipeline";
        std::string sourceName = "csi-source";
        std::string trackerName = "ktl-tracker";
        std::string primaryGieName = "primary-gie";
        std::string tilerName = "tiler";
        std::string sinkName = "overlay-sink";
        std::string osdName = "on-screen-tiler";
        std::string inferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string modelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint trackerW(300);
        uint trackerH(150);
        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), trackerW, trackerH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OSD_PTR pOsdBintr = 
            DSL_OSD_NEW(osdName.c_str(), true);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

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
        std::string pipelineName = "pipeline";
        std::string sourceName = "csi-source";
        std::string tilerName = "tiler";
        std::string sinkName = "overlay-sink";
        std::string trackerName = "ktl-tracker";
        std::string primaryGieName = "primary-gie";
        std::string primaryInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string primaryModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        std::string secondaryGieName = "secondary-gie";
        std::string secondaryInferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string secondaryModelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";
        std::string osdName = "on-screen-tiler";
        
        uint interval(1);
        uint trackerW(300);
        uint trackerH(150);
        uint tilerW(1280);
        uint tilerH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), tilerW, tilerH, fps_n, fps_d);

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), trackerW, trackerH);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), primaryInferConfigFile.c_str(), 
            primaryModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), secondaryInferConfigFile.c_str(), 
            secondaryModelEngineFile.c_str(), primaryGieName.c_str());

        DSL_OSD_PTR pOsdBintr = 
            DSL_OSD_NEW(osdName.c_str(), true);

        DSL_TILER_PTR pTilerBintr = 
            DSL_TILER_NEW(tilerName.c_str(), tilerW, tilerH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

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

SCENARIO( "A Pipeline can have at most one DewarperBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new DewarperBintrs and PipelineBintr" ) 
    {
        std::string pipelineName = "pipeline";
        std::string dewarperName1 = "dewarper-1";
        std::string dewarperName2 = "dewarper-2";
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_DEWARPER_PTR pDewarperBintr1 = 
            DSL_DEWARPER_NEW(dewarperName1.c_str(), defConfigFile.c_str());

        DSL_DEWARPER_PTR pDewarperBintr2 = 
            DSL_DEWARPER_NEW(dewarperName1.c_str(), defConfigFile.c_str());

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "A DewarperBintr is added to the PipelineBintr" )
        {
            REQUIRE( pDewarperBintr1->AddToParent(pPipelineBintr) == true );

            THEN( "A second DewarperBintr can not be added" )
            {
                REQUIRE( pDewarperBintr2->AddToParent(pPipelineBintr) == false );
            }
        }
    }
}

SCENARIO( "A Pipeline can have at most one TilerBintr", "[PipelineBintr]" )
{
    GIVEN( "Two new TilerBintrs and PipelineBintr" ) 
    {
        std::string pipelineName = "pipeline";
        std::string tilerName1 = "tiler-1";
        std::string tilerName2 = "tiler-2";

        uint tilerW(1280);
        uint tilerH(720);

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
        std::string pipelineName = "pipeline";
        std::string trackerName1 = "tracker-1";
        std::string trackerName2 = "tracker-2";

        uint trackerW(300);
        uint trackerH(150);

        DSL_KTL_TRACKER_PTR pTrackerBintr1 = 
            DSL_KTL_TRACKER_NEW(trackerName1.c_str(), trackerW, trackerH);

        DSL_KTL_TRACKER_PTR pTrackerBintr2 = 
            DSL_KTL_TRACKER_NEW(trackerName2.c_str(), trackerW, trackerH);

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
        std::string pipelineName = "pipeline";
        std::string primaryGieName1 = "primary-gie-1";
        std::string primaryGieName2 = "primary-gie-2";

        std::string primaryInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string primaryModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        uint interval(1);

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
        std::string pipelineName = "pipeline";
        std::string osdName1 = "on-screen-tiler-1";
        std::string osdName2 = "on-screen-tiler-2";

        DSL_OSD_PTR pOsdBintr1 = 
            DSL_OSD_NEW(osdName1.c_str(), true);

        DSL_OSD_PTR pOsdBintr2 = 
            DSL_OSD_NEW(osdName2.c_str(), true);

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

