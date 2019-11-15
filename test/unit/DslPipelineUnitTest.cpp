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
#include "DslDisplayBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"
#include "DslPipelineBintr.h"

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

        uint displayW(1280);
        uint displayH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), displayW, displayH, fps_n, fps_d);

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
    GIVEN( "A PipelineBintr with a Display" ) 
    {
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";

        uint displayW(1280);
        uint displayH(720);

        DSL_PIPELINE_PTR pPipelineBintr = 
            DSL_PIPELINE_NEW(pipelineName.c_str());

        DSL_DISPLAY_PTR pDisplayBintr = 
            DSL_DISPLAY_NEW(displayName.c_str(), displayW, displayH);

        pDisplayBintr->AddToParent(pPipelineBintr);

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
    GIVEN( "A new DisplayBintr, CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";

        uint displayW(1280);
        uint displayH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), displayW, displayH, fps_n, fps_d);

        DSL_DISPLAY_PTR pDisplayBintr = 
            DSL_DISPLAY_NEW(displayName.c_str(), displayW, displayH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pDisplayBintr->AddToParent(pPipelineBintr) == true );
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
    GIVEN( "A new DisplayBintr, CsiSourceBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";

        uint displayW(1280);
        uint displayH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), displayW, displayH, fps_n, fps_d);

        DSL_DISPLAY_PTR pDisplayBintr = 
            DSL_DISPLAY_NEW(displayName.c_str(), displayW, displayH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added and the PipelineBintr is Linked" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pDisplayBintr->AddToParent(pPipelineBintr) == true );
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
    GIVEN( "A new DisplayBintr, CsiSourceBintr, PrimaryGieBintr, OverlaySinkBintr, and a PipelineBintr" ) 
    {
        std::string sourceName = "csi-source";
        std::string sinkName = "overlay-sink";
        std::string displayName = "tiled-display";
        std::string pipelineName = "pipeline";
        std::string primaryGieName = "primary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string modelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel";
        
        uint interval(1);
        uint uniqueId(1);
        uint displayW(1280);
        uint displayH(720);
        uint fps_n(1);
        uint fps_d(30);
        uint offsetX(0);
        uint offsetY(0);
        uint sinkW(0);
        uint sinkH(0);

        DSL_CSI_SOURCE_PTR pSourceBintr = 
            DSL_CSI_SOURCE_NEW(sourceName.c_str(), displayW, displayH, fps_n, fps_d);

        DSL_DISPLAY_PTR pDisplayBintr = 
            DSL_DISPLAY_NEW(displayName.c_str(), displayW, displayH);

        DSL_OVERLAY_SINK_PTR pSinkBintr = 
            DSL_OVERLAY_SINK_NEW(sinkName.c_str(), offsetX, offsetY, sinkW, sinkH);

            DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
                DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
                modelEngineFile.c_str(), interval, uniqueId);

        DSL_PIPELINE_PTR pPipelineBintr = DSL_PIPELINE_NEW(pipelineName.c_str());
            
        WHEN( "All components are added to the PipelineBintr" )
        {
            REQUIRE( pSourceBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pPrimaryGieBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pDisplayBintr->AddToParent(pPipelineBintr) == true );
            REQUIRE( pSinkBintr->AddToParent(pPipelineBintr) == true );

            THEN( "The Pipeline components are Linked correctly" )
            {
                REQUIRE( pPipelineBintr->LinkAll() == true );
            }
        }
    }
}

