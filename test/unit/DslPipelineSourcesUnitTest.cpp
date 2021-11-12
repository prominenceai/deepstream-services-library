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
#include "DslSourceBintr.h"
#include "DslPipelineSourcesBintr.h"

using namespace DSL;

SCENARIO( "A PipelineSourcesBintr is created correctly", "[PipelineSourcesBintr]" )
{
    GIVEN( "A name for a PipelineSourcesBintr" ) 
    {
        std::string pipelineSourcesName = "pipeline-sources";

        WHEN( "The PipelineSourcesBintr is created" )
        {
            DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
                DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSourcesBintr->GetName() == pipelineSourcesName );
                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );
                REQUIRE( pPipelineSourcesBintr->m_pStreamMux != nullptr );
            }
        }
    }
}

SCENARIO( "Adding a single Source to a PipelineSourcesBintr is managed correctly",  "[PipelineSourcesBintr]" )
{
    GIVEN( "A new Pipeline Sources Bintr and new Source in memory" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(1);
        uint fps_d(30);
        std::string sourceName = "test-csi-source";
        std::string pipelineSourcesName = "pipeline-sources";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);
            
        REQUIRE( pSourceBintr->GetId() == -1 );
            
        WHEN( "The Source is added to the Pipeline Sources Bintr" )
        {
            pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
            
            THEN( "The Pipeline Sources Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 1 );
                REQUIRE( pPipelineSourcesBintr->IsChild(pSourceBintr) == true );
                REQUIRE( pSourceBintr->IsInUse() == true );
                REQUIRE( pSourceBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Removing a single Source from a PipelineSourcesBintr is managed correctly", "[PipelineSourcesBintr]" )
{
    GIVEN( "A Pipeline Sources Bintr with a Source in memory" ) 
    {
        std::string sourceName = "csi-source";
        std::string pipelineSourcesName = "pipeline-sources";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName.c_str(), 1280, 720, 30, 1));
            
        REQUIRE( pSourceBintr->GetId() == -1 );

        pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
        REQUIRE( pSourceBintr->IsInUse() == true );
            
        WHEN( "The Source is removed from the Pipeline Sources Bintr" )
        {
            pPipelineSourcesBintr->RemoveChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr->IsInUse() == false );
                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );
                REQUIRE( pSourceBintr->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "Linking a single Source to a Pipeline StreamMux is managed correctly",  "[PipelineSourcesBintr]" )
{
    GIVEN( "A new PipelineSourcesBintr with single SourceBintr" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(1);
        uint fps_d(30);
        std::string sourceName = "test-csi-source";
        std::string pipelineSourcesName = "pipeline-sources";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);
        REQUIRE( pSourceBintr->GetId() == -1 );

        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == true );
            
        WHEN( "The Single Source is Linked to the StreamMux" )
        {
            REQUIRE( pPipelineSourcesBintr->LinkAll() == true );
             
            THEN( "The Source and SourcesBintr are updated correctly" )
            {
                REQUIRE( pSourceBintr->IsInUse() == true );
                REQUIRE( pSourceBintr->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr->GetId() == 0 );
            }
        }
    }
}


SCENARIO( "Linking multiple Sources to a StreamMux is managed correctly", "[PipelineSourcesBintr]" )
{
    GIVEN( "A Pipeline Sources Bintr with multiple Source in memory" ) 
    {
        std::string pipelineSourcesName = "pipeline-sources";
        std::string sourceName0 = "csi-source-0";
        std::string sourceName1 = "csi-source-1";
        std::string sourceName2 = "csi-source-2";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr0 = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName0.c_str(), 1280, 720, 30, 1));
        REQUIRE( pSourceBintr0->GetId() == -1 );

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr1 = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName1.c_str(), 1280, 720, 30, 1));
        REQUIRE( pSourceBintr1->GetId() == -1 );

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr2 = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName2.c_str(), 1280, 720, 30, 1));
        REQUIRE( pSourceBintr2->GetId() == -1 );

        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );
        
        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 3 );
                    
        WHEN( "All Sources are linked to the StreamMux" )
        {
            REQUIRE( pPipelineSourcesBintr->LinkAll() == true );
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsInUse() == true );
                REQUIRE( pSourceBintr0->GetId() == 0 );
                REQUIRE( pSourceBintr0->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr1->IsInUse() == true );
                REQUIRE( pSourceBintr1->GetId() == 1 );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr2->IsInUse() == true );
                REQUIRE( pSourceBintr2->GetId() == 2 );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == true );
            }
        }
    }
}

SCENARIO( "Unlinking multiple Sources from a StreamMux is managed correctly", "[PipelineSourcesBintr]" )
{
    GIVEN( "A Pipeline Sources Bintr with multiple Sources a linked to the StreamMux" ) 
    {
        std::string pipelineSourcesName = "pipeline-sources";
        std::string sourceName0 = "csi-source-0";
        std::string sourceName1 = "csi-source-1";
        std::string sourceName2 = "csi-source-2";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr0 = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName0.c_str(), 1280, 720, 30, 1));

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr1 = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName1.c_str(), 1280, 720, 30, 1));

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr2 = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName2.c_str(), 1280, 720, 30, 1));

        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );
                    
        REQUIRE( pPipelineSourcesBintr->LinkAll() == true );

        WHEN( "All Sources are unlinked from the StreamMux" )
        {
            pPipelineSourcesBintr->UnlinkAll();
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsLinkedToSink() == false );
                REQUIRE( pSourceBintr0->GetId() == -1 );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == false );
                REQUIRE( pSourceBintr1->GetId() == -1 );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == false );
                REQUIRE( pSourceBintr2->GetId() == -1 );
            }
        }
    }
}

SCENARIO( "All GST Resources are released on PipelineSourcesBintr destruction", "[PipelineSourcesBintr]" )
{
    GIVEN( "Attributes for a new PipelineSourcesBintr and several new SourcesBintrs" ) 
    {
        std::string pipelineSourcesName = "pipeline-sources";
        std::string sourceName0 = "csi-source-0";
        std::string sourceName1 = "csi-source-1";
        std::string sourceName2 = "csi-source-2";

        WHEN( "The Bintrs are created and the Sources are added as children and linked" )
        {
            DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
                DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr0 = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName0.c_str(), 1280, 720, 30, 1));

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr1 = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName1.c_str(), 1280, 720, 30, 1));

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr2 = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName2.c_str(), 1280, 720, 30, 1));

            REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );

            REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 3 );
            REQUIRE( pPipelineSourcesBintr->LinkAll() == true );

            THEN( "The SinkBintrs are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsInUse() == true );
                REQUIRE( pSourceBintr0->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr1->IsInUse() == true );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr2->IsInUse() == true );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == true );
            }
        }
        WHEN( "After destruction, all SourceBintrs and Request Pads can be recreated and linked again" )
        {
            DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
                DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr0 = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName0.c_str(), 1280, 720, 30, 1));

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr1 = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName1.c_str(), 1280, 720, 30, 1));

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr2 = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName2.c_str(), 1280, 720, 30, 1));

            REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );

            REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 3 );
            REQUIRE( pPipelineSourcesBintr->LinkAll() == true );

            THEN( "The SinkBintrs are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsInUse() == true );
                REQUIRE( pSourceBintr0->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr1->IsInUse() == true );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr2->IsInUse() == true );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == true );
            }
        }
    }
}

SCENARIO( "The Pipeline Streammuxer's num-surfaces-per-frame can be read and updaed",  "[PipelineSourcesBintr]" )
{
    GIVEN( "A new PipelineSourcesBintr with single SourceBintr" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(1);
        uint fps_d(30);
        std::string sourceName = "test-csi-source";
        std::string pipelineSourcesName = "pipeline-sources";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        REQUIRE( pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == true );
        REQUIRE( pPipelineSourcesBintr->LinkAll() == true );

        uint num;
        
        pPipelineSourcesBintr->GetStreamMuxNumSurfacesPerFrame(&num);
        REQUIRE( num == 1 );
            
        WHEN( "The Stream Muxer's num-surfaces-per-frame is set to a new value " )
        {
            pPipelineSourcesBintr->SetStreamMuxNumSurfacesPerFrame(2);
             
            THEN( "The correct value is returned on get" )
            {
                uint num;
                
                pPipelineSourcesBintr->GetStreamMuxNumSurfacesPerFrame(&num);
                REQUIRE( num == 2 );
            }
        }
    }
}

SCENARIO( "The Pipeline Streammuxer's nvbuf-memory-type can be read and updated",  "[PipelineSourcesBintr]" )
{
    GIVEN( "A new PipelineSourcesBintr" ) 
    {
        std::string pipelineSourcesName = "pipeline-sources";

        DSL_PIPELINE_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(pipelineSourcesName.c_str());

        REQUIRE( pPipelineSourcesBintr->GetStreamMuxNvbufMemType() == DSL_NVBUF_MEM_DEFAULT );
            
        WHEN( "The Stream Muxer's num-surfaces-per-frame is set to a new value " )
        {
            uint newNvbufMemType = DSL_NVBUF_MEM_UNIFIED;
        
            pPipelineSourcesBintr->SetStreamMuxNvbufMemType(newNvbufMemType);
             
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( pPipelineSourcesBintr->GetStreamMuxNvbufMemType() == newNvbufMemType );
            }
        }
    }
}

