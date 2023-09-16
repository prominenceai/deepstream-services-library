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
#include "DslMultiSourcesBintr.h"

using namespace DSL;

static const std::string pipelineSourcesName("pipeline-sources");
static const std::string pipelineSourcesName0("pipeline-sources-0");
static const std::string pipelineSourcesName1("pipeline-sources-1");
static const std::string pipelineSourcesName2("pipeline-sources-2");

static const std::string sourceName("source");
static const std::string sourceName0("source-0");
static const std::string sourceName1("source-1");
static const std::string sourceName2("source-2");
static const std::string sourceName3("source-3");
static const std::string sourceName4("source-4");
static const std::string sourceName5("source-5");

static const uint pipelineId(0);

static const std::string filePath(
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");


SCENARIO( "A PipelineSourcesBintr is created correctly", "[PipelineSourcesBintr]" )
{
    GIVEN( "A name for a PipelineSourcesBintr" ) 
    {
        WHEN( "The PipelineSourcesBintr is created" )
        {
            DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
                DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSourcesBintr->GetName() == pipelineSourcesName );
                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );
                REQUIRE( pPipelineSourcesBintr->m_pStreammux != nullptr );
            }
        }
    }
}

SCENARIO( "Adding a single Source to a PipelineSourcesBintr is managed correctly",  "[PipelineSourcesBintr]" )
{
    GIVEN( "A new Pipeline Sources Bintr and new Source in memory" ) 
    {
        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false, false, 0);
            
        WHEN( "The Source is added to the Pipeline Sources Bintr" )
        {
            pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
            
            THEN( "The Pipeline Sources Bintr is updated correctly" )
            {
                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 1 );
                REQUIRE( pPipelineSourcesBintr->IsChild(pSourceBintr) == true );
                REQUIRE( pSourceBintr->IsInUse() == true );
            }
        }
    }
}

SCENARIO( "Removing a single Source from a PipelineSourcesBintr is managed correctly", 
    "[PipelineSourcesBintr]" )
{
    GIVEN( "A Pipeline Sources Bintr with a Source in memory" ) 
    {
        std::string sourceName = "csi-source";
        std::string pipelineSourcesName = "pipeline-sources";

        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false, false, 0);
            
        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == true );
        REQUIRE( pSourceBintr->IsInUse() == true );
            
        // second call must fail
        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == false );
            
        WHEN( "The Source is removed from the Pipeline Sources Bintr" )
        {
            REQUIRE(pPipelineSourcesBintr->RemoveChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == true );

            // second call must fail
            REQUIRE(pPipelineSourcesBintr->RemoveChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == false );
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr->IsInUse() == false );
                REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );
            }
        }
    }
}

SCENARIO( "Linking a single Source to a Pipeline Streammux is managed correctly",  
    "[PipelineSourcesBintr]" )
{
    GIVEN( "A new PipelineSourcesBintr with single SourceBintr" ) 
    {
        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 0 );

        DSL_URI_SOURCE_PTR pSourceBintr = DSL_URI_SOURCE_NEW(
            sourceName.c_str(), filePath.c_str(), false, false, 0);

        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)) == true );
            
        WHEN( "The Single Source is Linked to the Streammux" )
        {
            REQUIRE( pPipelineSourcesBintr->LinkAll() == true );
             
            THEN( "The Source and SourcesBintr are updated correctly" )
            {
                REQUIRE( pSourceBintr->IsInUse() == true );
                REQUIRE( pSourceBintr->IsLinkedToSink() == true );
            }
        }
    }
}


SCENARIO( "Linking multiple Sources to a Streammux is managed correctly",
    "[PipelineSourcesBintr]" )
{
    GIVEN( "A Pipeline Sources Bintr with multiple Source in memory" ) 
    {
        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        DSL_URI_SOURCE_PTR pSourceBintr0 = DSL_URI_SOURCE_NEW(
            sourceName0.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr0->GetRequestPadId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr1  = DSL_URI_SOURCE_NEW(
            sourceName1.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr1->GetRequestPadId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr2 = DSL_URI_SOURCE_NEW(
            sourceName2.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr2->GetRequestPadId() == -1 );

        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );
        
        REQUIRE( pPipelineSourcesBintr->GetNumChildren() == 3 );
                    
        WHEN( "All Sources are linked to the Streammux" )
        {
            REQUIRE( pPipelineSourcesBintr->LinkAll() == true );
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsInUse() == true );
                REQUIRE( pSourceBintr0->GetRequestPadId() == 0 );
                REQUIRE( pSourceBintr0->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr1->IsInUse() == true );
                REQUIRE( pSourceBintr1->GetRequestPadId() == 1 );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr2->IsInUse() == true );
                REQUIRE( pSourceBintr2->GetRequestPadId() == 2 );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == true );
            }
        }
    }
}

SCENARIO( "Unlinking multiple Sources from a Streammux is managed correctly", 
    "[PipelineSourcesBintr]" )
{
    GIVEN( "A Pipeline Sources Bintr with multiple Sources a linked to the Streammux" ) 
    {

        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        DSL_URI_SOURCE_PTR pSourceBintr0 = DSL_URI_SOURCE_NEW(
            sourceName0.c_str(), filePath.c_str(), false, false, 0);

        DSL_URI_SOURCE_PTR pSourceBintr1  = DSL_URI_SOURCE_NEW(
            sourceName1.c_str(), filePath.c_str(), false, false, 0);

        DSL_URI_SOURCE_PTR pSourceBintr2 = DSL_URI_SOURCE_NEW(
            sourceName2.c_str(), filePath.c_str(), false, false, 0);

        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
        REQUIRE( pPipelineSourcesBintr->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );
                    
        REQUIRE( pPipelineSourcesBintr->LinkAll() == true );

        WHEN( "All Sources are unlinked from the Streammux" )
        {
            pPipelineSourcesBintr->UnlinkAll();
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsLinkedToSink() == false );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == false );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == false );
            }
        }
    }
}

SCENARIO( "Linking and unlinking multiple Sources to multiple Streammuxers is managed correctly", 
    "[PipelineSourcesBintr]" )
{
    GIVEN( "Multiple Pipeline Sources Bintrs with multiple Sources in memory" ) 
    {
        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr0 = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName0.c_str(), 0);

        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr1 = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName1.c_str(), 1);

        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr2 = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName2.c_str(), 2);

        DSL_URI_SOURCE_PTR pSourceBintr0 = DSL_URI_SOURCE_NEW(
            sourceName0.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr0->GetRequestPadId() == -1 );
        REQUIRE( pSourceBintr0->GetUniqueId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr1  = DSL_URI_SOURCE_NEW(
            sourceName1.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr1->GetRequestPadId() == -1 );
        REQUIRE( pSourceBintr1->GetUniqueId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr2 = DSL_URI_SOURCE_NEW(
            sourceName2.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr2->GetRequestPadId() == -1 );
        REQUIRE( pSourceBintr2->GetUniqueId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr3 = DSL_URI_SOURCE_NEW(
            sourceName3.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr3->GetRequestPadId() == -1 );
        REQUIRE( pSourceBintr3->GetUniqueId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr4  = DSL_URI_SOURCE_NEW(
            sourceName4.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr4->GetRequestPadId() == -1 );
        REQUIRE( pSourceBintr4->GetUniqueId() == -1 );

        DSL_URI_SOURCE_PTR pSourceBintr5 = DSL_URI_SOURCE_NEW(
            sourceName5.c_str(), filePath.c_str(), false, false, 0);
        REQUIRE( pSourceBintr5->GetRequestPadId() == -1 );
        REQUIRE( pSourceBintr5->GetUniqueId() == -1 );

        REQUIRE( pPipelineSourcesBintr0->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
        REQUIRE( pPipelineSourcesBintr0->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
        REQUIRE( pPipelineSourcesBintr0->GetNumChildren() == 2 );
            
        REQUIRE( pPipelineSourcesBintr1->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );
        REQUIRE( pPipelineSourcesBintr1->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr3)) == true );
        REQUIRE( pPipelineSourcesBintr1->GetNumChildren() == 2 );
        
        REQUIRE( pPipelineSourcesBintr2->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr4)) == true );
        REQUIRE( pPipelineSourcesBintr2->AddChild(
            std::dynamic_pointer_cast<SourceBintr>(pSourceBintr5)) == true );
        REQUIRE( pPipelineSourcesBintr2->GetNumChildren() == 2 );
        
        WHEN( "All Sources are linked to the Streammux" )
        {
            REQUIRE( pPipelineSourcesBintr0->LinkAll() == true );
            REQUIRE( pPipelineSourcesBintr1->LinkAll() == true );
            REQUIRE( pPipelineSourcesBintr2->LinkAll() == true );
            
            THEN( "The Pipeline Sources Bintr and Source are updated correctly" )
            {
                REQUIRE( pSourceBintr0->IsInUse() == true );
                REQUIRE( pSourceBintr0->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr0->GetRequestPadId() == 0 );
                REQUIRE( pSourceBintr0->GetUniqueId() == 0x00000000 );

                REQUIRE( pSourceBintr1->IsInUse() == true );
                REQUIRE( pSourceBintr1->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr1->GetRequestPadId() == 1 );
                REQUIRE( pSourceBintr1->GetUniqueId() == 0x00000001 );

                REQUIRE( pSourceBintr2->IsInUse() == true );
                REQUIRE( pSourceBintr2->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr2->GetRequestPadId() == 0 );
                REQUIRE( pSourceBintr2->GetUniqueId() == 0x00010000 );

                REQUIRE( pSourceBintr3->IsInUse() == true );
                REQUIRE( pSourceBintr3->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr3->GetRequestPadId() == 1 );
                REQUIRE( pSourceBintr3->GetUniqueId() == 0x00010001 );

                REQUIRE( pSourceBintr4->IsInUse() == true );
                REQUIRE( pSourceBintr4->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr4->GetRequestPadId() == 0 );
                REQUIRE( pSourceBintr4->GetUniqueId() == 0x00020000 );

                REQUIRE( pSourceBintr5->IsInUse() == true );
                REQUIRE( pSourceBintr5->IsLinkedToSink() == true );
                REQUIRE( pSourceBintr5->GetRequestPadId() == 1 );
                REQUIRE( pSourceBintr5->GetUniqueId() == 0x00020001 );

            }
        }
    }
}

SCENARIO( "All GST Resources are released on PipelineSourcesBintr destruction", 
    "[PipelineSourcesBintr]" )
{
    GIVEN( "Attributes for a new PipelineSourcesBintr and several new SourcesBintrs" ) 
    {
        WHEN( "The Bintrs are created and the Sources are added as children and linked" )
        {
            DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
                DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

            DSL_URI_SOURCE_PTR pSourceBintr0 = DSL_URI_SOURCE_NEW(
                sourceName0.c_str(), filePath.c_str(), false, false, 0);

            DSL_URI_SOURCE_PTR pSourceBintr1  = DSL_URI_SOURCE_NEW(
                sourceName1.c_str(), filePath.c_str(), false, false, 0);

            DSL_URI_SOURCE_PTR pSourceBintr2 = DSL_URI_SOURCE_NEW(
                sourceName2.c_str(), filePath.c_str(), false, false, 0);

            REQUIRE( pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );

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
            DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
                DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

            DSL_URI_SOURCE_PTR pSourceBintr0 = DSL_URI_SOURCE_NEW(
                sourceName0.c_str(), filePath.c_str(), false, false, 0);

            DSL_URI_SOURCE_PTR pSourceBintr1  = DSL_URI_SOURCE_NEW(
                sourceName1.c_str(), filePath.c_str(), false, false, 0);

            DSL_URI_SOURCE_PTR pSourceBintr2 = DSL_URI_SOURCE_NEW(
                sourceName2.c_str(), filePath.c_str(), false, false, 0);

            REQUIRE( pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr0)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr1)) == true );
            REQUIRE( pPipelineSourcesBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceBintr2)) == true );

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

SCENARIO( "The Pipeline Streammuxer's num-surfaces-per-frame can be read and updaed",
    "[PipelineSourcesBintr]" )
{
    GIVEN( "A new PipelineSourcesBintr with single SourceBintr" ) 
    {
        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        REQUIRE( pPipelineSourcesBintr->GetStreammuxNumSurfacesPerFrame() == 1 );
            
        WHEN( "The Stream Muxer's num-surfaces-per-frame is set to a new value " )
        {
            pPipelineSourcesBintr->SetStreammuxNumSurfacesPerFrame(2);
             
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( pPipelineSourcesBintr->
                    GetStreammuxNumSurfacesPerFrame() == 2 );
            }
        }
    }
}

SCENARIO( "The Pipeline Streammuxer's nvbuf-memory-type can be read and updated",
    "[PipelineSourcesBintr]" )
{
    GIVEN( "A new PipelineSourcesBintr" ) 
    {
        DSL_MULTI_SOURCES_PTR pPipelineSourcesBintr = 
            DSL_MULTI_SOURCES_NEW(pipelineSourcesName.c_str(), pipelineId);

        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            REQUIRE( pPipelineSourcesBintr->GetStreammuxNvbufMemType() 
                == DSL_NVBUF_MEM_TYPE_DEFAULT );
        }
        else
        {
            REQUIRE( pPipelineSourcesBintr->GetStreammuxNvbufMemType() 
                == DSL_NVBUF_MEM_TYPE_CUDA_DEVICE );
        }
            
        WHEN( "The Stream Muxer's num-surfaces-per-frame is set to a new value " )
        {
            uint newNvbufMemType;
            if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
            {
                newNvbufMemType = DSL_NVBUF_MEM_TYPE_SURFACE_ARRAY;
            }
            else
            {
                newNvbufMemType = DSL_NVBUF_MEM_TYPE_CUDA_UNIFIED;
            }
        
            pPipelineSourcesBintr->SetStreammuxNvbufMemType(newNvbufMemType);
             
            THEN( "The correct value is returned on get" )
            {
                REQUIRE( pPipelineSourcesBintr->GetStreammuxNvbufMemType() == newNvbufMemType );
            }
        }
    }
}

