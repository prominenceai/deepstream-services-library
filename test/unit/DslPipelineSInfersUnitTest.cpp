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
#include "DslPipelineSInfersBintr.h"
#include "DslInferBintr.h"

using namespace DSL;

static const std::string pipelineSGiesName = "pipeline-sgies";

static const std::string primaryGieName("primary-gie");
static const std::string secondaryGieName("secondary-gie");
static const std::string pgieInferConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static const std::string pgieModelEngineFile(
    "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static const std::string secondaryGieName1("secondary-gie-1");
static const std::string secondaryGieName2("secondary-gie-2");
static const std::string secondaryGieName3("secondary-gie-3");
static const std::string sgieInferConfigFile1(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carcolor.txt");
static const std::string sgieModelEngineFile1(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine");
static const std::string sgieInferConfigFile2(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carmake.txt");
static const std::string sgieModelEngineFile2(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarMake/resnet18.caffemodel_b8_gpu0_fp16.engine");
static const std::string sgieInferConfigFile3(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.txt");
static const std::string sgieModelEngineFile3(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_VehicleTypes/resnet18.caffemodel_b8_gpu0_fp16.engine");

static const uint primaryUniqueId = std::hash<std::string>{}(primaryGieName.c_str());
static const uint secondaryUniqueId = std::hash<std::string>{}(secondaryGieName.c_str());

static const uint interval(1);

SCENARIO( "A PipelineSInfersBintr is created correctly", "[PipelineSInfersBintr]" )
{
    GIVEN( "A name for a PipelineSInfersBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        WHEN( "The PipelineSinksBintr is created" )
        {
            DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
                DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSInfersBintr->GetName() == pipelineSGiesName );
                REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 0 );
            }
        }
    }
}

SCENARIO( "A SecondaryInferBintr can be added to a PipelineSInfersBintr", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr and SecondearyGieBintr" ) 
    {

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), 0, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        WHEN( "The SecondaryGie is added to the PipelineSInfersBintr" )
        {
            REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 1 );
                REQUIRE( pPipelineSInfersBintr->IsChild(pSecondaryInferBintr) == true );
                REQUIRE( pSecondaryInferBintr->IsInUse() == true );
            }
        }
    }
}

SCENARIO( "A SecondaryInferBintr can be removed from a PipelineSInfersBintr", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr with a child SecondearyGieBintr" ) 
    {

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );
        REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 1 );

        WHEN( "The SecondearyGieBintr is removed from the PipelineSInfersBintr" )
        {
            REQUIRE( pPipelineSInfersBintr->RemoveChild(pSecondaryInferBintr) == true );
            
            THEN( "All members have been updated correctly" )
            {
                REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 0 );
                REQUIRE( pPipelineSInfersBintr->IsChild(pSecondaryInferBintr) == false );
                REQUIRE( pSecondaryInferBintr->IsInUse() == false );
            }
        }
    }
}

SCENARIO( "A SecondaryInferBintr can only be added to a PipelineSInfersBintr once", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr with a child SecondearyGieBintr" ) 
    {
        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        WHEN( "The SecondaryInferBintr is added to the pPipelineSInfersBintr" )
        {
            REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );
            REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 1 );
            THEN( "It can not be added a second time" )
            {
                REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == false );
                REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 1 );
            }
        }
    }
}


SCENARIO( "A PipelineSInfersBintr can not LinkAll without setting the PrimaryGieId first", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr with a child SecondearyGieBintr" ) 
    {
        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );

        WHEN( "The PipelineSGiessBintr is asked to LinkAll prior to setting the PrimaryGieName" )
        {
            REQUIRE( pPipelineSInfersBintr->LinkAll() == false );

            THEN( "The PipelineSGiessBintr remains in an unlinked state" )
            {
                REQUIRE( pPipelineSInfersBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSInfersBintr can not LinkAll without setting the Batch Size first", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new  PrimaryGieBintr and PipelineSInfersBintr with a child SecondearyGieBintr" ) 
    {
        uint interval(1);

        DSL_PRIMARY_INFER_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_INFER_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );
        
        pPipelineSInfersBintr->SetInferOnId(pPrimaryGieBintr->GetUniqueId());

        WHEN( "The PipelineSInfersBintr is asked to LinkAll prior to setting the Batch Size" )
        {
            REQUIRE( pPipelineSInfersBintr->LinkAll() == false );

            THEN( "The PipelineSInfersBintr remains in an unlinked state" )
            {
                REQUIRE( pPipelineSInfersBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSInfersBintr with its PrimaryGieId and Batch Size set can LinkAll", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr with a child SecondearyGieBintr" ) 
    {
        uint interval(1);

        DSL_PRIMARY_INFER_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_INFER_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );

        WHEN( "The the PrimaryGieName and Batch Size is set for the PipelineSInfersBintr" )
        {
            pPipelineSInfersBintr->SetInferOnId(pPrimaryGieBintr->GetUniqueId());
            pPipelineSInfersBintr->SetBatchSize(1);

            THEN( "The PipelineSInfersBintr is able to LinkAll" )
            {
                REQUIRE( pPipelineSInfersBintr->LinkAll() == true );
                REQUIRE( pPipelineSInfersBintr->IsLinked() == true );
                REQUIRE( pSecondaryInferBintr->GetBatchSize() == 1 );
            }
        }
    }
}
SCENARIO( "A PipelineSInfersBintr Linked with a SecondaryInferBintr can UnlinkAll", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr Linked with a child SecondearyGieBintr" ) 
    {
        uint interval(1);

        DSL_PRIMARY_INFER_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_INFER_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());
            
        pPipelineSInfersBintr->SetInferOnId(pPrimaryGieBintr->GetUniqueId());
        pPipelineSInfersBintr->SetBatchSize(1);
        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr) == true );

        WHEN( "The pPipelineSInfersBintr is in a IsLinked state" )
        {
            REQUIRE( pPipelineSInfersBintr->LinkAll() == true );
            REQUIRE( pPipelineSInfersBintr->IsLinked() == true );
            REQUIRE( pSecondaryInferBintr->GetBatchSize() == 1 );

            THEN( "It can UnlinkAll with a single SecondaryInferBintr" )
            {
                pPipelineSInfersBintr->UnlinkAll();
                REQUIRE( pPipelineSInfersBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSInfersBintr with several SecondaryInferBintrs can LinkAll", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr with three child SecondearyGieBintrs" ) 
    {
        uint interval(0);

        DSL_PRIMARY_INFER_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_INFER_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr1 = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName1.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr2 = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName2.c_str(), sgieInferConfigFile2.c_str(), 
            sgieModelEngineFile2.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr3 = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName3.c_str(), sgieInferConfigFile3.c_str(), 
            sgieModelEngineFile3.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        pPipelineSInfersBintr->SetInferOnId(pPrimaryGieBintr->GetUniqueId());
        pPipelineSInfersBintr->SetBatchSize(3);

        WHEN( "All three SecondaryInferBintrs are added to the pPipelineSInfersBintr" )
        {
            REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr1) == true );
            REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr2) == true );
            REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr3) == true );
            REQUIRE( pPipelineSInfersBintr->IsLinked() == false );

            THEN( "The pPipelineSInfersBintr can update and LinkAll SecondaryInferBintrs" )
            {
                REQUIRE( pPipelineSInfersBintr->LinkAll() == true );
                REQUIRE( pPipelineSInfersBintr->IsLinked() == true );
                REQUIRE( pSecondaryInferBintr1->GetBatchSize() == 3 );
                REQUIRE( pSecondaryInferBintr2->GetBatchSize() == 3 );
                REQUIRE( pSecondaryInferBintr3->GetBatchSize() == 3 );
            }
        }
    }
}

SCENARIO( "A PipelineSInfersBintr with several SecondaryInferBintrs can UnlinkAll", "[PipelineSInfersBintr]" )
{
    GIVEN( "A new PipelineSInfersBintr with a child SecondearyGieBintr" ) 
    {
        uint interval(0);

        DSL_PRIMARY_INFER_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_INFER_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr1 = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName1.c_str(), sgieInferConfigFile1.c_str(), 
            sgieModelEngineFile1.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr2 = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName2.c_str(), sgieInferConfigFile2.c_str(), 
            sgieModelEngineFile2.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_SECONDARY_INFER_PTR pSecondaryInferBintr3 = 
            DSL_SECONDARY_INFER_NEW(secondaryGieName3.c_str(), sgieInferConfigFile3.c_str(), 
            sgieModelEngineFile2.c_str(), primaryGieName.c_str(), interval, DSL_INFER_TYPE_GIE);

        DSL_PIPELINE_SINFERS_PTR pPipelineSInfersBintr = 
            DSL_PIPELINE_SINFERS_NEW(pipelineSGiesName.c_str());

        pPipelineSInfersBintr->SetInferOnId(pPrimaryGieBintr->GetUniqueId());
        pPipelineSInfersBintr->SetBatchSize(3);

        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr1) == true );
        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr2) == true );
        REQUIRE( pPipelineSInfersBintr->AddChild(pSecondaryInferBintr3) == true );
        REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 3 );

        WHEN( "The pPipelineSInfersBintr is Linked" )
        {
            REQUIRE( pPipelineSInfersBintr->LinkAll() == true );
            REQUIRE( pPipelineSInfersBintr->IsLinked() == true );
                
            THEN( "The pPipelineSInfersBintr can be Unlinked and all SecondaryInferBintrs removed" )
            {
                pPipelineSInfersBintr->UnlinkAll();
                REQUIRE( pPipelineSInfersBintr->IsLinked() == false );
                REQUIRE( pPipelineSInfersBintr->RemoveChild(pSecondaryInferBintr1) == true );
                REQUIRE( pPipelineSInfersBintr->RemoveChild(pSecondaryInferBintr2) == true );
                REQUIRE( pPipelineSInfersBintr->RemoveChild(pSecondaryInferBintr3) == true );
                REQUIRE( pPipelineSInfersBintr->GetNumChildren() == 0 );
            }
        }
    }
}
