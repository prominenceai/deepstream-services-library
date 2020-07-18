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
#include "DslPipelineSGiesBintr.h"
#include "DslGieBintr.h"

using namespace DSL;

SCENARIO( "A PipelineSGiesBintr is created correctly", "[PipelineSGiesBintr]" )
{
    GIVEN( "A name for a PipelineSGiesBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        WHEN( "The PipelineSinksBintr is created" )
        {
            DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
                DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSGiesBintr->GetName() == pipelineSGiesName );
                REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 0 );
            }
        }
    }
}

SCENARIO( "A SecondaryGieBintr can be added to a PipelineSGiesBintr", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr and SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint primaryUniqueId = std::hash<std::string>{}(primaryGieName.c_str());
        uint secondaryUniqueId = std::hash<std::string>{}(secondaryGieName.c_str());

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), 0);

        REQUIRE( pSecondaryGieBintr->GetUniqueId() == secondaryUniqueId);
        REQUIRE( pSecondaryGieBintr->SetInferOnGieName(primaryGieName.c_str()) == true );
        REQUIRE( pSecondaryGieBintr->GetInferOnGieUniqueId() == primaryUniqueId);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        WHEN( "The SecondaryGie is added to the PipelineSGiesBintr" )
        {
            REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 1 );
                REQUIRE( pPipelineSGiesBintr->IsChild(pSecondaryGieBintr) == true );
                REQUIRE( pSecondaryGieBintr->IsInUse() == true );
            }
        }
    }
}

SCENARIO( "A SecondaryGieBintr can be removed from a PipelineSGiesBintr", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint secondaryUniqueId = std::hash<std::string>{}(secondaryGieName.c_str());

        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        REQUIRE( pSecondaryGieBintr->GetUniqueId() == secondaryUniqueId);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );
        REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 1 );

        WHEN( "The SecondearyGieBintr is removed from the PipelineSGiesBintr" )
        {
            REQUIRE( pPipelineSGiesBintr->RemoveChild(pSecondaryGieBintr) == true );
            
            THEN( "All members have been updated correctly" )
            {
                REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 0 );
                REQUIRE( pPipelineSGiesBintr->IsChild(pSecondaryGieBintr) == false );
                REQUIRE( pSecondaryGieBintr->IsInUse() == false );
            }
        }
    }
}

SCENARIO( "A SecondaryGieBintr can only be added to a PipelineSGiesBintr once", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";

        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        WHEN( "The SecondaryGieBintr is added to the pPipelineSGiesBintr" )
        {
            REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );
            REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 1 );
            THEN( "It can not be added a second time" )
            {
                REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == false );
                REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 1 );
            }
        }
    }
}


SCENARIO( "A PipelineSGiesBintr can not LinkAll without setting the PrimaryGieId first", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";

        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );

        WHEN( "The PipelineSGiessBintr is asked to LinkAll prior to setting the PrimaryGieName" )
        {
            REQUIRE( pPipelineSGiesBintr->LinkAll() == false );

            THEN( "The PipelineSGiessBintr remains in an unlinked state" )
            {
                REQUIRE( pPipelineSGiesBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSGiesBintr can not LinkAll without setting the Batch Size first", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new  PrimaryGieBintr and PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string pgieInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string pgieModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine";

        std::string secondaryGieName = "secondary-gie";
        std::string sgieInferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string sgieModelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint interval(1);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), sgieInferConfigFile.c_str(), 
            sgieModelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );
        
        pPipelineSGiesBintr->SetInferOnGieId(pPrimaryGieBintr->GetUniqueId());

        WHEN( "The PipelineSGiesBintr is asked to LinkAll prior to setting the Batch Size" )
        {
            REQUIRE( pPipelineSGiesBintr->LinkAll() == false );

            THEN( "The PipelineSGiesBintr remains in an unlinked state" )
            {
                REQUIRE( pPipelineSGiesBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSGiesBintr with its PrimaryGieId and Batch Size set can LinkAll", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string pgieInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string pgieModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine";

        std::string secondaryGieName = "secondary-gie";
        std::string sgieInferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string sgieModelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint interval(1);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), sgieInferConfigFile.c_str(), 
            sgieModelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );

        WHEN( "The the PrimaryGieName and Batch Size is set for the PipelineSGiesBintr" )
        {
            pPipelineSGiesBintr->SetInferOnGieId(pPrimaryGieBintr->GetUniqueId());
            pPipelineSGiesBintr->SetBatchSize(1);

            THEN( "The PipelineSGiesBintr is able to LinkAll" )
            {
                REQUIRE( pPipelineSGiesBintr->LinkAll() == true );
                REQUIRE( pPipelineSGiesBintr->IsLinked() == true );
                REQUIRE( pSecondaryGieBintr->GetBatchSize() == 1 );
            }
        }
    }
}
SCENARIO( "A PipelineSGiesBintr Linked with a SecondaryGieBintr can UnlinkAll", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr Linked with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string pgieInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string pgieModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine";

        std::string secondaryGieName = "secondary-gie";
        std::string sgieInferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string sgieModelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint interval(1);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), sgieInferConfigFile.c_str(), 
            sgieModelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());
            
        pPipelineSGiesBintr->SetInferOnGieId(pPrimaryGieBintr->GetUniqueId());
        pPipelineSGiesBintr->SetBatchSize(1);
        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr) == true );

        WHEN( "The pPipelineSGiesBintr is in a IsLinked state" )
        {
            REQUIRE( pPipelineSGiesBintr->LinkAll() == true );
            REQUIRE( pPipelineSGiesBintr->IsLinked() == true );
            REQUIRE( pSecondaryGieBintr->GetBatchSize() == 1 );

            THEN( "It can UnlinkAll with a single SecondaryGieBintr" )
            {
                pPipelineSGiesBintr->UnlinkAll();
                REQUIRE( pPipelineSGiesBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSGiesBintr with several SecondaryGieBintrs can LinkAll", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with three child SecondearyGieBintrs" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string pgieInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string pgieModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine";

        std::string secondaryGieName1 = "secondary-gie-1";
        std::string secondaryGieName2 = "secondary-gie-2";
        std::string secondaryGieName3 = "secondary-gie-3";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint interval(0);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr1 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName1.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr2 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName2.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr3 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName3.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        pPipelineSGiesBintr->SetInferOnGieId(pPrimaryGieBintr->GetUniqueId());
        pPipelineSGiesBintr->SetBatchSize(3);

        WHEN( "All three SecondaryGieBintrs are added to the pPipelineSGiesBintr" )
        {
            REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr1) == true );
            REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr2) == true );
            REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr3) == true );
            REQUIRE( pPipelineSGiesBintr->IsLinked() == false );

            THEN( "The pPipelineSGiesBintr can update and LinkAll SecondaryGieBintrs" )
            {
                REQUIRE( pPipelineSGiesBintr->LinkAll() == true );
                REQUIRE( pPipelineSGiesBintr->IsLinked() == true );
                REQUIRE( pSecondaryGieBintr1->GetBatchSize() == 3 );
                REQUIRE( pSecondaryGieBintr2->GetBatchSize() == 3 );
                REQUIRE( pSecondaryGieBintr3->GetBatchSize() == 3 );
            }
        }
    }
}

SCENARIO( "A PipelineSGiesBintr with several SecondaryGieBintrs can UnlinkAll", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string pgieInferConfigFile = "./test/configs/config_infer_primary_nano.txt";
        std::string pgieModelEngineFile = "./test/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine";

        std::string secondaryGieName1 = "secondary-gie-1";
        std::string secondaryGieName2 = "secondary-gie-2";
        std::string secondaryGieName3 = "secondary-gie-3";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor_nano.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine";
        uint interval(0);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), pgieInferConfigFile.c_str(), 
            pgieModelEngineFile.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr1 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName1.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr2 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName2.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr3 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName3.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str(), interval);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiesBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        pPipelineSGiesBintr->SetInferOnGieId(pPrimaryGieBintr->GetUniqueId());
        pPipelineSGiesBintr->SetBatchSize(3);

        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr1) == true );
        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr2) == true );
        REQUIRE( pPipelineSGiesBintr->AddChild(pSecondaryGieBintr3) == true );
        REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 3 );

        WHEN( "The pPipelineSGiesBintr is Linked" )
        {
            REQUIRE( pPipelineSGiesBintr->LinkAll() == true );
            REQUIRE( pPipelineSGiesBintr->IsLinked() == true );
                
            THEN( "The pPipelineSGiesBintr can be Unlinked and all SecondaryGieBintrs removed" )
            {
                pPipelineSGiesBintr->UnlinkAll();
                REQUIRE( pPipelineSGiesBintr->IsLinked() == false );
                REQUIRE( pPipelineSGiesBintr->RemoveChild(pSecondaryGieBintr1) == true );
                REQUIRE( pPipelineSGiesBintr->RemoveChild(pSecondaryGieBintr2) == true );
                REQUIRE( pPipelineSGiesBintr->RemoveChild(pSecondaryGieBintr3) == true );
                REQUIRE( pPipelineSGiesBintr->GetNumChildren() == 0 );
            }
        }
    }
}
