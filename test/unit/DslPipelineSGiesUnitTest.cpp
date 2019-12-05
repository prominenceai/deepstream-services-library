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
            DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
                DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSGiessBintr->GetName() == pipelineSGiesName );
                REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 0 );
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
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";
        uint primaryUniqueId = std::hash<std::string>{}(primaryGieName.c_str());
        uint secondaryUniqueId = std::hash<std::string>{}(secondaryGieName.c_str());

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        REQUIRE( pSecondaryGieBintr->GetUniqueId() == secondaryUniqueId);
        REQUIRE( pSecondaryGieBintr->GetInferOnGieUniqueId() == primaryUniqueId);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        WHEN( "The SecondaryGie is added to the PipelineSGiesBintr" )
        {
            REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );
            
            THEN( "All members have been setup correctly" )
            {
                REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 1 );
                REQUIRE( pPipelineSGiessBintr->IsChild(pSecondaryGieBintr) == true );
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
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";
        uint primaryUniqueId = std::hash<std::string>{}(primaryGieName.c_str());
        uint secondaryUniqueId = std::hash<std::string>{}(secondaryGieName.c_str());

//        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        REQUIRE( pSecondaryGieBintr->GetUniqueId() == secondaryUniqueId);
        REQUIRE( pSecondaryGieBintr->GetInferOnGieUniqueId() == primaryUniqueId);

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );
        REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 1 );

        WHEN( "The SecondearyGieBintr is removed from the PipelineSGiesBintr" )
        {
            REQUIRE( pPipelineSGiessBintr->RemoveChild(pSecondaryGieBintr) == true );
            
            THEN( "All members have been updated correctly" )
            {
                REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 0 );
                REQUIRE( pPipelineSGiessBintr->IsChild(pSecondaryGieBintr) == false );
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
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

//        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        WHEN( "The SecondaryGieBintr is added to the pPipelineSGiessBintr" )
        {
            REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );
            REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 1 );
            THEN( "It can not be added a second time" )
            {
                REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == false );
                REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 1 );
            }
        }
    }
}


SCENARIO( "A PipelineSGiesBintr can not LinkAll without setting the PrimaryGieName first", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );

        WHEN( "The PipelineSGiessBintr is asked to LinkAll prior to setting the PrimaryGieName" )
        {
            REQUIRE( pPipelineSGiessBintr->LinkAll() == false );

            THEN( "The PipelineSGiessBintr remains in an unlinked state" )
            {
                REQUIRE( pPipelineSGiessBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSGiesBintr can not LinkAll without setting the Batch Size first", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );
        
        pPipelineSGiessBintr->SetPrimaryGieName(primaryGieName.c_str());

        WHEN( "The PipelineSGiessBintr is asked to LinkAll prior to setting the Batch Size" )
        {
            REQUIRE( pPipelineSGiessBintr->LinkAll() == false );

            THEN( "The PipelineSGiessBintr remains in an unlinked state" )
            {
                REQUIRE( pPipelineSGiessBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PipelineSGiesBintr with its PrimaryGie Name and Batch Size set can LinkAll", "[PipelineSGiesBintr]" )
{
    GIVEN( "A new PipelineSGiesBintr with a child SecondearyGieBintr" ) 
    {
        std::string pipelineSGiesName = "pipeline-sgies";

        std::string primaryGieName = "primary-gie";
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );

        WHEN( "The the PrimaryGieName and Batch Size is set for the PipelineSGiessBintr" )
        {
            pPipelineSGiessBintr->SetPrimaryGieName(primaryGieName.c_str());
            pPipelineSGiessBintr->SetBatchSize(1);

            THEN( "The PipelineSGiessBintr is able to LinkAll" )
            {
                REQUIRE( pPipelineSGiessBintr->LinkAll() == true );
                REQUIRE( pPipelineSGiessBintr->IsLinked() == true );
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
        std::string secondaryGieName = "secondary-gie";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

        uint interval(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());
            
        pPipelineSGiessBintr->SetPrimaryGieName(primaryGieName.c_str());
        pPipelineSGiessBintr->SetBatchSize(1);
        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr) == true );

        WHEN( "The pPipelineSGiessBintr is in a IsLinked state" )
        {
            REQUIRE( pPipelineSGiessBintr->LinkAll() == true );
            REQUIRE( pPipelineSGiessBintr->IsLinked() == true );
            REQUIRE( pSecondaryGieBintr->GetBatchSize() == 1 );

            THEN( "It can UnlinkAll with a single SecondaryGieBintr" )
            {
                pPipelineSGiessBintr->UnlinkAll();
                REQUIRE( pPipelineSGiessBintr->IsLinked() == false );
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
        std::string secondaryGieName1 = "secondary-gie-1";
        std::string secondaryGieName2 = "secondary-gie-2";
        std::string secondaryGieName3 = "secondary-gie-3";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr1 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName1.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr2 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName2.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr3 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName3.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        pPipelineSGiessBintr->SetPrimaryGieName(primaryGieName.c_str());
        pPipelineSGiessBintr->SetBatchSize(3);

        WHEN( "All three SecondaryGieBintrs are added to the pPipelineSGiessBintr" )
        {
            REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr1) == true );
            REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr2) == true );
            REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr3) == true );
            REQUIRE( pPipelineSGiessBintr->IsLinked() == false );

            THEN( "The pPipelineSGiessBintr can update and LinkAll SecondaryGieBintrs" )
            {
                REQUIRE( pPipelineSGiessBintr->LinkAll() == true );
                REQUIRE( pPipelineSGiessBintr->IsLinked() == true );
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
        std::string secondaryGieName1 = "secondary-gie-1";
        std::string secondaryGieName2 = "secondary-gie-2";
        std::string secondaryGieName3 = "secondary-gie-3";
        std::string inferConfigFile = "./test/configs/config_infer_secondary_carcolor.txt";
        std::string modelEngineFile = "./test/models/Secondary_CarColor/resnet18.caffemodel";

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr1 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName1.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr2 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName2.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr3 = 
            DSL_SECONDARY_GIE_NEW(secondaryGieName3.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), primaryGieName.c_str());

        DSL_PIPELINE_SGIES_PTR pPipelineSGiessBintr = 
            DSL_PIPELINE_SGIES_NEW(pipelineSGiesName.c_str());

        pPipelineSGiessBintr->SetPrimaryGieName(primaryGieName.c_str());
        pPipelineSGiessBintr->SetBatchSize(3);

        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr1) == true );
        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr2) == true );
        REQUIRE( pPipelineSGiessBintr->AddChild(pSecondaryGieBintr3) == true );
        REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 3 );

        WHEN( "The pPipelineSGiessBintr is Linked" )
        {
            REQUIRE( pPipelineSGiessBintr->LinkAll() == true );
            REQUIRE( pPipelineSGiessBintr->IsLinked() == true );
                
            THEN( "The pPipelineSGiessBintr can be Unlinked and all SecondaryGieBintrs removed" )
            {
                pPipelineSGiessBintr->UnlinkAll();
                REQUIRE( pPipelineSGiessBintr->IsLinked() == false );
                REQUIRE( pPipelineSGiessBintr->RemoveChild(pSecondaryGieBintr1) == true );
                REQUIRE( pPipelineSGiessBintr->RemoveChild(pSecondaryGieBintr2) == true );
                REQUIRE( pPipelineSGiessBintr->RemoveChild(pSecondaryGieBintr3) == true );
                REQUIRE( pPipelineSGiessBintr->GetNumChildren() == 0 );
            }
        }
    }
}
