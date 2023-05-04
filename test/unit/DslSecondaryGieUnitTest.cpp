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
#include "DslInferBintr.h"

static std::string primaryGieName("primary-gie");
static std::string secondaryGieName("secondary-gie");
static std::string inferConfigFileJetson(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carcolor.txt");
static std::string modelEngineFileJetson(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/resnet18.caffemodel_b8_gpu0_fp16.engine");
static std::string inferConfigFileDgpu(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carcolor.txt");
static std::string modelEngineFileDgpu(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/resnet18.resnet18.caffemodel_b8_gpu0_int8.engine");
    
static std::string newInferConfigFileJetson(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carmake.txt");
static std::string newModelEngineFileJetson(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarMake/resnet18.caffemodel_b8_gpu0_fp16.engine");
static std::string newInferConfigFileDgpu(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_carmake.txt");
static std::string newModelEngineFileDgpu(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarMake/resnet18.caffemodel_b8_gpu0_int8.engine");
    
static uint interval(1);

using namespace DSL;

SCENARIO( "A new SecondaryGieBintr is created correctly",  "[SecondaryGieBintr]" )
{
    GIVEN( "Attributes for a new SecondaryGieBintr" ) 
    {
        WHEN( "A new SecondaryGieBintr is created" )
        {
            DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
            if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
            {
                pSecondaryGieBintr = 
                    DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                    inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                    primaryGieName.c_str(), interval);
            }
            else
            {
                pSecondaryGieBintr = 
                    DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                    inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                    primaryGieName.c_str(), interval);
            }

            THEN( "The SecondaryGieBintr's memebers are setup and returned correctly" )
            {
                std::string returnedInferConfigFile = pSecondaryGieBintr->GetInferConfigFile();
                std::string returnedModelEngineFile = pSecondaryGieBintr->GetModelEngineFile();

                if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
                {
                    REQUIRE( returnedInferConfigFile == inferConfigFileJetson );
                    REQUIRE( returnedModelEngineFile == modelEngineFileJetson );
                }
                else
                {
                    REQUIRE( returnedInferConfigFile == inferConfigFileDgpu );
                    REQUIRE( returnedModelEngineFile == modelEngineFileDgpu );
                }
                
                REQUIRE( pSecondaryGieBintr->GetBatchSize() == 0 );
                REQUIRE( pSecondaryGieBintr->GetInterval() == interval );
                REQUIRE( pSecondaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A SecondaryGieBintr can not LinkAll before setting batch size",  "[SecondaryGieBintr]" )
{
    GIVEN( "A new SecondaryGieBintr in an Unlinked state" ) 
    {
        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                primaryGieName.c_str(), interval);
        }
        else
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                primaryGieName.c_str(), interval);
        }

        WHEN( "A new SecondaryGieBintr is called to LinkAll" )
        {
            REQUIRE( pSecondaryGieBintr->IsLinked() == false );
            REQUIRE( pSecondaryGieBintr->LinkAll() == false );
            
            THEN( "The SecondaryGieBintr IsLinked state is not updated" )
            {
                REQUIRE( pSecondaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "After setting batch size, a new SecondaryGieBintr can LinkAll Child Elementrs",  "[SecondaryGieBintr]" )
{
    GIVEN( "A new SecondaryGieBintr in an Unlinked state" ) 
    {
        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                primaryGieName.c_str(), interval);
        }
        else
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                primaryGieName.c_str(), interval);
        }

        WHEN( "A new SecondaryGieBintr is Linked" )
        {
            pSecondaryGieBintr->SetBatchSize(1);
            REQUIRE( pSecondaryGieBintr->IsLinked() == false );
            REQUIRE( pSecondaryGieBintr->LinkAll() == true );
            
            THEN( "The SecondaryGieBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSecondaryGieBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked SecondaryGieBintr can UnlinkAll Child Elementrs",  "[SecondaryGieBintr]" )
{
    GIVEN( "A Linked SecondaryGieBintr" ) 
    {
        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                primaryGieName.c_str(), interval);
        }
        else
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                primaryGieName.c_str(), interval);
        }

        pSecondaryGieBintr->SetBatchSize(1);
        REQUIRE( pSecondaryGieBintr->LinkAll() == true );

        WHEN( "A SecondaryGieBintr is Unlinked" )
        {
            pSecondaryGieBintr->UnlinkAll();
            
            THEN( "The SecondaryGieBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSecondaryGieBintr->IsLinked() == false );
            }
        }
    }
}
SCENARIO( "A Linked SecondaryGieBintr can not be linked again",  "[SecondaryGieBintr]" )
{
    GIVEN( "A new SecondaryGieBintr" ) 
    {
        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                primaryGieName.c_str(), interval);
        }
        else
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                primaryGieName.c_str(), interval);
        }

        WHEN( "A SecondaryGieBintr is Linked" )
        {
            pSecondaryGieBintr->SetBatchSize(1);
            REQUIRE( pSecondaryGieBintr->LinkAll() == true );
            
            THEN( "The SecondaryGieBintr can not be linked again" )
            {
                REQUIRE( pSecondaryGieBintr->LinkAll() == false );
            }
        }
    }
}

SCENARIO( "A SecondaryGieBintr can Get and Set its GPU ID",  "[SecondaryGieBintr]" )
{
    GIVEN( "A new SecondaryGieBintr in memory" ) 
    {
        uint GPUID0(0);
        uint GPUID1(1);

        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                primaryGieName.c_str(), interval);
        }
        else
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                primaryGieName.c_str(), interval);
        }

        REQUIRE( pSecondaryGieBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The SecondaryGieBintr's  GPU ID is set" )
        {
            REQUIRE( pSecondaryGieBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pSecondaryGieBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}

SCENARIO( "A SecondaryGieBintr can Set and Get its Infer Config and Model Engine Files",  "[SecondaryGieBintr]" )
{
    GIVEN( "A new SecondaryGieBintr in memory" ) 
    {
        DSL_SECONDARY_GIE_PTR pSecondaryGieBintr;
        if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileJetson.c_str(), modelEngineFileJetson.c_str(), 
                primaryGieName.c_str(), interval);
        }
        else
        {
            pSecondaryGieBintr = 
                DSL_SECONDARY_GIE_NEW(secondaryGieName.c_str(), 
                inferConfigFileDgpu.c_str(), modelEngineFileDgpu.c_str(), 
                primaryGieName.c_str(), interval);
        }

        WHEN( "The SecondaryGieBintr's Infer Config File is set" )
        {
            if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
            {
                REQUIRE( pSecondaryGieBintr->SetInferConfigFile(
                    newInferConfigFileJetson.c_str()) == true );
                REQUIRE( pSecondaryGieBintr->SetModelEngineFile(
                    newModelEngineFileJetson.c_str()) == true );

            }
            else
            {
                REQUIRE( pSecondaryGieBintr->SetInferConfigFile(
                    newInferConfigFileDgpu.c_str()) == true );
                REQUIRE( pSecondaryGieBintr->SetModelEngineFile(
                    newModelEngineFileDgpu.c_str()) == true );
            }
            
            THEN( "The correct Infer Config File is returned on get" )
            {
                std::string retInferConfigFile = pSecondaryGieBintr->GetInferConfigFile();
                std::string retModelEngineFile = pSecondaryGieBintr->GetModelEngineFile();
                if (dsl_info_gpu_type_get(0) == DSL_GPU_TYPE_INTEGRATED)
                {
                    REQUIRE( retInferConfigFile == newInferConfigFileJetson );
                    REQUIRE( retModelEngineFile == newModelEngineFileJetson );
                }
                else
                {
                    REQUIRE( retInferConfigFile == newInferConfigFileDgpu );
                    REQUIRE( retModelEngineFile == newModelEngineFileDgpu );
                }
            }
        }
    }
}

