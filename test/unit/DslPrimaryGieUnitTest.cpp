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
static std::string inferConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary_nano.txt");
static std::string modelEngineFile(
    "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector_Nano/resnet10.caffemodel_b8_gpu0_fp16.engine");

static uint interval(1);

using namespace DSL;

SCENARIO( "A new PrimaryGieBintr is created correctly",  "[PrimaryGieBintr]" )
{
    GIVEN( "Attributes for a new PrimaryGieBintr" ) 
    {
        
        WHEN( "A new PrimaryGieBintr is created" )
        {
            
            DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
                DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
                modelEngineFile.c_str(), interval);

            THEN( "The PrimaryGieBintr's memebers are setup and returned correctly" )
            {
                std::string returnedInferConfigFile = pPrimaryGieBintr->GetInferConfigFile();
                REQUIRE( returnedInferConfigFile == inferConfigFile );

                std::string returnedModelEngineFile = pPrimaryGieBintr->GetModelEngineFile();
                REQUIRE( returnedModelEngineFile == modelEngineFile );
                
                REQUIRE( interval == pPrimaryGieBintr->GetInterval() );
                
                REQUIRE( pPrimaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new PrimaryGieBintr can not LinkAll without setting the Batch Size first",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in an Unlinked state" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        WHEN( "A new PrimaryGieBintr is requested to LinkAll prior to setting the Batch Size" )
        {
            REQUIRE( pPrimaryGieBintr->LinkAll() == false );
            
            THEN( "The PrimaryGieBintr remains in an Unlinked state" )
            {
                REQUIRE( pPrimaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new PrimaryGieBintr with its Batch Size set can LinkAll Child Elementrs",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in an Unlinked state" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        WHEN( "The Batch Size is set and the PrimaryGieBintr is asked to LinkAll" )
        {
            pPrimaryGieBintr->SetBatchSize(1);
            REQUIRE( pPrimaryGieBintr->LinkAll() == true );
            
            THEN( "The PrimaryGieBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pPrimaryGieBintr->IsLinked() == true );
            }
        }
    }
}


SCENARIO( "A Linked PrimaryGieBintr can UnlinkAll Child Elementrs",  "[PrimaryGieBintr]" )
{
    GIVEN( "A Linked PrimaryGieBintr" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        pPrimaryGieBintr->SetBatchSize(1);
        REQUIRE( pPrimaryGieBintr->LinkAll() == true );

        WHEN( "A new PrimaryGieBintr is created" )
        {
            pPrimaryGieBintr->UnlinkAll();
            
            THEN( "The PrimaryGieBintr can LinkAll Child Elementrs" )
            {
                REQUIRE( pPrimaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Linked PrimaryGieBintr can not be linked again", "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in an Unlinked state" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        WHEN( "A new PrimaryGieBintr is Linked" )
        {
            pPrimaryGieBintr->SetBatchSize(1);
            REQUIRE( pPrimaryGieBintr->LinkAll() == true );
            
            THEN( "The PrimaryGieBintr can not be linked again" )
            {
                REQUIRE( pPrimaryGieBintr->LinkAll() == false );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr can Get and Set its GPU ID",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        uint GPUID0(0);
        uint GPUID1(1);

        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        REQUIRE( pPrimaryGieBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The PrimaryGieBintr's  GPU ID is set" )
        {
            REQUIRE( pPrimaryGieBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pPrimaryGieBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr can Enable and Disable raw layer info output",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);
        
        WHEN( "The PrimaryGieBintr's raw output is enabled" )
        {
            REQUIRE( pPrimaryGieBintr->SetRawOutputEnabled(true, "./") == true );

            THEN( "The raw output can then be disabled" )
            {
                REQUIRE( pPrimaryGieBintr->SetRawOutputEnabled(false, "") == true );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr fails to Enable raw layer info output given a bad path",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);
        
        WHEN( "A bad path is constructed" )
        {
            std::string badPath("this/is/an/invalid/path");
            
            THEN( "The raw output will fail to enale" )
            {
                REQUIRE( pPrimaryGieBintr->SetRawOutputEnabled(true, badPath.c_str()) == false );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr can Get and Set its Interval",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        REQUIRE( pPrimaryGieBintr->GetInterval() == interval );
        
        WHEN( "The PrimaryGieBintr's Interval is set" )
        {
            uint newInterval = 5;
            REQUIRE( pPrimaryGieBintr->SetInterval(newInterval) == true );

            THEN( "The correct Interval is returned on get" )
            {
                REQUIRE( pPrimaryGieBintr->GetInterval() == newInterval );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr in a Linked state fails to Set its Interval",  "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        REQUIRE( pPrimaryGieBintr->GetInterval() == interval );
        
        WHEN( "The PrimaryGieBintr is Linked" )
        {
            pPrimaryGieBintr->SetBatchSize(1);
            REQUIRE( pPrimaryGieBintr->LinkAll() == true );

            uint newInterval = 5;

            THEN( "The PrimaryGieBintr fails on SetInterval" )
            {
                REQUIRE( pPrimaryGieBintr->SetInterval(newInterval) == false );

                pPrimaryGieBintr->UnlinkAll();
                REQUIRE( pPrimaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr in a Linked state fails to Set its tensor-meta settings correctly",  
    "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        bool inputEnabled(true), outputEnabled(true);
        
        pPrimaryGieBintr->GetTensorMetaSettings(&inputEnabled, 
            &outputEnabled);
            
        REQUIRE( inputEnabled == false );
        REQUIRE( outputEnabled == false );
        
        WHEN( "The PrimaryGieBintr is Linked" )
        {
            pPrimaryGieBintr->SetBatchSize(1);
            REQUIRE( pPrimaryGieBintr->LinkAll() == true );

            inputEnabled = true;
            outputEnabled = true;

            THEN( "The PrimaryGieBintr fails on SetInterval" )
            {
                REQUIRE( pPrimaryGieBintr->SetTensorMetaSettings(
                    inputEnabled, outputEnabled) == false );

                pPrimaryGieBintr->GetTensorMetaSettings(&inputEnabled, 
                    &outputEnabled);
                    
                REQUIRE( inputEnabled == false );
                REQUIRE( outputEnabled == false );

                pPrimaryGieBintr->UnlinkAll();
                REQUIRE( pPrimaryGieBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A PrimaryGieBintr manages its batch-size settings correctly", 
    "[PrimaryGieBintr]" )
{
    GIVEN( "A new PrimaryGieBintr in memory" ) 
    {
        DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
            DSL_PRIMARY_GIE_NEW(primaryGieName.c_str(), inferConfigFile.c_str(), 
            modelEngineFile.c_str(), interval);

        WHEN( "The Client sets the PrimaryGieBintr's batch-size" )
        {
            uint clientBatchSize = 5;
            REQUIRE( pPrimaryGieBintr->SetBatchSizeByClient(clientBatchSize) == true );

            THEN( "The batch-size is not updated on internal/pipeline call" )
            {
                REQUIRE( pPrimaryGieBintr->SetBatchSize(10) == true );
                REQUIRE( pPrimaryGieBintr->GetBatchSize() == clientBatchSize );
            }
        }
        WHEN( "The Client sets and then clears the PrimaryGieBintr's batch-size" )
        {
            REQUIRE( pPrimaryGieBintr->SetBatchSizeByClient(10) == true );
            REQUIRE( pPrimaryGieBintr->SetBatchSizeByClient(0) == true );

            THEN( "The batch-size is correctly updated on internal/pipeline call" )
            {
                uint newBatchSize(2);
                REQUIRE( pPrimaryGieBintr->SetBatchSize(newBatchSize) == true );
                REQUIRE( pPrimaryGieBintr->GetBatchSize() == newBatchSize );
            }
        }
    }
}

