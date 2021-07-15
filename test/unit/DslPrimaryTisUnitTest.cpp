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

using namespace DSL;

static const std::string primaryTisName("primary-tis");
static const std::string inferConfigFile(
    "/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt");
    
SCENARIO( "A new PrimaryTisBintr is created correctly",  "[PrimaryTisBintr]" )
{
    GIVEN( "Attributes for a new PrimaryTisBintr" ) 
    {
        uint interval(1);

        WHEN( "A new PrimaryTisBintr is created" )
        {
            DSL_PRIMARY_TIS_PTR pPrimaryTisBintr = 
                DSL_PRIMARY_TIS_NEW(primaryTisName.c_str(), inferConfigFile.c_str(), interval);

            THEN( "The PrimaryTisBintr's memebers are setup and returned correctly" )
            {
                std::string returnedInferConfigFile = pPrimaryTisBintr->GetInferConfigFile();
                REQUIRE( pPrimaryTisBintr->GetInferType() == DSL_INFER_TYPE_TIS);
                REQUIRE( returnedInferConfigFile == inferConfigFile );
                REQUIRE( interval == pPrimaryTisBintr->GetInterval() );
                REQUIRE( pPrimaryTisBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A new PrimaryTisBintr with its Batch Size set can LinkAll Child Elementrs",  "[PrimaryTisBintr]" )
{
    GIVEN( "A new PrimaryTisBintr in an Unlinked state" ) 
    {
        std::string primaryTisName("primary-tis");
        std::string inferConfigFile(
            "/opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_plan_engine_primary.txt");
        uint interval(1);

        DSL_PRIMARY_TIS_PTR pPrimaryTisBintr = 
            DSL_PRIMARY_TIS_NEW(primaryTisName.c_str(), inferConfigFile.c_str(), interval);

        WHEN( "The Batch Size is set and the PrimaryTisBintr is asked to LinkAll" )
        {
            pPrimaryTisBintr->SetBatchSize(1);
            REQUIRE( pPrimaryTisBintr->LinkAll() == true );
            
            THEN( "The PrimaryTisBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pPrimaryTisBintr->IsLinked() == true );
            }
        }
    }
}


SCENARIO( "A Linked PrimaryTisBintr can UnlinkAll Child Elementrs",  "[PrimaryTisBintr]" )
{
    GIVEN( "A Linked PrimaryTisBintr" ) 
    {
        uint interval(1);

        DSL_PRIMARY_TIS_PTR pPrimaryTisBintr = 
            DSL_PRIMARY_TIS_NEW(primaryTisName.c_str(), inferConfigFile.c_str(), interval);

        pPrimaryTisBintr->SetBatchSize(1);
        REQUIRE( pPrimaryTisBintr->LinkAll() == true );

        WHEN( "A new PrimaryTisBintr is created" )
        {
            pPrimaryTisBintr->UnlinkAll();
            
            THEN( "The PrimaryTisBintr can LinkAll Child Elementrs" )
            {
                REQUIRE( pPrimaryTisBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A Linked PrimaryTisBintr can not be linked again", "[PrimaryTisBintr]" )
{
    GIVEN( "A new PrimaryTisBintr in an Unlinked state" ) 
    {
        uint interval(1);

        DSL_PRIMARY_TIS_PTR pPrimaryTisBintr = 
            DSL_PRIMARY_TIS_NEW(primaryTisName.c_str(), inferConfigFile.c_str(), interval);

        WHEN( "A new PrimaryTisBintr is Linked" )
        {
            pPrimaryTisBintr->SetBatchSize(1);
            REQUIRE( pPrimaryTisBintr->LinkAll() == true );
            
            THEN( "The PrimaryTisBintr can not be linked again" )
            {
                REQUIRE( pPrimaryTisBintr->LinkAll() == false );
            }
        }
    }
}
