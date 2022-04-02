/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

static const std::string primaryGieName = "primary-gie";
static const std::string secondaryGieName = "secondary-gie";
static const std::string inferConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app-trtis/config_infer_secondary_plan_engine_carcolor.txt");

SCENARIO( "A new SecondaryTisBintr is created correctly",  "[SecondaryTisBintr]" )
{
    GIVEN( "Attributes for a new SecondaryTisBintr" ) 
    {
        uint interval(1);
        
        WHEN( "A new SecondaryTisBintr is created" )
        {
            DSL_SECONDARY_TIS_PTR pSecondaryTisBintr = 
                DSL_SECONDARY_TIS_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
                    primaryGieName.c_str(), interval);

            THEN( "The SecondaryTisBintr's memebers are setup and returned correctly" )
            {
                std::string returnedInferConfigFile = pSecondaryTisBintr->GetInferConfigFile();
                REQUIRE( returnedInferConfigFile == inferConfigFile );

                REQUIRE( pSecondaryTisBintr->GetBatchSize() == 0 );
                REQUIRE( pSecondaryTisBintr->GetInterval() == interval );
                REQUIRE( pSecondaryTisBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "After setting batch size, a new SecondaryTisBintr can LinkAll Child Elementrs",  "[SecondaryTisBintr]" )
{
    GIVEN( "A new SecondaryTisBintr in an Unlinked state" ) 
    {
        uint interval(1);
        
        DSL_SECONDARY_TIS_PTR pSecondaryTisBintr = 
            DSL_SECONDARY_TIS_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            primaryGieName.c_str(), interval);

        WHEN( "A new SecondaryTisBintr is Linked" )
        {
            pSecondaryTisBintr->SetBatchSize(1);
            REQUIRE( pSecondaryTisBintr->IsLinked() == false );
            REQUIRE( pSecondaryTisBintr->LinkAll() == true );
            
            THEN( "The SecondaryTisBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSecondaryTisBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A Linked SecondaryTisBintr can UnlinkAll Child Elementrs",  "[SecondaryTisBintr]" )
{
    GIVEN( "A Linked SecondaryTisBintr" ) 
    {
        uint interval(1);

        DSL_SECONDARY_TIS_PTR pSecondaryTisBintr = 
            DSL_SECONDARY_TIS_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            primaryGieName.c_str(), interval);

        pSecondaryTisBintr->SetBatchSize(1);
        REQUIRE( pSecondaryTisBintr->LinkAll() == true );

        WHEN( "A SecondaryTisBintr is Unlinked" )
        {
            pSecondaryTisBintr->UnlinkAll();
            
            THEN( "The SecondaryTisBintr IsLinked state is updated correctly" )
            {
                REQUIRE( pSecondaryTisBintr->IsLinked() == false );
            }
        }
    }
}
SCENARIO( "A Linked SecondaryTisBintr can not be linked again",  "[SecondaryTisBintr]" )
{
    GIVEN( "A new SecondaryTisBintr" ) 
    {
        uint interval(1);

        DSL_SECONDARY_TIS_PTR pSecondaryTisBintr = 
            DSL_SECONDARY_TIS_NEW(secondaryGieName.c_str(), inferConfigFile.c_str(), 
            primaryGieName.c_str(), interval);

        WHEN( "A SecondaryTisBintr is Linked" )
        {
            pSecondaryTisBintr->SetBatchSize(1);
            REQUIRE( pSecondaryTisBintr->LinkAll() == true );
            
            THEN( "The SecondaryTisBintr can not be linked again" )
            {
                REQUIRE( pSecondaryTisBintr->LinkAll() == false );
            }
        }
    }
}
