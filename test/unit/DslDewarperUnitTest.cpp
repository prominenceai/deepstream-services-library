/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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
#include "DslDewarperBintr.h"

using namespace DSL;

static const std::string dewarperName("dewarper");
static const std::string defConfigFile(
"/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper.txt");

SCENARIO( "A DewarperBintr is created correctly",  "[DewarperBintr]" )
{
    GIVEN( "Attributes for a new Dewarper" ) 
    {
        uint GPUID0(0);

        WHEN( "The Dewarper is created" )
        {
            DSL_DEWARPER_PTR pDewarperBintr = 
                DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str(), 0);

            THEN( "The Dewarper's config file is found, loaded, and returned correctly")
            {
                REQUIRE( pDewarperBintr->GetGpuId() == GPUID0 );
                std::string retConfigFile(pDewarperBintr->GetConfigFile());
                
                REQUIRE( retConfigFile == defConfigFile );
            }
        }
    }
}

SCENARIO( "A DewarperBintr can LinkAll child Elementrs correctly",  "[DewarperBintr]" )
{
    GIVEN( "A new DewarperBintr in memory" ) 
    {
        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str(), 0);

        WHEN( "The DewarperBintr is called to LinkAll" )
        {
            REQUIRE( pDewarperBintr->LinkAll() == true );

            THEN( "The DewarperBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pDewarperBintr->IsLinked() == true );
            }
        }
    }
}

SCENARIO( "A DewarperBintr can UnlinkAll child Elementrs correctly",  "[DewarperBintr]" )
{
    GIVEN( "A new DewarperBintr in memory" ) 
    {
        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str(), 0);

        REQUIRE( pDewarperBintr->LinkAll() == true );
        
        WHEN( "The DewarperBintr is called to UnlinkAll" )
        {
            pDewarperBintr->UnlinkAll();

            THEN( "The DewarperBintr's IsLinked state is updated correctly" )
            {
                REQUIRE( pDewarperBintr->IsLinked() == false );
            }
        }
    }
}

SCENARIO( "A DewarperBintr can Get and Set it's GPU ID",  "[DewarperBintr]" )
{
    GIVEN( "A new DewarperBintr in memory" ) 
    {
        uint GPUID0(0);
        uint GPUID1(1);

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str(), 0);

        REQUIRE( pDewarperBintr->GetGpuId() == GPUID0 );
        
        WHEN( "The DewarperBintr's  GPU ID is set" )
        {
            REQUIRE( pDewarperBintr->SetGpuId(GPUID1) == true );

            THEN( "The correct GPU ID is returned on get" )
            {
                REQUIRE( pDewarperBintr->GetGpuId() == GPUID1 );
            }
        }
    }
}
