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
#include "DslPreprocBintr.h"

using namespace DSL;

static const std::string preprocName("preprocessor");

static const std::string configFile1(
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/config_preprocess.txt");

static const std::string configFile2(
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-action-recognition/config_preprocess_3d_custom.txt");

SCENARIO( "A PreprocBintr is created correctly", "[PreprocBintr]" )
{
    GIVEN( "Attributes for a new PreprocBintr" ) 
    {
        WHEN( "The PreprocBintr is created" )
        {
            DSL_PREPROC_PTR pPreprocBintr = 
                DSL_PREPROC_NEW(preprocName.c_str(), configFile1.c_str());

            THEN( "The PreprocBintr's parameters are set and returned correctly")
            {
                std::string retConfigFile = pPreprocBintr->GetConfigFile();
                REQUIRE( retConfigFile == configFile1 );
                REQUIRE( pPreprocBintr->GetEnabled() == true );
                REQUIRE( pPreprocBintr->GetUniqueId() == 0 );
            }
        }
    }
}

SCENARIO( "A PreprocBintr's config file can be set/get correctly", "[PreprocBintr]" )
{
    GIVEN( "A new PreprocBintr in memory" ) 
    {
        DSL_PREPROC_PTR pPreprocBintr = 
            DSL_PREPROC_NEW(preprocName.c_str(), configFile1.c_str());

        std::string retConfigFile = pPreprocBintr->GetConfigFile();
        REQUIRE( retConfigFile == configFile1 );

        WHEN( "The PreprocBintr's config file is Set" )
        {
            REQUIRE( pPreprocBintr->SetConfigFile(configFile2.c_str()) == true );

            THEN( "The PreprocBintr's new config file is returned on Get")
            {
                std::string retConfigFile = pPreprocBintr->GetConfigFile();
                REQUIRE( retConfigFile == configFile2 );
            }
        }
    }
}

SCENARIO( "A PreprocBintr's enabled setting can be set/get correctly", "[PreprocBintr]" )
{
    GIVEN( "A new PreprocBintr in memory" ) 
    {
        DSL_PREPROC_PTR pPreprocBintr = 
            DSL_PREPROC_NEW(preprocName.c_str(), configFile1.c_str());

        bool retEnabled = pPreprocBintr->GetEnabled();
        REQUIRE( retEnabled == true );

        WHEN( "The PreprocBintr is disabled" )
        {
            REQUIRE( pPreprocBintr->SetEnabled(false) == true );

            THEN( "The PreprocBintr's new config file is returned on Get")
            {
                retEnabled = pPreprocBintr->GetEnabled();
                REQUIRE( retEnabled == false );
            }
        }
    }
}

