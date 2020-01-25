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
#include "DslDewarperBintr.h"

using namespace DSL;

SCENARIO( "A Dewarper is created correctly",  "[DewarperBintr]" )
{
    GIVEN( "Attributes for a new Dewarper" ) 
    {
        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        WHEN( "The Dewarper is created" )
        {
            DSL_DEWARPER_PTR pDewarperBintr = 
                DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

            THEN( "The Dewarper's config file is found, loaded, and returned correctly")
            {
                std::string retConfigFile(pDewarperBintr->GetConfigFile());
                
                REQUIRE( retConfigFile == defConfigFile );
            }
        }
    }
}

SCENARIO( "A Dewarper can LinkAll child Elementrs correctly",  "[DewarperBintr]" )
{
    GIVEN( "A new DewarperBintr in memory" ) 
    {
        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

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

SCENARIO( "A Dewarper can UnlinkAll child Elementrs correctly",  "[DewarperBintr]" )
{
    GIVEN( "A new DewarperBintr in memory" ) 
    {
        std::string dewarperName("dewarper");
        std::string defConfigFile("./test/configs/config_dewarper.txt");

        DSL_DEWARPER_PTR pDewarperBintr = 
            DSL_DEWARPER_NEW(dewarperName.c_str(), defConfigFile.c_str());

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
