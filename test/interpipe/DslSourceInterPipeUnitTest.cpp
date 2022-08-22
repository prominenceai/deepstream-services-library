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
#include "DslApi.h"
#include "DslSinkBintr.h"
#include "DslSourceBintr.h"
#include "DslPipelineSourcesBintr.h"

static std::string interPipeSourceName("inter-pipe-source");
static std::string interPipeSinkName("inter-pipe-sink");
static bool isLive(true); 
static bool acceptEos(false); 
static bool acceptEvents(false);

using namespace DSL;

SCENARIO( "A new InterPipeSourceBintr is created correctly",
    "[InterPipeSourceBintr]" )
{
    GIVEN( "Attributes for a new InterPipeSourceBintr" ) 
    {
        WHEN( "The InterPipeSourceBintr is created " )
        {
            
            DSL_INTERPIPE_SOURCE_PTR pSourceBintr = DSL_INTERPIPE_SOURCE_NEW(
                interPipeSourceName.c_str(), interPipeSinkName.c_str(), isLive,
                acceptEos, acceptEvents);

            THEN( "All member variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->IsInUse() == false );
                REQUIRE( pSourceBintr->IsLive() == true );
                
                std::string retListenTo(pSourceBintr->GetListenTo());
                REQUIRE( retListenTo == interPipeSinkName );
                
                bool retAcceptEos(true), retAcceptEvents(true);
                pSourceBintr->GetAcceptSettings(&retAcceptEos, &retAcceptEvents);

                REQUIRE( retAcceptEos == false );
                REQUIRE( retAcceptEvents == false );
            }
        }
    }
}

SCENARIO( "A linked InterPipeSourceBintr can update its listen-to setting",
    "[InterPipeSourceBintr]" )
{
    GIVEN( "A new InterPipeSourceBintr" ) 
    {
        DSL_INTERPIPE_SOURCE_PTR pSourceBintr = DSL_INTERPIPE_SOURCE_NEW(
            interPipeSourceName.c_str(), interPipeSinkName.c_str(), isLive,
            acceptEos, acceptEvents);

        WHEN( "The InterPipeSourceBintr is linked " )
        {
            pSourceBintr->LinkAll();
            
            THEN( "All memeber variables are initialized correctly" )
            {
                std::string newListenTo("other-inter-pipe-sink");
                pSourceBintr->SetListenTo(newListenTo.c_str());

                std::string retListenTo(pSourceBintr->GetListenTo());
                REQUIRE( retListenTo == newListenTo );
            }
        }
    }
}

SCENARIO( "A linked InterPipeSourceBintr fails to set Accept settings",
    "[InterPipeSourceBintr]" )
{
    GIVEN( "A new InterPipeSourceBintr" ) 
    {
        DSL_INTERPIPE_SOURCE_PTR pSourceBintr = DSL_INTERPIPE_SOURCE_NEW(
            interPipeSourceName.c_str(), interPipeSinkName.c_str(), isLive,
            acceptEos, acceptEvents);

        WHEN( "The InterPipeSourceBintr is linked " )
        {
            pSourceBintr->LinkAll();
            
            THEN( "Setting the Accepts settings must fail" )
            {
                bool newAcceptEos(true), newAcceptEvents(true);
                REQUIRE( pSourceBintr->SetAcceptSettings(newAcceptEos, 
                    newAcceptEvents) == false );
            }
        }
    }
}

