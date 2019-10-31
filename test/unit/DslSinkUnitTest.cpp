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
#include "DslSinkBintr.h"

SCENARIO( "Set Display Id updates Sink correctly",  "[overlay-sink]" )
{
    GIVEN( "A new Overlay Sink in memory" ) 
    {
        std::string sinkName = "overlay-sink";
        int displayId = 1;

        std::shared_ptr<DSL::OverlaySinkBintr> pSinkBintr = 
            std::shared_ptr<DSL::OverlaySinkBintr>(new DSL::OverlaySinkBintr(
            sinkName.c_str(), 1280, 720, 30, 1));
            
        // ensure display id reflects not is use
        REQUIRE( pSinkBintr->GetDisplayId() == -1 );

        WHEN( "The Display Id is set " )
        {
            pSinkBintr->SetDisplayId(displayId);
            THEN( "The returned Display Id is correct" )
            {
                REQUIRE( pSinkBintr->GetDisplayId() == displayId );
                
            }
        }
    }
}
