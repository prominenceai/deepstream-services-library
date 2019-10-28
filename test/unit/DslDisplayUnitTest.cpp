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
#include "DslDisplayBintr.h"

SCENARIO( "A Tiled Display's dimensions can be updated",  "[display]" )
{
    GIVEN( "A new Tiled Display in memory" ) 
    {
        std::string displayName = "tiled-display";
        uint initWidth(10);
        uint initHeight(10);

        std::shared_ptr<DSL::DisplayBintr> pDisplayBintr = 
            std::shared_ptr<DSL::DisplayBintr>(new DSL::DisplayBintr(
            displayName.c_str(), initWidth, initHeight));
            
        uint currWidth(0);
        uint currHeight(0);
    
        pDisplayBintr->GetDimensions(currWidth, currHeight);
        REQUIRE( currWidth == initWidth );
        REQUIRE( currHeight == initHeight );

        WHEN( "The Display's demensions are Set" )
        {
            uint newWidth(1280);
            uint newHeight(720);
            
            pDisplayBintr->SetDimensions(newWidth, newHeight);

            THEN( "The Display's new demensions are returned on Get")
            {
                pDisplayBintr->GetDimensions(currWidth, currHeight);
                REQUIRE( currWidth == newWidth );
                REQUIRE( currHeight == newHeight );
            }
        }
    }
}

SCENARIO( "A Tiled Display's tiles can be updated",  "[display]" )
{
    GIVEN( "A new Tiled Display in memory" ) 
    {
        std::string displayName = "tiled-display";

        std::shared_ptr<DSL::DisplayBintr> pDisplayBintr = 
            std::shared_ptr<DSL::DisplayBintr>(new DSL::DisplayBintr(
            displayName.c_str(), 1280, 720));
            
        uint currRows(0);
        uint currColumns(0);
    
        pDisplayBintr->GetTiles(currRows, currColumns);
        REQUIRE( currRows == 1 );
        REQUIRE( currColumns == 1 );

        WHEN( "The Display's demensions are Set" )
        {
            uint newRows(10);
            uint newColumns(10);
            
            pDisplayBintr->SetTiles(newRows, newColumns);

            THEN( "The Display's new demensions are returned on Get")
            {
                pDisplayBintr->GetTiles(currRows, currColumns);
                REQUIRE( currRows == newRows );
                REQUIRE( currColumns == newColumns );
            }
        }
    }
}