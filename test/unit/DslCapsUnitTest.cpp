/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "DslCaps.h"

using namespace DSL;

SCENARIO( "A DslCaps helper is constructed correctly", "[Caps]" )
{
    GIVEN( "Possible values for new caps" )
    {
        std::string media("video/x-raw");
        std::string format("I420");
        
        uint height(1920), width(1280);
        uint fpsN(30), fpsD(1);
        
        WHEN( "A CapsFiler is created with media, format, and memory feture" )
        {
            DslCaps Caps(media.c_str(), format.c_str(), 0, 0, 0, 0, true);
            
            THEN( "Its member variables are initialized correctly" )
            {
                std::string expected("video/x-raw(memory:NVMM),format=(string)I420");
                std::string actual(Caps.c_str());
                REQUIRE( expected == actual );
            }
        }
    }
}
