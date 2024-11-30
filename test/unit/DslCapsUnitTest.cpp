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
#include "DslApi.h"

using namespace DSL;

SCENARIO( "A DslCaps for Video is constructed correctly", "[Caps]" )
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
                std::string expected_str("video/x-raw(memory:NVMM),format=(string)I420");
                std::string actual_str(Caps.c_str());
                REQUIRE( expected_str == actual_str );
            }
        }
        WHEN( "A CapsFiler is created from string representation" )
        {
            std::string initial_str("video/x-raw(memory:NVMM),format=(string)I420");

            DSL_CAPS_PTR pCaps = DSL_CAPS_NEW(initial_str.c_str());
            
            THEN( "Its member variables are initialized correctly" )
            {
                std::string actual_str(pCaps->c_str());
                REQUIRE( initial_str == actual_str );
            }
        }
    }
}

SCENARIO( "A DslCaps for Audio is constructed correctly", "[Caps]" )
{
    GIVEN( "Possible values for new caps" )
    {
        std::string media("audio/x-raw");
        std::string format("S16LE");
        std::string layout("non-interleaved");
        uint rate(DSL_AUDIO_RESAMPLE_RATE_DEFAULT);
        uint channels(2);
        
        WHEN( "A CapsFiler is created with media, format, and memory feture" )
        {
            DslCaps Caps(media.c_str(), format.c_str(), layout.c_str(), 0, 0);
            
            THEN( "Its member variables are initialized correctly" )
            {
                std::string expected_str(
                    "audio/x-raw,format=(string)S16LE,layout=(string)non-interleaved");
                std::string actual_str(Caps.c_str());
                REQUIRE( expected_str == actual_str );
            }
        }
        WHEN( "A CapsFiler is created from string representation" )
        {
            DslCaps Caps(media.c_str(), format.c_str(), layout.c_str(), rate, channels);
            
            THEN( "Its member variables are initialized correctly" )
            {
                std::string expected_str(
                    "audio/x-raw,format=(string)S16LE,layout=(string)non-interleaved,rate=(int)44100,channels=(int)2");
                std::string actual_str(Caps.c_str());
                REQUIRE( expected_str == actual_str );
            }
        }
    }
}