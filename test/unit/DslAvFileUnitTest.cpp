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
#include "DslAvFile.h"

using namespace DSL;

SCENARIO( "An AvFile utility can read a MP4 file correctly",  "[AvFile]" )
{
    GIVEN( "A file path to an MP4 file" ) 
    {
        std::string filepath(
            "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4");
    
        WHEN( "When the AvFile object is created" )
        {
            AvFile avFile(filepath.c_str());
            
            THEN( "The AvFile properties are setup correctly")
            {
                REQUIRE( avFile.videoWidth == 1920 );
                REQUIRE( avFile.videoHeight == 1080 );
                REQUIRE( avFile.fpsN == 30 );
                REQUIRE( avFile.fpsD == 1 );
            }
        }
    }
}

SCENARIO( "An AvFile utility can read a MOV file correctly",  "[AvFile]" )
{
    GIVEN( "A file path to an MOV file" ) 
    {
        std::string filepath(
            "/opt/nvidia/deepstream/deepstream/samples/streams/sample_push.mov");
    
        WHEN( "When the AvFile object is created" )
        {
            AvFile avFile(filepath.c_str());
            
            THEN( "The AvFile properties are setup correctly")
            {
                // based on imperical results for this .mov file
                REQUIRE( avFile.videoWidth == 1920 );
                REQUIRE( avFile.videoHeight == 1080 );
                REQUIRE( avFile.fpsN == 30000 ); 
                REQUIRE( avFile.fpsD == 1001 );
            }
        }
    }
}

SCENARIO( "An AvFile utility can read a JPG file correctly",  "[AvFile]" )
{
    GIVEN( "A file path to a JPG file" ) 
    {
        std::string filepath(
            "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg");
    
        WHEN( "When the AvFile object is created" )
        {
            AvFile avFile(filepath.c_str());
            
            THEN( "The AvFile properties are setup correctly")
            {
                // based on imperical results for this .mov file
                REQUIRE( avFile.videoWidth == 1280 );
                REQUIRE( avFile.videoHeight == 720 );
                REQUIRE( avFile.fpsN == 25 ); 
                REQUIRE( avFile.fpsD == 1 );
            }
        }
    }
}

SCENARIO( "An AvFile utility can read a MJPG file correctly",  "[AvFile]" )
{
    GIVEN( "A file path to a MJPG file" ) 
    {
        std::string filepath(
            "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mjpeg");
    
        WHEN( "When the AvFile object is created" )
        {
            AvFile avFile(filepath.c_str());
            
            THEN( "The AvFile properties are setup correctly")
            {
                // based on imperical results for this .mov file
                REQUIRE( avFile.videoWidth == 1280 );
                REQUIRE( avFile.videoHeight == 720 );
                REQUIRE( avFile.fpsN == 25 ); 
                REQUIRE( avFile.fpsD == 1 );
            }
        }
    }
}
