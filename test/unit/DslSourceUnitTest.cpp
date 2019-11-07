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
#include "DslSourceBintr.h"

using namespace DSL;

SCENARIO( "A new base SourceBintr is created correctly",  "[SourceBintr]" )
{
    GIVEN( "A name for a new SourceBintr" ) 
    {
        std::string sourceName = "test-source";

        WHEN( "The SourceBintr is created " )
        {
            DSL_SOURCE_PTR pSourceBintr = DSL_SOURCE_NEW(sourceName.c_str());
            
            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->m_gpuId == 0 );
                REQUIRE( pSourceBintr->m_nvbufMemoryType == 0 );
                REQUIRE( pSourceBintr->m_pGstObj != NULL );
                REQUIRE( pSourceBintr->GetSensorId() == -1 );
                REQUIRE( pSourceBintr->IsInUse() == false );
            }
        }
    }
}

SCENARIO( "A new CsiSourceBintr is created correctly",  "[CsiSourceBintr]" )
{
    GIVEN( "A name for a new CsiSourceBintr" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(1);
        uint fps_d(30);
        std::string sourceName = "test-csi-source";

        WHEN( "The SourceBintr is created " )
        {
        
            DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
                sourceName.c_str(), width, height, fps_n, fps_d);

            THEN( "All memeber variables are initialized correctly" )
            {
                REQUIRE( pSourceBintr->IsLive() == true );
                REQUIRE( pSourceBintr->m_width == width );
                REQUIRE( pSourceBintr->m_height == height );
                REQUIRE( pSourceBintr->m_fps_n == fps_n );
                REQUIRE( pSourceBintr->m_fps_d == fps_d );
            }
        }
    }
}

SCENARIO( "Set Sensor Id updates SourceBintr correctly",  "[CsiSourceBintr]" )
{
    GIVEN( "A new CsiSourceBintr in memory" ) 
    {
        uint width(1280);
        uint height(720);
        uint fps_n(1);
        uint fps_d(30);
        std::string sourceName = "test-csi-source";
        int sensorId = 1;

        DSL_CSI_SOURCE_PTR pSourceBintr = DSL_CSI_SOURCE_NEW(
            sourceName.c_str(), width, height, fps_n, fps_d);

        WHEN( "The Sensor Id is set " )
        {
            pSourceBintr->SetSensorId(sensorId);

            THEN( "The returned Sensor Id is correct" )
            {
                REQUIRE( pSourceBintr->GetSensorId() == sensorId );
            }
        }
    }
}
