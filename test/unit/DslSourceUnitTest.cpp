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

SCENARIO( "Set Sensor Id updates Source correctly",  "[source]" )
{
    GIVEN( "A new Source in memory" ) 
    {
        std::string sourceName = "csi-source";
        int sensorId = 1;

        std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr = 
            std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
            sourceName.c_str(), 1280, 720, 30, 1));
            
        // ensure source id reflects not is use
        REQUIRE( pSourceBintr->GetSensorId() == -1 );
        REQUIRE( pSourceBintr->IsInUse() == false );

        WHEN( "The Sensor Id is set " )
        {
            pSourceBintr->SetSensorId(sensorId);
            THEN( "The returned Sensor Id is correct")
            {
                REQUIRE( pSourceBintr->GetSensorId() == sensorId );
                
            }
        }
    }
}

SCENARIO( "CSI Source returns the correct play-type on new",  "[source]" )
{
    std::string sourceName  = "csi-source";

    GIVEN( "No setup requirements" ) 
    {

        WHEN( "A new CSI Camera Source is created" ) 
        {

            std::shared_ptr<DSL::CsiSourceBintr> pSourceBintr = 
                std::shared_ptr<DSL::CsiSourceBintr>(new DSL::CsiSourceBintr(
                sourceName.c_str(), 1280, 720, 30, 1));
            
            THEN( "The Source is correctly created as live")
            {
                REQUIRE( pSourceBintr->IsLive() == true );
            }
        }
    }
}