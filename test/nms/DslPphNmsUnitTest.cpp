/*
The MIT License

Copyright (c) 2022, Prominence AI, Inc.

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
#include "DslServices.h"
#include "DslPadProbeHandlerNms.h"

static std::string name("nms-pph");

static std::string labelFile1(
    "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/labels.txt");

static std::string labelFile2(
    "/opt/nvidia/deepstream/deepstream/samples/models/Secondary_CarColor/labels.txt");

using namespace DSL;

SCENARIO( "A new Non Maximum Suppression PPH is created correctly", 
    "[NmsPph]" )
{
    GIVEN( "Attributes for a new NMS PPH" )
    {
        uint matchMethod(DSL_NMS_MATCH_METHOD_IOS);
        float matchThreshold(0.5);

        WHEN( "When the NMS PPH is created with a lableFile - 1 label per line" )
        {
            DSL_PPH_NMS_PTR pNmsPph = DSL_PPH_NMS_NEW(name.c_str(), 
                labelFile1.c_str(), matchMethod, matchThreshold);

            THEN( "All members are setup correctly" )
            {
                std::string retLabelFile(pNmsPph->GetLabelFile());
                REQUIRE( retLabelFile == labelFile1 );
                
                // 4 labels in "Primary_Detector/labels.txt"
                REQUIRE( pNmsPph->GetNumLabels() == 4 );
                
                uint retMatchMethod(9);
                float retMatchThreshold(0);
                pNmsPph->GetMatchSettings(&retMatchMethod, &retMatchThreshold);
                REQUIRE( retMatchMethod == matchMethod );
                REQUIRE( retMatchThreshold == matchThreshold );
            }
        }
        WHEN( "When the NMS PPH is created with a lableFile - all labels on 1 line" )
        {
            DSL_PPH_NMS_PTR pNmsPph = DSL_PPH_NMS_NEW(name.c_str(), 
                labelFile2.c_str(), matchMethod, matchThreshold);

            THEN( "All members are setup correctly" )
            {
                std::string retLabelFile(pNmsPph->GetLabelFile());
                REQUIRE( retLabelFile == labelFile2 );
                
                // 12 labels in "Secondary_CarColor/labels.txt"
                REQUIRE( pNmsPph->GetNumLabels() == 12 );
            }
        }
        WHEN( "When the NMS PPH is created without a lableFile" )
        {
            std::string noLabelFile;
            DSL_PPH_NMS_PTR pNmsPph = DSL_PPH_NMS_NEW(name.c_str(), 
                noLabelFile.c_str(), matchMethod, matchThreshold);

            THEN( "All members are setup correctly" )
            {
                std::string retLabelFile(pNmsPph->GetLabelFile());
                REQUIRE( retLabelFile == noLabelFile );
                
                // one label for class agnostic nms"
                REQUIRE( pNmsPph->GetNumLabels() == 1 );
            }
        }
    }
}