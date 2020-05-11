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
#include "DslDetectionEvent.h"

using namespace DSL;

SCENARIO( "A new FirstOccurrenceEvent is created correctly", "[DetectionEvent]" )
{
    GIVEN( "Attributes for a new DetectionEvent" ) 
    {
        std::string eventName = "first-occurence";
        uint classId(1);

        WHEN( "A new DetectionEvent is created" )
        {
            DSL_EVENT_FIRST_OCCURRENCE_PTR pFirstOccurrenceEvent = 
                DSL_EVENT_FIRST_OCCURRENCE_NEW(eventName.c_str(), classId);

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                REQUIRE( pFirstOccurrenceEvent->GetClassId() == classId );
                uint minWidth(123), minHeight(123);
                pFirstOccurrenceEvent->GetMinDimensions(&minWidth, &minHeight);
                REQUIRE( minWidth == 0 );
                REQUIRE( minHeight == 0 );
                uint minFrameCountN(123), minFrameCountD(123);
                pFirstOccurrenceEvent->GetMinFrameCount(&minFrameCountN, &minFrameCountD);
                REQUIRE( minFrameCountN == 0 );
                REQUIRE( minFrameCountD == 0 );
            }
        }
    }
}
