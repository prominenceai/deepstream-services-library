/*
The MIT License

Copyright (c) 2024, Prominence AI, Inc.

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
#include "Dsl.h"
#include "DslApi.h"
#include "DslTrackerBintr.h"

using namespace DSL;

// Filespec for the NvDCF Tracker config file
static const std::string dcfTrackerConfigFile(
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml");

static const std::string iouTrackerConfigFile("/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.ym");

SCENARIO( "A Component's Queue is created correctly", "[QBintr]" )
{
    GIVEN( "Attributes for a new IOU Tracker as sample component" ) 
    {
        std::string trackerName("iou-tracker");
        uint width(200);
        uint height(100);

        WHEN( "The IOU Tracker is created" )
        {
            DSL_TRACKER_PTR pTrackerBintr = 
                DSL_TRACKER_NEW(trackerName.c_str(), iouTrackerConfigFile.c_str(), width, height);

            THEN( "The Tracker's lib is found, loaded, and returned correctly")
            {
                REQUIRE( pTrackerBintr->GetQueueLeaky() == DSL_COMPONENT_QUEUE_LEAKY_NO );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 200 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 10485760 );
                REQUIRE( pTrackerBintr->GetQueueMaxSize(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 1000000000 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BUFFERS) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_BYTES) == 0 );
                REQUIRE( pTrackerBintr->GetQueueMinThreshold(
                        DSL_COMPONENT_QUEUE_UNIT_OF_TIME) == 0 );
            }
        }
    }
}

static void queue_overrun_listener_cb1(const wchar_t* name, void* client_data)
{
    std::wcout << L"Overrun listener-1 called for Component '" 
        << name << L"'" << std::endl;
}
static void queue_overrun_listener_cb2(const wchar_t* name, void* client_data)
{
    std::wcout << L"Overrun listener-2 called for Component '" 
        << name << L"'" << std::endl;
}
static void queue_underrun_listener_cb1(const wchar_t* name, void* client_data)
{
    std::wcout << L"Underrun listener-1 called for Component '" 
        << name << L"'" << std::endl;
}
static void queue_underrun_listener_cb2(const wchar_t* name, void* client_data)
{
    std::wcout << L"Underrun listener-2 called for Component '" 
        << name << L"'" << std::endl;
}
SCENARIO( "Queue overrun and underrun events are handled correctly", "[QBintr]" )
{
    GIVEN( "Attributes for a new IOU Tracker as sample component" ) 
    {
        std::string trackerName1("iou-tracker-1");
        std::string trackerName2("iou-tracker-2");
        uint width(200);
        uint height(100);

        WHEN( "The IOU Tracker is created" )
        {
            DSL_TRACKER_PTR pTrackerBintr1 = 
                DSL_TRACKER_NEW(trackerName1.c_str(), iouTrackerConfigFile.c_str(), width, height);

            DSL_TRACKER_PTR pTrackerBintr2 = 
                DSL_TRACKER_NEW(trackerName2.c_str(), iouTrackerConfigFile.c_str(), width, height);

            REQUIRE( pTrackerBintr1->AddQueueOverrunListener(
                queue_overrun_listener_cb1, (void*)0x12345678) == true );

            REQUIRE( pTrackerBintr2->AddQueueOverrunListener(
                queue_overrun_listener_cb1, (void*)0x87654321) == true );

            REQUIRE( pTrackerBintr1->AddQueueOverrunListener(
                queue_overrun_listener_cb2, (void*)0x12345678) == true );

            REQUIRE( pTrackerBintr2->AddQueueOverrunListener(
                queue_overrun_listener_cb2, (void*)0x87654321) == true );

            REQUIRE( pTrackerBintr1->AddQueueUnderrunListener(
                queue_underrun_listener_cb1, (void*)0x12345678) == true );

            REQUIRE( pTrackerBintr2->AddQueueUnderrunListener(
                queue_underrun_listener_cb1, (void*)0x87654321) == true );

            REQUIRE( pTrackerBintr1->AddQueueUnderrunListener(
                queue_underrun_listener_cb2, (void*)0x12345678) == true );

            REQUIRE( pTrackerBintr2->AddQueueUnderrunListener(
                queue_underrun_listener_cb2, (void*)0x87654321) == true );

            THEN( "The Tracker's lib is found, loaded, and returned correctly")
            {
                // Note: requires manual verification using callback cout calls.
                QueueOverrunCB(NULL, (void*)&(*pTrackerBintr1));
                QueueOverrunCB(NULL, (void*)&(*pTrackerBintr2));
                QueueUnderrunCB(NULL, (void*)&(*pTrackerBintr1));
                QueueUnderrunCB(NULL, (void*)&(*pTrackerBintr2));
            }
        }
    }
}
