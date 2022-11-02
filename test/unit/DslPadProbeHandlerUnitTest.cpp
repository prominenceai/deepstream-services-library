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
#include "DslPadProbeHandler.h"
#include "DslTrackerBintr.h"

using namespace DSL;

SCENARIO( "A new OdePadProbeHandler is created correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new OdePadProbeHandler" ) 
    {
        std::string odeHandlerName("ode-handler");

        WHEN( "A new PadProbeHandler is created" )
        {
            DSL_PPH_ODE_PTR pPadProbeHandler = DSL_PPH_ODE_NEW(odeHandlerName.c_str());

            THEN( "The PadProbeHandler's memebers are setup and returned correctly" )
            {
                REQUIRE( pPadProbeHandler->GetEnabled() == true );
            }
        }
    }
}

SCENARIO( "A new OdePadProbeHandler can Disable and Re-enable", "[PadProbeHandler]" )
{
    GIVEN( "A new OdePadProbeHandler" ) 
    {
        std::string odeHandlerName("ode-handler");

        DSL_PPH_ODE_PTR pPadProbeHandler = DSL_PPH_ODE_NEW(odeHandlerName.c_str());
        REQUIRE( pPadProbeHandler->GetEnabled() == true );

        // Attempting to enable and enabled PadProbeHandler must fail
        REQUIRE( pPadProbeHandler->SetEnabled(true) == false );

        WHEN( "A new OdePadProbeHandler is Disabled'" )
        {
            REQUIRE( pPadProbeHandler->SetEnabled(false) == true );

            // Attempting to disable a disabled PadProbeHandler must fail
            REQUIRE( pPadProbeHandler->SetEnabled(false) == false );

            THEN( "The OdePadProbeHandler can be enabled again" )
            {
                REQUIRE( pPadProbeHandler->SetEnabled(true) == true );
                REQUIRE( pPadProbeHandler->GetEnabled() == true );
            }
        }
    }
}

SCENARIO( "A OdePadProbeHandler can add and remove an OdeTrigger", "[PadProbeHandler]" )
{
    GIVEN( "A new OdePadProbeHandler and OdeTrigger" ) 
    {
        std::string odeHandlerName = "ode-handler";
        std::string odeTriggerName = "first-occurence";
        uint classId(1);
        uint limit(1);

        DSL_PPH_ODE_PTR pPadProbeHandler = 
            DSL_PPH_ODE_NEW(odeHandlerName.c_str());

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pFirstOccurrenceTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), "", classId, limit);

        WHEN( "The Trigger is added to the OdePadProbeHandler" )
        {
            REQUIRE( pPadProbeHandler->AddChild(pFirstOccurrenceTrigger) == true );
            
            // ensure that the Trigger can not be added twice
            REQUIRE( pPadProbeHandler->AddChild(pFirstOccurrenceTrigger) == false );
            
            THEN( "The Trigger can be found and removed" )
            {
                REQUIRE( pPadProbeHandler->IsChild(pFirstOccurrenceTrigger) == true );
                REQUIRE( pFirstOccurrenceTrigger->IsParent(pPadProbeHandler) == true );
                REQUIRE( pFirstOccurrenceTrigger->IsInUse() == true );
                
                REQUIRE( pPadProbeHandler->RemoveChild(pFirstOccurrenceTrigger) == true );
                
                REQUIRE( pPadProbeHandler->IsChild(pFirstOccurrenceTrigger) == false );
                REQUIRE( pFirstOccurrenceTrigger->GetName() == odeTriggerName );
                REQUIRE( pFirstOccurrenceTrigger->IsParent(pPadProbeHandler) == false );
                REQUIRE( pFirstOccurrenceTrigger->IsInUse() == false );
                // ensure removal fails on second call, 
                REQUIRE( pPadProbeHandler->RemoveChild(pFirstOccurrenceTrigger) == false );
            }
        }
    }
}

SCENARIO( "A new MeterPadProbeHandler is created correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new MeterPadProbeHandler" ) 
    {
        std::string meterHandlerName("meter-handler");
        uint interval(1);
        dsl_pph_meter_client_handler_cb clientHandler;

        WHEN( "The PadProbeHandler is created " )
        {
            DSL_PPH_METER_PTR pPadProbeHandler = 
                DSL_PPH_METER_NEW(meterHandlerName.c_str(), interval, clientHandler, NULL);
                
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pPadProbeHandler->GetEnabled() == true );
                REQUIRE( pPadProbeHandler->GetInterval() == interval );
            }
        }
    }
}

SCENARIO( "A new MeterPadProbeHandler can Get/Set attributes correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new MeterPadProbeHandler" ) 
    {
        std::string meterHandlerName("meter-meter");
        uint interval(1);
        dsl_pph_meter_client_handler_cb clientHandler;

        DSL_PPH_METER_PTR pPadProbeHandler = 
            DSL_PPH_METER_NEW(meterHandlerName.c_str(), interval, clientHandler, NULL);

        WHEN( "The MeterPadProbeHandler's enabled setting is disabled " )
        {
            pPadProbeHandler->SetEnabled(false);
            
            THEN( "The correct attribute value is returned" )
            {
                REQUIRE( pPadProbeHandler->GetEnabled() == false );
            }
        }
        WHEN( "The MeterPadProbeHandler's reporting interval is updated " )
        {
            pPadProbeHandler->SetInterval(123);
            
            THEN( "The correct attribute value is returned" )
            {
                REQUIRE( pPadProbeHandler->GetInterval() == 123 );
            }
        }
    }
}

SCENARIO( "A new EosConsumerPadProbeEventHandler is created correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new EosConsumerPadProbeEventHandler" ) 
    {
        std::string handlerName("eos-consumer");

        WHEN( "The PadProbeHandler is created " )
        {
            DSL_PPEH_EOS_CONSUMER_PTR pPadProbeHandler = 
                DSL_PPEH_EOS_CONSUMER_NEW(handlerName.c_str());
                
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pPadProbeHandler->GetName()  == handlerName );
            }
        }
    }
}

SCENARIO( "A new EosHandlerPadProbeEventHandler is created correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new EosHandlerPadProbeEventHandler" ) 
    {
        std::string handlerName("eos-handler");
        dsl_pph_eos_handler_cb clientHandler;

        WHEN( "The PadProbeHandler is created " )
        {
            DSL_PPEH_EOS_HANDLER_PTR pPadProbeHandler = 
                DSL_PPEH_EOS_HANDLER_NEW(handlerName.c_str(), clientHandler, NULL);
                
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pPadProbeHandler->GetName()  == handlerName );
            }
        }
    }
}

SCENARIO( "A new TimestampPadProbeHandler is created correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new TimestampPadProbeHandler" ) 
    {
        std::string handlerName("timestamp-handler");
        struct timeval timestamp{0};
        timestamp.tv_sec = 123;
        timestamp.tv_usec = 456;

        WHEN( "The PadProbeHandler is created " )
        {
            DSL_PPH_TIMESTAMP_PTR pPadProbeHandler = 
                DSL_PPH_TIMESTAMP_NEW(handlerName.c_str());
                
            THEN( "The correct attribute values are returned" )
            {
                pPadProbeHandler->GetTime(timestamp);
                REQUIRE( timestamp.tv_sec == 0 );
                REQUIRE( timestamp.tv_usec == 0 );
            }
        }
    }
}

SCENARIO( "When a TimestampPadProbeHandler timestamp is Set the correct value is returned on Get ", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new TimestampPadProbeHandler" ) 
    {
        std::string handlerName("timestamp-handler");
        struct timeval timestamp{123,345};
        struct timeval retTimestamp{0,0};

        DSL_PPH_TIMESTAMP_PTR pPadProbeHandler = 
            DSL_PPH_TIMESTAMP_NEW(handlerName.c_str());

        WHEN( "The PadProbeHandler is created " )
        {
                
            THEN( "The correct attribute values are returned" )
            {
                pPadProbeHandler->SetTime(timestamp);
                pPadProbeHandler->GetTime(retTimestamp);
                REQUIRE( retTimestamp.tv_sec == timestamp.tv_sec );
                REQUIRE( retTimestamp.tv_usec == timestamp.tv_usec );
            }
        }
    }
}


SCENARIO( "A PadProbeHandler can be added to the Sink Pad of a Bintr", "[PadProbeHandler]" )
{
    GIVEN( "A new Tracker and OdePadProbeHandlr in memory" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        std::string odeHandlerName("ode-handler");

        DSL_PPH_ODE_PTR pPadProbeHandler = DSL_PPH_ODE_NEW(odeHandlerName.c_str());

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        WHEN( "The PadProbeHandler is called to add itself to the Sink Pad of a Bintr" )
        {
            REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SINK) == true );

            THEN( "The PadProbeHandler can remove itself successfully" )
            {
                REQUIRE( pPadProbeHandler->RemoveFromParent(pTrackerBintr, DSL_PAD_SINK) == true );
            }
        }
        WHEN( "The PadProbeHandler is called to add itself to the Sink Pad of a Bintr" )
        {
            REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SINK) == true );

            THEN( "The PadProbeHandler can't be added twice" )
            {
                REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SINK) == false );
            }
        }
        WHEN( "The PadProbeHandler is added and removed from the Sink Pad of a Bintr" )
        {
            REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SINK) == true );
            REQUIRE( pPadProbeHandler->RemoveFromParent(pTrackerBintr, DSL_PAD_SINK) == true );

            THEN( "The PadProbeHandler can't remove itself twice" )
            {
                REQUIRE( pPadProbeHandler->RemoveFromParent(pTrackerBintr, DSL_PAD_SINK) == false );
            }
        }
    }
}

SCENARIO( "A PadProbeHandler can be added to the Source Pad of a Bintr", "[PadProbeHandler]" )
{
    GIVEN( "A new Tracker and OdePadProbeHandlr in memory" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        std::string odeHandlerName("ode-handler");

        DSL_PPH_ODE_PTR pPadProbeHandler = DSL_PPH_ODE_NEW(odeHandlerName.c_str());

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        WHEN( "The PadProbeHandler is called to add itself to the Source Pad of a Bintr" )
        {
            REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SRC) == true );

            THEN( "The PadProbeHandler can remove itself successfully" )
            {
                REQUIRE( pPadProbeHandler->RemoveFromParent(pTrackerBintr, DSL_PAD_SRC) == true );
            }
        }
        WHEN( "The PadProbeHandler is called to add itself to the Source Pad of a Bintr" )
        {
            REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SRC) == true );

            THEN( "The PadProbeHandler can't be added twice" )
            {
                REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SRC) == false );
            }
        }
        WHEN( "The PadProbeHandler is added and removed from the Source Pad of a Bintr" )
        {
            REQUIRE( pPadProbeHandler->AddToParent(pTrackerBintr, DSL_PAD_SRC) == true );
            REQUIRE( pPadProbeHandler->RemoveFromParent(pTrackerBintr, DSL_PAD_SRC) == true );

            THEN( "The PadProbeHandler can't remove itself twice" )
            {
                REQUIRE( pPadProbeHandler->RemoveFromParent(pTrackerBintr, DSL_PAD_SRC) == false );
            }
        }
    }
}


SCENARIO( "Multiple PadProbeHandlers can be added to the Sink Pad of a Bintr", "[PadProbeHandler]" )
{
    GIVEN( "A new Tracker and three new PadProbeHandlers" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        std::string odeHandlerName1("ode-handler-1");
        std::string odeHandlerName2("ode-handler-2");
        std::string odeHandlerName3("ode-handler-3");

        DSL_PPH_ODE_PTR pPadProbeHandler1 = DSL_PPH_ODE_NEW(odeHandlerName1.c_str());
        DSL_PPH_ODE_PTR pPadProbeHandler2 = DSL_PPH_ODE_NEW(odeHandlerName2.c_str());
        DSL_PPH_ODE_PTR pPadProbeHandler3 = DSL_PPH_ODE_NEW(odeHandlerName3.c_str());

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        WHEN( "Multiple PadProbeHandlers are added to the Sink " )
        {
            REQUIRE( pPadProbeHandler1->AddToParent(pTrackerBintr, DSL_PAD_SINK) == true );
            REQUIRE( pPadProbeHandler2->AddToParent(pTrackerBintr, DSL_PAD_SINK) == true );
            REQUIRE( pPadProbeHandler3->AddToParent(pTrackerBintr, DSL_PAD_SINK) == true );
            
            THEN( "A second Sink Pad Batch Meta Handler can be added " )
            {
                REQUIRE( pPadProbeHandler1->RemoveFromParent(pTrackerBintr, DSL_PAD_SINK) == true );
                REQUIRE( pPadProbeHandler3->RemoveFromParent(pTrackerBintr, DSL_PAD_SINK) == true );
                REQUIRE( pPadProbeHandler2->RemoveFromParent(pTrackerBintr, DSL_PAD_SINK) == true );
            }
        } 
    }
}

SCENARIO( "Multiple PadProbeHandlers can be added to the Source Pad of a Bintr", "[PadProbeHandler]" )
{
    GIVEN( "A new Tracker and three new PadProbeHandlers" ) 
    {
        std::string trackerName("ktl-tracker");
        uint initWidth(200);
        uint initHeight(100);
        
        std::string odeHandlerName1("ode-handler-1");
        std::string odeHandlerName2("ode-handler-2");
        std::string odeHandlerName3("ode-handler-3");

        DSL_PPH_ODE_PTR pPadProbeHandler1 = DSL_PPH_ODE_NEW(odeHandlerName1.c_str());
        DSL_PPH_ODE_PTR pPadProbeHandler2 = DSL_PPH_ODE_NEW(odeHandlerName2.c_str());
        DSL_PPH_ODE_PTR pPadProbeHandler3 = DSL_PPH_ODE_NEW(odeHandlerName3.c_str());

        DSL_KTL_TRACKER_PTR pTrackerBintr = 
            DSL_KTL_TRACKER_NEW(trackerName.c_str(), initWidth, initHeight);

        WHEN( "Multiple PadProbeHandlers are added to the Sink " )
        {
            REQUIRE( pPadProbeHandler1->AddToParent(pTrackerBintr, DSL_PAD_SRC) == true );
            REQUIRE( pPadProbeHandler2->AddToParent(pTrackerBintr, DSL_PAD_SRC) == true );
            REQUIRE( pPadProbeHandler3->AddToParent(pTrackerBintr, DSL_PAD_SRC) == true );
            
            THEN( "A second Sink Pad Batch Meta Handler can be added " )
            {
                REQUIRE( pPadProbeHandler1->RemoveFromParent(pTrackerBintr, DSL_PAD_SRC) == true );
                REQUIRE( pPadProbeHandler3->RemoveFromParent(pTrackerBintr, DSL_PAD_SRC) == true );
                REQUIRE( pPadProbeHandler2->RemoveFromParent(pTrackerBintr, DSL_PAD_SRC) == true );
            }
        } 
    }
}

SCENARIO( "A new FrameNumberAdderPadProbeHandler is created correctly", "[PadProbeHandler]" )
{
    GIVEN( "Attributes for a new FrameNumberAdderPadProbeHandler" ) 
    {
        std::string adderHandlerName("adder-handler");

        WHEN( "The PadProbeHandler is created " )
        {
            DSL_PPEH_FRAME_NUMBER_ADDER_PTR pPadProbeHandler = 
                DSL_PPEH_FRAME_NUMBER_ADDER_NEW(adderHandlerName.c_str());
                
            THEN( "The correct attribute values are returned" )
            {
                REQUIRE( pPadProbeHandler->GetFrameNumber() == 0 );
            }
        }
    }
}
