/*
The MIT License

Copyright (c) 2019-2022, Prominence AI, Inc.

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
#include "DslOdeAccumulator.h"
#include "DslOdeAction.h"
#include "DslServices.h"

using namespace DSL;

static std::vector<NvDsDisplayMeta*> displayMetaData;

static void ode_occurrence_handler_cb_1(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
{
    std::cout << "Custom Action callback 1. called\n";
}    

static void ode_occurrence_handler_cb_2(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
{
    std::cout << "Custom Action callback 2. called\n";
}    
static void ode_occurrence_handler_cb_3(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
{
    std::cout << "Custom Action callback 3. called\n";
}    

SCENARIO( "A new OdeAccumulator is created correctly", "[OdeAccumulator]" )
{
    GIVEN( "Attributes for a new OdeAccumulator" ) 
    {
        std::string odeAccumulatorName("accumulator");

        WHEN( "A new OdeAccumulator is created" )
        {
            DSL_ODE_ACCUMULATOR_PTR pOdeAccumlator = 
                DSL_ODE_ACCUMULATOR_NEW(odeAccumulatorName.c_str());

            THEN( "The OdeAccumulator's memebers are setup and returned correctly" )
            {
                REQUIRE( pOdeAccumlator->GetName() == odeAccumulatorName );
            }
        }
    }
}

SCENARIO( "An OdeAccumulator executes its ODE Actions in the correct order ", "[OdeAccumulator]" )
{
    GIVEN( "A new OdeAccumulator and three print actions" ) 
    {
        std::string odeAccumulatorName("accumulator");

        std::string odeTriggerName("occurence");
        uint classId(1);
        uint limit(0); // not limit

        std::string source;

        // The unindexed Child map is order alpha-numerically
        std::string odeActionName1("1-action");
        std::string odeActionName2("2-action");
        std::string odeActionName3("3-action");
        
        DSL_ODE_ACCUMULATOR_PTR pOdeAccumlator = 
            DSL_ODE_ACCUMULATOR_NEW(odeAccumulatorName.c_str());

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pOdeTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        // Three custom actions using the calbacks defined above. 
        DSL_ODE_ACTION_CUSTOM_PTR pOdeAction1 = 
            DSL_ODE_ACTION_CUSTOM_NEW(odeActionName1.c_str(), ode_occurrence_handler_cb_1, NULL);
        DSL_ODE_ACTION_CUSTOM_PTR pOdeAction2 = 
            DSL_ODE_ACTION_CUSTOM_NEW(odeActionName2.c_str(), ode_occurrence_handler_cb_2, NULL);
        DSL_ODE_ACTION_CUSTOM_PTR pOdeAction3 = 
            DSL_ODE_ACTION_CUSTOM_NEW(odeActionName3.c_str(), ode_occurrence_handler_cb_3, NULL);

        // Frame Meta test data
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 1;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        // Object Meta test data
        
        WHEN( "The Three actions are added in a specific order" )
        {
            // The indexed Child map is ordered by add-order - used for execution.
            REQUIRE( pOdeAccumlator->AddAction(pOdeAction3) == true );        
            REQUIRE( pOdeAccumlator->AddAction(pOdeAction1) == true );        
            REQUIRE( pOdeAccumlator->AddAction(pOdeAction2) == true );        
            
            THEN( "The actions are executed in the correct order" )
            {
                // Note: this requires manual/visual confirmation at this time.
                pOdeAccumlator->HandleOccurrences(pOdeTrigger, 
                    NULL, displayMetaData, &frameMeta);
                
                // Remove Action 3 and add back in to change order    
                REQUIRE( pOdeAccumlator->RemoveAction(pOdeAction3) == true );        
                REQUIRE( pOdeTrigger->AddAction(pOdeAction3) == true );        
                
                pOdeAccumlator->HandleOccurrences(pOdeTrigger, 
                    NULL, displayMetaData, &frameMeta);
            }
        }
    }
}
