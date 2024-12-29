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
#include "DslSdeTrigger.h"
#include "DslSdeAction.h"

using namespace DSL;

SCENARIO( "A new PrintSdeAction is created correctly", "[SdeAction]" )
{
    GIVEN( "Attributes for a new PrintSdeAction" ) 
    {
        std::string actionName("sde-action");

        WHEN( "A new SdeAction is created" )
        {
            DSL_SDE_ACTION_PRINT_PTR pAction = 
                DSL_SDE_ACTION_PRINT_NEW(actionName.c_str(), false);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A PrintSdeAction handles an SDE Occurence correctly", "[SdeAction]" )
{
    GIVEN( "A new PrintSdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "sde-action";

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_SDE_ACTION_PRINT_PTR pAction = 
            DSL_SDE_ACTION_PRINT_NEW(actionName.c_str(), false);

        WHEN( "A new SDE is created" )
        {
            NvDsAudioFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = 1615768434973357000;
            frameMeta.source_id = 2;
            frameMeta.class_id = classId; // must match Trigger's classId
            
            THEN( "The SdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, 
                    &frameMeta);
            }
        }
    }
}

static void sde_occurrence_monitor_cb(dsl_sde_occurrence_info* pInfo, 
    void* client_data)
{
    std::wcout << "Trigger Name        : " << pInfo->trigger_name << "\n";
    std::cout << "  Unique Id         : " << pInfo->unique_sde_id << "\n";
    std::cout << "  NTP Timestamp     : " << pInfo->ntp_timestamp << "\n";
    std::cout << "  Source Data       : ------------------------" << "\n";
    std::cout << "    Id              : " << pInfo->source_info.source_id << "\n";
    std::cout << "    Batch Id        : " << pInfo->source_info.batch_id << "\n";
    std::cout << "    Pad Index       : " << pInfo->source_info.pad_index << "\n";
    std::cout << "    Frame           : " << pInfo->source_info.frame_num << "\n";
    std::cout << "    Sample Rate     : " << pInfo->source_info.sample_rate << "\n";
    std::cout << "    Samples/Frame   : " << pInfo->source_info.num_samples_per_frame << "\n";
    std::cout << "    Channels        : " << pInfo->source_info.num_channels << "\n";

    std::cout << "  Sound Data        : ------------------------" << "\n";
    std::cout << "    Class Id        : " << pInfo->sound_info.class_id << "\n";
    std::cout << "    Infer Comp Id   : " << pInfo->sound_info.inference_component_id << "\n";
    std::cout << "    Label           : " << pInfo->sound_info.label << "\n";
    std::cout << "    Infer Conf      : " << pInfo->sound_info.inference_confidence << "\n";
    std::cout << "  Trigger Criteria  : ------------------------" << "\n";
    std::cout << "    Source Id       : " << pInfo->criteria_info.source_id << "\n";
    std::cout << "    Class Id        : " << pInfo->criteria_info.class_id << "\n";
    std::cout << "    Min Infer Conf  : " << pInfo->criteria_info.min_inference_confidence << "\n";
    std::cout << "    Max Infer Conf  : " << pInfo->criteria_info.max_inference_confidence << "\n";
    std::cout << "    Interval        : " << pInfo->criteria_info.interval << "\n";
}

SCENARIO( "A new MonitorSdeAction is created correctly", "[SdeAction]" )
{
    GIVEN( "Attributes for a new MonitorSdeAction" ) 
    {
        std::string actionName("sde-action");

        WHEN( "A new SdeAction is created" )
        {
            DSL_SDE_ACTION_MONITOR_PTR pAction = 
                DSL_SDE_ACTION_MONITOR_NEW(actionName.c_str(), 
                    sde_occurrence_monitor_cb, NULL);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A MonitorSdeAction handles an SDE Occurence correctly", "[SdeAction]" )
{
    GIVEN( "A new MonitorSdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "sde-action";

        DSL_SDE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_SDE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), 
                source.c_str(), classId, limit);

        DSL_SDE_ACTION_MONITOR_PTR pAction = 
            DSL_SDE_ACTION_MONITOR_NEW(actionName.c_str(), 
                sde_occurrence_monitor_cb, NULL);

        WHEN( "A new SDE is created" )
        {
            NvDsAudioFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = 1615768434973357000;
            frameMeta.source_id = 2;
            frameMeta.class_id = classId; // must match Trigger's classId
            
            THEN( "The SdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, 
                    &frameMeta);
            }
        }
    }
}

