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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"

using namespace DSL;

static boolean ode_occurrence_handler_cb(uint64_t event_id, const wchar_t* name,
    void* frame_meta, void* object_meta, void* client_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta*)frame_meta;
    std::cout << "Event Name      : " << cstrName << "\n";
    std::cout << "  Unique Id     : " << event_id << "\n";
    std::cout << "  NTP Timestamp : " << pFrameMeta->ntp_timestamp << "\n";
    std::cout << "  Source Data   : ------------------------" << "\n";
    std::cout << "    Id          : " << pFrameMeta->source_id << "\n";
    std::cout << "    Frame       : " << pFrameMeta->frame_num << "\n";
    std::cout << "    Width       : " << pFrameMeta->source_frame_width << "\n";
    std::cout << "    Heigh       : " << pFrameMeta->source_frame_height << "\n";

    NvDsObjectMeta* pObjectMeta = (NvDsObjectMeta*)object_meta;
    if (pObjectMeta)
    {
        std::cout << "  Object Data   : ------------------------" << "\n";
        std::cout << "    Class Id    : " << pObjectMeta->class_id << "\n";
        std::cout << "    Tracking Id : " << pObjectMeta->object_id << "\n";
        std::cout << "    Label       : " << pObjectMeta->obj_label << "\n";
        std::cout << "    Confidence  : " << pObjectMeta->confidence << "\n";
        std::cout << "    Left        : " << pObjectMeta->rect_params.left << "\n";
        std::cout << "    Top         : " << pObjectMeta->rect_params.top << "\n";
        std::cout << "    Width       : " << pObjectMeta->rect_params.width << "\n";
        std::cout << "    Height      : " << pObjectMeta->rect_params.height << "\n";
    }
}    

SCENARIO( "A new CallbackOdeAction is created correctly", "[CallbackOdeAction]" )
{
    GIVEN( "Attributes for a new CallbackOdeAction" ) 
    {
        std::string odeActionName("ode-action");

        WHEN( "A new CallbackOdeAction is created" )
        {
            DSL_ODE_ACTION_CALLBACK_PTR pOdeAction = 
                DSL_ODE_ACTION_CALLBACK_NEW(odeActionName.c_str(), ode_occurrence_handler_cb, NULL);

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( odeActionName == retName );
            }
        }
    }
}

SCENARIO( "A CallbackOdeAction handles an Event Occurence correctly", "[CallbackOdeAction]" )
{
    GIVEN( "A new CallbackOdeAction" ) 
    {
        std::string odeTypeName("first-occurence");
        uint classId(1);
        uint limit(1);

        std::string odeActionName("ode-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTypeName.c_str(), classId, limit);

        DSL_ODE_ACTION_CALLBACK_PTR pOdeAction = 
            DSL_ODE_ACTION_CALLBACK_NEW(odeActionName.c_str(), ode_occurrence_handler_cb, NULL);

        WHEN( "A new Event is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Event's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new CaptureOdeAction is created correctly", "[CaptureOdeAction]" )
{
    GIVEN( "Attributes for a new CaptureOdeAction" ) 
    {
        std::string odeActionName("ode-action");
        std::string outdir("./");

        WHEN( "A new CaptureOdeAction is created" )
        {
            DSL_ODE_ACTION_CAPTURE_PTR pOdeAction = 
                DSL_ODE_ACTION_CAPTURE_NEW(odeActionName.c_str(), DSL_CAPTURE_TYPE_OBJECT, outdir.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( odeActionName == retName );
            }
        }
    }
}

SCENARIO( "A new LogOdeAction is created correctly", "[LogOdeAction]" )
{
    GIVEN( "Attributes for a new LogOdeAction" ) 
    {
        std::string odeActionName("ode-action");

        WHEN( "A new EventAction is created" )
        {
            DSL_ODE_ACTION_LOG_PTR pOdeAction = 
                DSL_ODE_ACTION_LOG_NEW(odeActionName.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( odeActionName == retName );
            }
        }
    }
}

SCENARIO( "A LogOdeAction handles an Event Occurence correctly", "[LogOdeAction]" )
{
    GIVEN( "A new LogOdeAction" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);
        uint limit(1);
        
        std::string odeActionName = "ode-action";

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(eventName.c_str(), classId, limit);

        DSL_ODE_ACTION_LOG_PTR pOdeAction = 
            DSL_ODE_ACTION_LOG_NEW(odeActionName.c_str());

        WHEN( "A new Event is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Event's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new PauseOdeAction is created correctly", "[PauseOdeAction]" )
{
    GIVEN( "Attributes for a new PrintOdeAction" ) 
    {
        std::string odeActionName("ode-action");
        std::string pipelineName("pipeline");

        WHEN( "A new PauseOdeAction is created" )
        {
            DSL_ODE_ACTION_PAUSE_PTR pOdeAction = 
                DSL_ODE_ACTION_PAUSE_NEW(odeActionName.c_str(), pipelineName.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( odeActionName == retName );
            }
        }
    }
}

SCENARIO( "A PauseOdeAction handles an Event Occurence correctly", "[PauseOdeAction]" )
{
    GIVEN( "A new PauseOdeAction" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);
        uint limit(1);
        
        std::string odeActionName = "ode-action";
        std::string pipelineName("pipeline");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(eventName.c_str(), classId, limit);

        DSL_ODE_ACTION_PAUSE_PTR pOdeAction = 
            DSL_ODE_ACTION_PAUSE_NEW(odeActionName.c_str(), pipelineName.c_str());

        WHEN( "A new Event is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Event's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                // NOTE:: Pipeline pause will produce an error message as it does not exist
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new PrintOdeAction is created correctly", "[PrintOdeAction]" )
{
    GIVEN( "Attributes for a new PrintOdeAction" ) 
    {
        std::string odeActionName("ode-action");

        WHEN( "A new EventAction is created" )
        {
            DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
                DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( odeActionName == retName );
            }
        }
    }
}

SCENARIO( "A PrintOdeAction handles an Event Occurence correctly", "[PrintOdeAction]" )
{
    GIVEN( "A new PrintOdeAction" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);
        uint limit(1);
        
        std::string odeActionName = "ode-action";

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(eventName.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pOdeAction = 
            DSL_ODE_ACTION_PRINT_NEW(odeActionName.c_str());

        WHEN( "A new Event is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Event's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new RedactOdeAction is created correctly", "[RedactOdeAction]" )
{
    GIVEN( "Attributes for a new RedactOdeAction" ) 
    {
        std::string odeActionName("ode-action");
        float red(1), green(1), blue(1), alpha(1);

        WHEN( "A new RedactOdeAction is created" )
        {
            DSL_ODE_ACTION_REDACT_PTR pOdeAction = 
                DSL_ODE_ACTION_REDACT_NEW(odeActionName.c_str(), red, green, blue, alpha);

            THEN( "The Events's memebers are setup and returned correctly" )
            {
                std::string retName = pOdeAction->GetCStrName();
                REQUIRE( odeActionName == retName );
            }
        }
    }
}

SCENARIO( "A RedactOdeAction handles an Event Occurence correctly", "[RedactOdeAction]" )
{
    GIVEN( "A new RedactOdeAction" ) 
    {
        std::string eventName("first-occurence");
        uint classId(1);
        uint limit(1);
        
        std::string odeActionName = "ode-action";
        float red(1), green(1), blue(1), alpha(1);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pFirstOccurrenceEvent = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(eventName.c_str(), classId, limit);

        DSL_ODE_ACTION_REDACT_PTR pOdeAction = 
            DSL_ODE_ACTION_REDACT_NEW(odeActionName.c_str(), red, green, blue, alpha);

        WHEN( "A new Event is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Event's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            objectMeta.rect_params.border_width = 9;
            objectMeta.rect_params.has_bg_color = 0;
            objectMeta.rect_params.bg_color.red = 0;
            objectMeta.rect_params.bg_color.green = 0;
            objectMeta.rect_params.bg_color.blue = 0;
            objectMeta.rect_params.bg_color.alpha = 0;
            
            THEN( "The EventAction can Handle the Occurrence" )
            {
                pOdeAction->HandleOccurrence(pFirstOccurrenceEvent, NULL, &frameMeta, &objectMeta);
                REQUIRE( objectMeta.rect_params.border_width == 0 );
                REQUIRE( objectMeta.rect_params.has_bg_color == 1 );
                REQUIRE( objectMeta.rect_params.bg_color.red == red );
                REQUIRE( objectMeta.rect_params.bg_color.green == green );
                REQUIRE( objectMeta.rect_params.bg_color.blue == blue );
                REQUIRE( objectMeta.rect_params.bg_color.alpha == alpha );
            }
        }
    }
}
