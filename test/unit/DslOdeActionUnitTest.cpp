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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"
#include "DslDisplayTypes.h"
#include "DslMailer.h"

using namespace DSL;

static void ode_occurrence_handler_cb(uint64_t event_id, const wchar_t* name,
    void* buffer, void* frame_meta, void* object_meta, void* client_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    NvDsFrameMeta* pFrameMeta = (NvDsFrameMeta*)frame_meta;
    std::cout << "Trigger Name    : " << cstrName << "\n";
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

SCENARIO( "A new CustomOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new CustomOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new CustomOdeAction is created" )
        {
            DSL_ODE_ACTION_CUSTOM_PTR pAction = 
                DSL_ODE_ACTION_CUSTOM_NEW(actionName.c_str(), ode_occurrence_handler_cb, NULL);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A CustomOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new CustomOdeAction" ) 
    {
        std::string odeTriggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string actionName("ode-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_CUSTOM_PTR pAction = 
            DSL_ODE_ACTION_CUSTOM_NEW(actionName.c_str(), ode_occurrence_handler_cb, NULL);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new CaptureFrameOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new CaptureFrameOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string outdir("./");
        bool annotate(true);

        WHEN( "A new CaptureFrameOdeAction is created" )
        {
            DSL_ODE_ACTION_CAPTURE_FRAME_PTR pAction = 
                DSL_ODE_ACTION_CAPTURE_FRAME_NEW(actionName.c_str(), outdir.c_str(), annotate);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A new CaptureOjbectOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new CaptureObjectOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string outdir("./");

        WHEN( "A new CaptureObjectOdeAction is created" )
        {
            DSL_ODE_ACTION_CAPTURE_OBJECT_PTR pAction = 
                DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), outdir.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

static void capture_complete_listener_cb1(dsl_capture_info* info, void* user_data)
{
    std::cout << "Capture complete lister 1 called \n";
    *(int*)user_data = 111;
}

static void capture_complete_listener_cb2(dsl_capture_info* info, void* user_data)
{
    std::cout << "Capture complete lister 2 called \n";
    *(int*)user_data = 222;
}

SCENARIO( "An CaptureOdeAction can add and remove Capture Complete Listeners",  "[OdeAction]" )
{
    GIVEN( "A new CaptureObjectOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string outdir("./");

        DSL_ODE_ACTION_CAPTURE_OBJECT_PTR pAction = 
            DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), outdir.c_str());
        
        WHEN( "Client Listeners are added" )
        {
            REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb1, NULL) == true );
            REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb2, NULL) == true );

            THEN( "Adding them a second time must fail" )
            {
                REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb1, NULL) == false );
                REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb2, NULL) == false );
            }
        }
        WHEN( "Client Listeners are added" )
        {
            REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb1, NULL) == true );
            REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb2, NULL) == true );

            THEN( "They can be successfully removed" )
            {
                REQUIRE( pAction->RemoveCaptureCompleteListener(capture_complete_listener_cb1) == true );
                REQUIRE( pAction->RemoveCaptureCompleteListener(capture_complete_listener_cb2) == true );
                
                // Calling a second time must fail
                REQUIRE( pAction->RemoveCaptureCompleteListener(capture_complete_listener_cb1) == false );
                REQUIRE( pAction->RemoveCaptureCompleteListener(capture_complete_listener_cb2) == false );
            }
        }
    }
}

SCENARIO( "An CaptureOdeAction calls all Listeners on Capture Complete", "[OdeAction]" )
{
    GIVEN( "A new CaptureObjectOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string outdir("./");
        uint userData1(0), userData2(0);
        uint width(1280), height(720);

        DSL_ODE_ACTION_CAPTURE_OBJECT_PTR pAction = 
            DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), outdir.c_str());
        
        REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb1, &userData1) == true );
        REQUIRE( pAction->AddCaptureCompleteListener(capture_complete_listener_cb2, &userData2) == true );
        
        WHEN( "When capture info is queued" )
        {
            std::shared_ptr<cv::Mat> pImageMate = 
                std::shared_ptr<cv::Mat>(new cv::Mat(cv::Size(width, height), CV_8UC3));

            pAction->QueueCapturedImage(pImageMate);
            
            THEN( "All client listeners are called on capture complete" )
            {
                // simulate timer callback
                REQUIRE( pAction->CompleteCapture() == FALSE );
                // Callbacks will change user data if called
                REQUIRE( userData1 == 111 );
                REQUIRE( userData2 == 222 );
            }
        }
    }
}

SCENARIO( "A new EmailOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new EmailOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string subject("email subject");
        std::string mailerName("mailer");

        DSL_MAILER_PTR pMailer = DSL_MAILER_NEW(mailerName.c_str());

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_EMAIL_PTR pAction = 
                DSL_ODE_ACTION_EMAIL_NEW(actionName.c_str(), 
                    pMailer, subject.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A EmailOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new EmailOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("ode-action");
        std::string subject("email subject");
        
        std::string mailerName("mailer");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_MAILER_PTR pMailer = DSL_MAILER_NEW(mailerName.c_str());
        
        DSL_ODE_ACTION_EMAIL_PTR pAction = 
            DSL_ODE_ACTION_EMAIL_NEW(actionName.c_str(), pMailer, subject.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new FileOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FileOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string filePath("./my-file.txt");
        bool forceFlush(true);

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_FILE_PTR pAction = DSL_ODE_ACTION_FILE_NEW(
                actionName.c_str(), filePath.c_str(), forceFlush);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A FileOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new FileOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string filePath("./my-file.txt");
        bool forceFlush(false);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_FILE_PTR pAction = DSL_ODE_ACTION_FILE_NEW(
            actionName.c_str(), filePath.c_str(), forceFlush);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action disable other Handler will produce an error message as Handler does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A FileOdeAction with forceFlush set flushes the stream correctly", "[OdeAction]" )
{
    GIVEN( "A new FileOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string filePath("./my-file.txt");
        bool forceFlush(true);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_FILE_PTR pAction = DSL_ODE_ACTION_FILE_NEW(
            actionName.c_str(), filePath.c_str(), forceFlush);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: verification requires visual post inspection of the file.
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
                
                // Simulate the idle thread callback
                // Flush must return false to unschedule, self remove
                REQUIRE( pAction->Flush() == false );
            }
        }
    }
}

SCENARIO( "A new HandlerDisableOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new HandlerDisableOdeAction" ) 
    {
        std::string actionName("action");
        std::string handlerName("handler");

        WHEN( "A new HandlerDisableOdeAction is created" )
        {
            DSL_ODE_ACTION_TRIGGER_DISABLE_PTR pAction = 
                DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(actionName.c_str(), handlerName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A HandlerDisableOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new HandlerDisableOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string handlerName("handler");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TRIGGER_DISABLE_PTR pAction = 
            DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(actionName.c_str(), handlerName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action disable other Handler will produce an error message as Handler does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new FillSurroundingsOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FillSurroundingsOdeAction" ) 
    {
        std::string actionName("ode-action");

        std::string colorName("my-custom-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pBgColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);

        WHEN( "A new FillSurroundingsOdeAction is created" )
        {
            DSL_ODE_ACTION_FILL_OBJECT_PTR pAction = 
                DSL_ODE_ACTION_FILL_OBJECT_NEW(actionName.c_str(), pBgColor);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A new FillObjectOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FillObjectOdeAction" ) 
    {
        std::string actionName("ode-action");

        std::string colorName("my-custom-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);

        WHEN( "A new FillObjectOdeAction is created" )
        {
            DSL_ODE_ACTION_FILL_OBJECT_PTR pAction = 
                DSL_ODE_ACTION_FILL_OBJECT_NEW(actionName.c_str(), pColor);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A FillObjectOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new FillObjectOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "ode-action";

        std::string colorName("my-custom-color");
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);

        DSL_RGBA_COLOR_PTR pColor = DSL_RGBA_COLOR_NEW(colorName.c_str(), red, green, blue, alpha);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_FILL_OBJECT_PTR pAction = 
            DSL_ODE_ACTION_FILL_OBJECT_NEW(actionName.c_str(), pColor);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            objectMeta.rect_params.border_width = 9;
            objectMeta.rect_params.has_bg_color = false;  // Set false, action must set true
            objectMeta.rect_params.bg_color.red = 0;
            objectMeta.rect_params.bg_color.green = 0;
            objectMeta.rect_params.bg_color.blue = 0;
            objectMeta.rect_params.bg_color.alpha = 0;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
                // Boarder Width must be unchanged
                REQUIRE( objectMeta.rect_params.border_width == 9 );
                
                // Has background color must be enabled
                REQUIRE( objectMeta.rect_params.has_bg_color == 1 );
                
                // Background color must be updated
                REQUIRE( objectMeta.rect_params.bg_color.red == red );
                REQUIRE( objectMeta.rect_params.bg_color.green == green );
                REQUIRE( objectMeta.rect_params.bg_color.blue == blue );
                REQUIRE( objectMeta.rect_params.bg_color.alpha == alpha );
            }
        }
    }
}

SCENARIO( "A new HideOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new HideOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_HIDE_PTR pAction = 
                DSL_ODE_ACTION_HIDE_NEW(actionName.c_str(), true, true);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A HideOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new HideOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("ode-action");
        std::string displayText("display-text");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_HIDE_PTR pAction = 
            DSL_ODE_ACTION_HIDE_NEW(actionName.c_str(), true, true);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.text_params.set_bg_clr = 1; // set true, hide action must disable
            objectMeta.text_params.font_params.font_size = 10; // set size, hide action must disable
            objectMeta.rect_params.border_width = 10; // set width, hide action must disable
            objectMeta.text_params.display_text = (char*)(123); // Must have text for hide action to hide
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
                REQUIRE( objectMeta.text_params.set_bg_clr == 0 );
                REQUIRE( objectMeta.text_params.font_params.font_size == 0 );
                REQUIRE( objectMeta.rect_params.border_width == 0 );
            }
        }
    }
}

SCENARIO( "A new LogOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new LogOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_LOG_PTR pAction = 
                DSL_ODE_ACTION_LOG_NEW(actionName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A LogOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new LogOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "ode-action";

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_LOG_PTR pAction = 
            DSL_ODE_ACTION_LOG_NEW(actionName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new PauseOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new PauseOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string pipelineName("pipeline");

        WHEN( "A new PauseOdeAction is created" )
        {
            DSL_ODE_ACTION_PAUSE_PTR pAction = 
                DSL_ODE_ACTION_PAUSE_NEW(actionName.c_str(), pipelineName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A PauseOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new PauseOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "ode-action";
        std::string pipelineName("pipeline");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PAUSE_PTR pAction = 
            DSL_ODE_ACTION_PAUSE_NEW(actionName.c_str(), pipelineName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Pipeline pause will produce an error message as it does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new PrintOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new PrintOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_PRINT_PTR pAction = 
                DSL_ODE_ACTION_PRINT_NEW(actionName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A PrintOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new PrintOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "ode-action";

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_PRINT_PTR pAction = 
            DSL_ODE_ACTION_PRINT_NEW(actionName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = 1615768434973357000;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10.123;
            objectMeta.rect_params.top = 10.123;
            objectMeta.rect_params.width = 200.123;
            objectMeta.rect_params.height = 100.123;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new RedactOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new RedactOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new RedactOdeAction is created" )
        {
            DSL_ODE_ACTION_REDACT_PTR pAction = 
                DSL_ODE_ACTION_REDACT_NEW(actionName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A RedactOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new RedactOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName = "ode-action";

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_REDACT_PTR pAction = 
            DSL_ODE_ACTION_REDACT_NEW(actionName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
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
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
                REQUIRE( objectMeta.rect_params.border_width == 0 );
                REQUIRE( objectMeta.rect_params.has_bg_color == 1 );
                REQUIRE( objectMeta.rect_params.bg_color.red == 0.0 );
                REQUIRE( objectMeta.rect_params.bg_color.green == 0.0 );
                REQUIRE( objectMeta.rect_params.bg_color.blue == 0.0 );
                REQUIRE( objectMeta.rect_params.bg_color.alpha == 1.0 );
            }
        }
    }
}

SCENARIO( "A new SinkAddOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new SinkAddOdeAction" ) 
    {
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sinkName("sink");

        WHEN( "A new SinkAddOdeAction is created" )
        {
            DSL_ODE_ACTION_SINK_ADD_PTR pAction = 
                DSL_ODE_ACTION_SINK_ADD_NEW(actionName.c_str(), pipelineName.c_str(), sinkName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A SinkAddOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new SinkAddOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sinkName("sink");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_SINK_ADD_PTR pAction = 
            DSL_ODE_ACTION_SINK_ADD_NEW(actionName.c_str(), pipelineName.c_str(), sinkName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Pipeline Sink add will produce an error message as neither components exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new SinkRemoveOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new SinkRemoveOdeAction" ) 
    {
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sinkName("sink");

        WHEN( "A new SinkRemoveOdeAction is created" )
        {
            DSL_ODE_ACTION_SINK_REMOVE_PTR pAction = 
                DSL_ODE_ACTION_SINK_REMOVE_NEW(actionName.c_str(), pipelineName.c_str(), sinkName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A SinkRemoveOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new SinkRemoveOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sinkName("sink");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_SINK_REMOVE_PTR pAction = 
            DSL_ODE_ACTION_SINK_REMOVE_NEW(actionName.c_str(), pipelineName.c_str(), sinkName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Pipeline Sink remove will produce an error message as neither components exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new SourceAddOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new SourceAddOdeAction" ) 
    {
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sourceName("source");

        WHEN( "A new SourceAddOdeAction is created" )
        {
            DSL_ODE_ACTION_SOURCE_ADD_PTR pAction = 
                DSL_ODE_ACTION_SOURCE_ADD_NEW(actionName.c_str(), pipelineName.c_str(), sourceName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A SourceAddOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new SourceAddOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sourceName("source");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_SOURCE_ADD_PTR pAction = 
            DSL_ODE_ACTION_SOURCE_ADD_NEW(actionName.c_str(), pipelineName.c_str(), sourceName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Pipeline Source add will produce an error message as neither components exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new SourceRemoveOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new SourceRemoveOdeAction" ) 
    {
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sourceName("source");

        WHEN( "A new SourceRemoveOdeAction is created" )
        {
            DSL_ODE_ACTION_SOURCE_REMOVE_PTR pAction = 
                DSL_ODE_ACTION_SOURCE_REMOVE_NEW(actionName.c_str(), pipelineName.c_str(), sourceName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A SourceRemoveOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new SourceRemoveOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string pipelineName("pipeline");
        std::string sourceName("source");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_SOURCE_REMOVE_PTR pAction = 
            DSL_ODE_ACTION_SOURCE_REMOVE_NEW(actionName.c_str(), pipelineName.c_str(), sourceName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Pipeline Source remove will produce an error message as neither components exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new ActionAddOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new ActionAddOdeAction" ) 
    {
        std::string actionName("action");
        std::string triggerName("trigger");
        std::string otherActionName("other-action");

        WHEN( "A new ActionAddOdeAction is created" )
        {
            DSL_ODE_ACTION_AREA_ADD_PTR pAction = 
                DSL_ODE_ACTION_AREA_ADD_NEW(actionName.c_str(), triggerName.c_str(), otherActionName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "An ActionAddOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new ActionAddOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string otherActionName("other-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_AREA_ADD_PTR pAction = 
            DSL_ODE_ACTION_AREA_ADD_NEW(actionName.c_str(), triggerName.c_str(), otherActionName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Trigger Action add will produce an error message as neither components exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new ActionDisableOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new ActionDisableOdeAction" ) 
    {
        std::string actionName("action");
        std::string otherActionName("trigger");

        WHEN( "A new ActionDisableOdeAction is created" )
        {
            DSL_ODE_ACTION_ACTION_DISABLE_PTR pAction = 
                DSL_ODE_ACTION_ACTION_DISABLE_NEW(actionName.c_str(), otherActionName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A ActionDisableOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new ActionDisableOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string otherActionName("trigger");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_ACTION_DISABLE_PTR pAction = 
            DSL_ODE_ACTION_ACTION_DISABLE_NEW(actionName.c_str(), otherActionName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action disable other action will produce an error message as it does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new ActionEnableOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new ActionEnableOdeAction" ) 
    {
        std::string actionName("action");
        std::string otherActionName("trigger");

        WHEN( "A new ActionEnableOdeAction is created" )
        {
            DSL_ODE_ACTION_ACTION_ENABLE_PTR pAction = 
                DSL_ODE_ACTION_ACTION_ENABLE_NEW(actionName.c_str(), otherActionName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A ActionEnableOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new ActionEnableOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string otherActionName("trigger");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_ACTION_ENABLE_PTR pAction = 
            DSL_ODE_ACTION_ACTION_ENABLE_NEW(actionName.c_str(), otherActionName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action enable other action will produce an error message as it does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new AreaAddOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new AreaAddOdeAction" ) 
    {
        std::string actionName("action");
        std::string triggerName("trigger");
        std::string areaName("area");

        WHEN( "A new AreaAddOdeAction is created" )
        {
            DSL_ODE_ACTION_AREA_ADD_PTR pAction = 
                DSL_ODE_ACTION_AREA_ADD_NEW(actionName.c_str(), triggerName.c_str(), areaName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A AreaAddOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new AreaAddOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string areaName("area");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_AREA_ADD_PTR pAction = 
            DSL_ODE_ACTION_AREA_ADD_NEW(actionName.c_str(), triggerName.c_str(), areaName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Trigger Area add will produce an error message as neither components exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}


SCENARIO( "A new TriggerResetOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new TriggerResetOdeAction" ) 
    {
        std::string actionName("action");
        std::string otherTriggerName("reset");

        WHEN( "A new TriggerResetOdeAction is created" )
        {
            DSL_ODE_ACTION_TRIGGER_DISABLE_PTR pAction = 
                DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(actionName.c_str(), otherTriggerName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A TriggerResetOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new TriggerResetOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string otherTriggerName("reset");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TRIGGER_RESET_PTR pAction = 
            DSL_ODE_ACTION_TRIGGER_RESET_NEW(actionName.c_str(), otherTriggerName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action reset other Trigger will produce an error message as Trigger does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}


SCENARIO( "A new TriggerDisableOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new TriggerDisableOdeAction" ) 
    {
        std::string actionName("action");
        std::string otherTriggerName("trigger");

        WHEN( "A new TriggerDisableOdeAction is created" )
        {
            DSL_ODE_ACTION_TRIGGER_DISABLE_PTR pAction = 
                DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(actionName.c_str(), otherTriggerName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A TriggerDisableOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new TriggerDisableOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string otherTriggerName("trigger");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TRIGGER_DISABLE_PTR pAction = 
            DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(actionName.c_str(), otherTriggerName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action disable other Trigger will produce an error message as Trigger does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new TriggerEnableOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new TriggerEnableOdeAction" ) 
    {
        std::string actionName("action");
        std::string otherTriggerName("trigger");

        WHEN( "A new TriggerEnableOdeAction is created" )
        {
            DSL_ODE_ACTION_TRIGGER_ENABLE_PTR pAction = 
                DSL_ODE_ACTION_TRIGGER_ENABLE_NEW(actionName.c_str(), otherTriggerName.c_str());

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A TriggerEnableOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new TriggerEnableOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string otherTriggerName("trigger");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TRIGGER_ENABLE_PTR pAction = 
            DSL_ODE_ACTION_TRIGGER_ENABLE_NEW(actionName.c_str(), otherTriggerName.c_str());

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action enable other Trigger will produce an error message as the Trigger does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new RecordSinkStartOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new RecordSinkStartOdeAction" ) 
    {
        std::string actionName("action");
        std::string recordSink("record-sink");
        
        std::string recordSinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb client_listener;
        
        DSL_RECORD_SINK_PTR pRecordingSinkBintr = DSL_RECORD_SINK_NEW(recordSinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, client_listener);

        WHEN( "A new RecordSinkStartOdeAction is created" )
        {
            DSL_ODE_ACTION_SINK_RECORD_START_PTR pAction = 
                DSL_ODE_ACTION_SINK_RECORD_START_NEW(actionName.c_str(), pRecordingSinkBintr, 1, 1, NULL);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A RecordSinkStartOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new RecordSinkStartOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        
        std::string recordSinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb client_listener;

        DSL_RECORD_SINK_PTR pRecordingSinkBintr = DSL_RECORD_SINK_NEW(recordSinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, client_listener);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);


        DSL_ODE_ACTION_SINK_RECORD_START_PTR pAction = 
            DSL_ODE_ACTION_SINK_RECORD_START_NEW(actionName.c_str(), pRecordingSinkBintr, 1, 1, NULL);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action will produce an error message as the Record Sink does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new RecordSinkStopOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new RecordSinkStopOdeAction" ) 
    {
        std::string actionName("action");
        std::string recordSink("record-sink");

        std::string recordSinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb client_listener;
        
        DSL_RECORD_SINK_PTR pRecordingSinkBintr = DSL_RECORD_SINK_NEW(recordSinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, client_listener);

        WHEN( "A new RecordSinkStopOdeAction is created" )
        {
            DSL_ODE_ACTION_SINK_RECORD_STOP_PTR pAction = 
                DSL_ODE_ACTION_SINK_RECORD_STOP_NEW(actionName.c_str(), pRecordingSinkBintr);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A RecordSinkStopOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new RecordSinkStopOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        
        std::string recordSinkName("record-sink");
        std::string outdir("./");
        uint codec(DSL_CODEC_H264);
        uint bitrate(2000000);
        uint interval(0);
        uint container(DSL_CONTAINER_MP4);
        
        dsl_record_client_listener_cb client_listener;
        
        DSL_RECORD_SINK_PTR pRecordingSinkBintr = DSL_RECORD_SINK_NEW(recordSinkName.c_str(), 
            outdir.c_str(), codec, container, bitrate, interval, client_listener);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_SINK_RECORD_STOP_PTR pAction = 
            DSL_ODE_ACTION_SINK_RECORD_STOP_NEW(actionName.c_str(), pRecordingSinkBintr);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action will produce an error message as the Record Sink does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new RecordTapStartOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new RecordSinkStartOdeAction" ) 
    {
        std::string actionName("action");
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);

        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);

        WHEN( "A new RecordTapStartOdeAction is created" )
        {
            DSL_ODE_ACTION_TAP_RECORD_START_PTR pAction = 
                DSL_ODE_ACTION_TAP_RECORD_START_NEW(actionName.c_str(), pRecordTapBintr, 1, 1, NULL);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A RecordTapStartOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new RecordTapStartOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);

        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TAP_RECORD_START_PTR pAction = 
            DSL_ODE_ACTION_TAP_RECORD_START_NEW(actionName.c_str(), pRecordTapBintr, 1, 1, NULL);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action will produce an error message as the Record Sink does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new RecordTapStopOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new RecordTapStopOdeAction" ) 
    {
        std::string actionName("action");
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);

        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);

        WHEN( "A new RecordTapStopOdeAction is created" )
        {
            DSL_ODE_ACTION_TAP_RECORD_STOP_PTR pAction = 
                DSL_ODE_ACTION_TAP_RECORD_STOP_NEW(actionName.c_str(), pRecordTapBintr);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A RecordTapStopOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new RecordTapStopOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string recordTapName("record-tap");
        std::string outDir("./");
        uint container(DSL_CONTAINER_MKV);

        dsl_record_client_listener_cb clientListener;

        DSL_RECORD_TAP_PTR pRecordTapBintr = 
            DSL_RECORD_TAP_NEW(recordTapName.c_str(), outDir.c_str(), container, clientListener);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TAP_RECORD_STOP_PTR pAction = 
            DSL_ODE_ACTION_TAP_RECORD_STOP_NEW(actionName.c_str(), pRecordTapBintr);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action will produce an error message as the Record Sink does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new TilerShowSourceOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new TilerShowSourceOdeAction" ) 
    {
        std::string actionName("action");
        std::string tilerName("tiler");
        uint timeout(2);
        bool hasPrecedence(true);

        WHEN( "A new TilerShowSourceOdeAction is created" )
        {
            DSL_ODE_ACTION_TILER_SHOW_SOURCE_PTR pAction = 
                DSL_ODE_ACTION_TILER_SHOW_SOURCE_NEW(actionName.c_str(), tilerName.c_str(), timeout, hasPrecedence);

            THEN( "The Action's memebers are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A TilerShowSourceOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new TilerShowSourceOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        bool hasPrecedence(true);
        
        std::string actionName("action");
        std::string tilerName("tiller");
        uint timeout(2);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_TILER_SHOW_SOURCE_PTR pAction = 
            DSL_ODE_ACTION_TILER_SHOW_SOURCE_NEW(actionName.c_str(), tilerName.c_str(), timeout, hasPrecedence);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action will produce an error message as the Trigger does not exist
                pAction->HandleOccurrence(pTrigger, NULL, NULL, &frameMeta, &objectMeta);
            }
        }
    }
}
