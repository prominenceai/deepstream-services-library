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

static std::vector<NvDsDisplayMeta*> displayMetaData;

static void ode_occurrence_handler_cb(uint64_t event_id, const wchar_t* name,
    void* buffer, void* display_meta, void* frame_meta, void* object_meta, void* client_data)
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

static void ode_occurrence_monitor_cb(dsl_ode_occurrence_info* pInfo, void* client_data)
{
    std::wcout << "Trigger Name        : " << pInfo->trigger_name << "\n";
    std::cout << "  Unique Id         : " << pInfo->unique_ode_id << "\n";
    std::cout << "  NTP Timestamp     : " << pInfo->ntp_timestamp << "\n";
    std::cout << "  Source Data       : ------------------------" << "\n";
    std::cout << "    Id              : " << pInfo->source_info.source_id << "\n";
    std::cout << "    Batch Id        : " << pInfo->source_info.batch_id << "\n";
    std::cout << "    Pad Index       : " << pInfo->source_info.pad_index << "\n";
    std::cout << "    Frame           : " << pInfo->source_info.frame_num << "\n";
    std::cout << "    Width           : " << pInfo->source_info.frame_width << "\n";
    std::cout << "    Height          : " << pInfo->source_info.frame_height << "\n";
    std::cout << "    Infer Done      : " << pInfo->source_info.inference_done << "\n";

    if (pInfo->is_object_occurrence)
    {
        std::cout << "  Object Data       : ------------------------" << "\n";
        std::cout << "    Class Id        : " << pInfo->object_info.class_id << "\n";
        std::cout << "    Infer Comp Id   : " << pInfo->object_info.inference_component_id << "\n";
        std::cout << "    Tracking Id     : " << pInfo->object_info.tracking_id << "\n";
        std::cout << "    Label           : " << pInfo->object_info.label << "\n";
        std::cout << "    Persistence     : " << pInfo->object_info.persistence << "\n";
        std::cout << "    Direction       : " << pInfo->object_info.direction << "\n";
        std::cout << "    Infer Conf      : " << pInfo->object_info.inference_confidence << "\n";
        std::cout << "    Track Conf      : " << pInfo->object_info.tracker_confidence << "\n";
        std::cout << "    Left            : " << pInfo->object_info.left << "\n";
        std::cout << "    Top             : " << pInfo->object_info.top << "\n";
        std::cout << "    Width           : " << pInfo->object_info.width << "\n";
        std::cout << "    Height          : " << pInfo->object_info.height << "\n";
    }
    else
    {
        std::cout << "  Accumulative Data : ------------------------" << "\n";
        std::cout << "    Occurrences     : " << pInfo->accumulative_info.occurrences_total << "\n";
        std::cout << "    Occurrences In  : " << pInfo->accumulative_info.occurrences_total << "\n";
        std::cout << "    Occurrences Out : " << pInfo->accumulative_info.occurrences_total << "\n";
    }
    std::cout << "  Trigger Criteria  : ------------------------" << "\n";
    std::cout << "    Class Id        : " << pInfo->criteria_info.class_id << "\n";
    std::cout << "    Infer Comp Id   : " << pInfo->criteria_info.inference_component_id << "\n";
    std::cout << "    Min Infer Conf  : " << pInfo->criteria_info.min_inference_confidence << "\n";
    std::cout << "    Min Track Conf  : " << pInfo->criteria_info.min_tracker_confidence << "\n";
    std::cout << "    Infer Done Only : " << pInfo->criteria_info.inference_done_only << "\n";
    std::cout << "    Min Width       : " << pInfo->criteria_info.min_width << "\n";
    std::cout << "    Min Height      : " << pInfo->criteria_info.min_height << "\n";
    std::cout << "    Max Width       : " << pInfo->criteria_info.max_width << "\n";
    std::cout << "    Max Height      : " << pInfo->criteria_info.max_height << "\n";
    std::cout << "    Interval        : " << pInfo->criteria_info.interval << "\n";
}

SCENARIO( "A new FormatBBoxOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FormatBBoxOdeAction" ) 
    {
        std::string actionName("ode-action");

        uint borderWidth(10);
        std::string borderColorName("border-color");
        bool hasBgColor(true);
        std::string bgColorName("border-color");
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        DSL_RGBA_COLOR_PTR pBorderColor = DSL_RGBA_COLOR_NEW(borderColorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_COLOR_PTR pBgColor = DSL_RGBA_COLOR_NEW(borderColorName.c_str(), 
            red, green, blue, alpha);

        WHEN( "A new FormatBBoxOdeAction is created" )
        {
            DSL_ODE_ACTION_BBOX_FORMAT_PTR pAction = 
                DSL_ODE_ACTION_BBOX_FORMAT_NEW(actionName.c_str(), 
                    borderWidth, pBorderColor, hasBgColor, pBgColor);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A FormatBBoxOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new FormatBBoxOdeAction" ) 
    {
        std::string odeTriggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string actionName("ode-action");
        uint borderWidth(10);
        std::string borderColorName("border-color");
        bool hasBgColor(true);
        std::string bgColorName("border-color");
        
        double red(0.12), green(0.34), blue(0.56), alpha(0.78);
        DSL_RGBA_COLOR_PTR pBorderColor = DSL_RGBA_COLOR_NEW(borderColorName.c_str(), 
            red, green, blue, alpha);
        DSL_RGBA_COLOR_PTR pBgColor = DSL_RGBA_COLOR_NEW(borderColorName.c_str(), 
            red, green, blue, alpha);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_BBOX_FORMAT_PTR pAction = 
            DSL_ODE_ACTION_BBOX_FORMAT_NEW(actionName.c_str(), 
                borderWidth, pBorderColor, hasBgColor, pBgColor);

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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new ScaleBBoxOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new ScaleBBoxOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new OdeAction is created with a valid scale factor" )
        {
            DSL_ODE_ACTION_BBOX_SCALE_PTR pAction = DSL_ODE_ACTION_BBOX_SCALE_NEW(
                actionName.c_str(), 110);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A ScaleBBoxOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new ScaleBBoxOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("ode-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 20;
        objectMeta.rect_params.top = 20;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        objectMeta.text_params.x_offset = 20;
        objectMeta.text_params.y_offset = 10;
        
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;
        frameMeta.source_frame_width = DSL_DEFAULT_STREAMMUX_WIDTH;
        frameMeta.source_frame_height = DSL_DEFAULT_STREAMMUX_HEIGHT;

        WHEN( "the scale factor stays within frame" )
        {
            uint scale(110);

            DSL_ODE_ACTION_BBOX_SCALE_PTR pAction = DSL_ODE_ACTION_BBOX_SCALE_NEW(
                actionName.c_str(), scale);
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                
                uint exp_x_offset(10), exp_y_offset(5);
                float exp_left(10), exp_top(15);
                float exp_width(220), exp_height(110);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
                REQUIRE( objectMeta.rect_params.left == exp_left );
                REQUIRE( objectMeta.rect_params.top == exp_top );
                REQUIRE( objectMeta.rect_params.width == exp_width );
                REQUIRE( objectMeta.rect_params.height == exp_height );
            }
        }
        WHEN( "the scale factor exceeds top left corner of the frame" )
        {
            uint scale(200);

            DSL_ODE_ACTION_BBOX_SCALE_PTR pAction = DSL_ODE_ACTION_BBOX_SCALE_NEW(
                actionName.c_str(), scale);
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                uint exp_x_offset(0), exp_y_offset(0);
                float exp_left(0), exp_top(0);
                float exp_width(320), exp_height(170);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
                REQUIRE( objectMeta.rect_params.left == exp_left );
                REQUIRE( objectMeta.rect_params.top == exp_top );
                REQUIRE( objectMeta.rect_params.width == exp_width );
                REQUIRE( objectMeta.rect_params.height == exp_height );
            }
        }
        WHEN( "the scale factor exceeds bottom right corner of the frame" )
        {
            uint scale(200);

            DSL_ODE_ACTION_BBOX_SCALE_PTR pAction = DSL_ODE_ACTION_BBOX_SCALE_NEW(
                actionName.c_str(), scale);
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                objectMeta.rect_params.left = 
                    (DSL_DEFAULT_STREAMMUX_WIDTH-210);
                objectMeta.rect_params.top = 
                    (DSL_DEFAULT_STREAMMUX_HEIGHT-110);
                objectMeta.text_params.x_offset = objectMeta.rect_params.left;
                objectMeta.text_params.y_offset = objectMeta.rect_params.top-10;
                
                uint exp_x_offset(1610), exp_y_offset(910);
                float exp_left(1610), exp_top(920);
                
                // width and height clipped to max dimensions
                float exp_width(309), exp_height(159);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
                REQUIRE( objectMeta.rect_params.left == exp_left );
                REQUIRE( objectMeta.rect_params.top == exp_top );
                REQUIRE( objectMeta.rect_params.width == exp_width );
                REQUIRE( objectMeta.rect_params.height == exp_height );
            }
        }
    }
}

SCENARIO( "A new FormatLabelOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FormatLabelOdeAction" ) 
    {
        std::string actionName("ode-action");

        uint borderWidth(10);
        std::string fontName("label-font");
        std::string font("arial");
        uint size(14);
        std::string fontColorName("black");
        double redFont(0.0), greenFont(0.0), blueFont(0.0), alphaFont(1.0);
        
        bool hasBgColor(true);
        std::string bgColorName("bg-color");
        
        double redBgColor(0.12), greenBgColor(0.34), blueBgColor(0.56), alphaBgColor(0.78);
        
        DSL_RGBA_COLOR_PTR pFontColor = DSL_RGBA_COLOR_NEW(fontColorName.c_str(), 
            redFont, greenFont, blueFont, alphaFont);
        DSL_RGBA_COLOR_PTR pBgColor = DSL_RGBA_COLOR_NEW(bgColorName.c_str(), 
            redBgColor, greenBgColor, blueBgColor, alphaBgColor);
        DSL_RGBA_FONT_PTR pFont = DSL_RGBA_FONT_NEW(fontName.c_str(),
            font.c_str(), size, pFontColor);

        WHEN( "A new FormatLabelOdeAction is created" )
        {
            DSL_ODE_ACTION_LABEL_FORMAT_PTR pAction = 
                DSL_ODE_ACTION_LABEL_FORMAT_NEW(actionName.c_str(), 
                    pFont, hasBgColor, pBgColor);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A FormatLabelOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new FormatLabelOdeAction" ) 
    {
        std::string odeTriggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string actionName("ode-action");

        uint borderWidth(10);
        std::string fontName("label-font");
        std::string font("arial");
        uint size(14);
        std::string fontColorName("black");
        double redFont(0.0), greenFont(0.0), blueFont(0.0), alphaFont(1.0);
        
        bool hasBgColor(true);
        std::string bgColorName("bg-color");
        
        double redBgColor(0.12), greenBgColor(0.34), blueBgColor(0.56), alphaBgColor(0.78);
        
        DSL_RGBA_COLOR_PTR pFontColor = DSL_RGBA_COLOR_NEW(fontColorName.c_str(), 
            redFont, greenFont, blueFont, alphaFont);
        DSL_RGBA_COLOR_PTR pBgColor = DSL_RGBA_COLOR_NEW(bgColorName.c_str(), 
            redBgColor, greenBgColor, blueBgColor, alphaBgColor);
        DSL_RGBA_FONT_PTR pFont = DSL_RGBA_FONT_NEW(fontName.c_str(),
            font.c_str(), size, pFontColor);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_LABEL_FORMAT_PTR pAction = 
            DSL_ODE_ACTION_LABEL_FORMAT_NEW(actionName.c_str(), 
                pFont, hasBgColor, pBgColor);

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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new MonitorOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new MonitorOdeAction" ) 
    {
        std::string actionName("ode-action");

        WHEN( "A new MonitorOdeAction is created" )
        {
            DSL_ODE_ACTION_MONITOR_PTR pAction = 
                DSL_ODE_ACTION_MONITOR_NEW(actionName.c_str(), ode_occurrence_monitor_cb, NULL);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A MonitorOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new MonitorOdeAction" ) 
    {
        std::string odeTriggerName("first-occurrence");
        std::string source;
        uint classId(1);
        uint limit(1);

        std::string actionName("ode-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(odeTriggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_MONITOR_PTR pAction = 
            DSL_ODE_ACTION_MONITOR_NEW(actionName.c_str(), ode_occurrence_monitor_cb, NULL);
            
        std::string inferCompName("infer-comp");
        pTrigger->SetInfer(inferCompName.c_str());
        pTrigger->SetMinConfidence(0.234);
        pTrigger->SetMinTrackerConfidence(0.567);
        pTrigger->SetMinDimensions(123,123);
        pTrigger->SetMaxDimensions(456,456);
        pTrigger->SetInferDoneOnlySetting(true);
        pTrigger->SetInterval(4);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            NvDsObjectMeta objectMeta = {0};
            objectMeta.class_id = classId; // must match Detections Trigger's classId
            
            std::string objectLabel("detected-object");
            objectMeta.obj_label[objectLabel.copy(objectMeta.obj_label, 127)] = 0;
            objectMeta.object_id = INT64_MAX; 
            objectMeta.rect_params.left = 10;
            objectMeta.rect_params.top = 10;
            objectMeta.rect_params.width = 200;
            objectMeta.rect_params.height = 100;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
        WHEN( "Object Meta is excluded" )
        {
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.bInferDone = true;  // required to process
            frameMeta.frame_num = 444;
            frameMeta.ntp_timestamp = INT64_MAX;
            frameMeta.source_id = 2;

            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, NULL);
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
                DSL_ODE_ACTION_CAPTURE_FRAME_NEW(actionName.c_str(), 
                    outdir.c_str(), annotate);

            THEN( "The Action's members are setup and returned correctly" )
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
                DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), 
                    outdir.c_str());

            THEN( "The Action's members are setup and returned correctly" )
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
            DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), 
                outdir.c_str());
        
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
            DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), 
                outdir.c_str());
        
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

SCENARIO( "A new CustomLabelOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new CustomLabelOdeAction" ) 
    {
        std::string actionName("ode-action");
        const std::vector<uint> label_types = {DSL_METRIC_OBJECT_LOCATION,
            DSL_METRIC_OBJECT_DIMENSIONS, DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE,
            DSL_METRIC_OBJECT_PERSISTENCE};

        WHEN( "A new OdeAction is created with an array of content types" )
        {
            DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR pAction = DSL_ODE_ACTION_LABEL_CUSTOMIZE_NEW(
                actionName.c_str(), label_types);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
        WHEN( "A new OdeAction is created with an empty array of content types" )
        {
            std::vector<uint> label_types;
            
            DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR pAction = DSL_ODE_ACTION_LABEL_CUSTOMIZE_NEW(
                actionName.c_str(), label_types);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A CustomLabelOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new CustomLabelOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        
        std::string actionName("ode-action");
        const std::vector<uint> label_types = {DSL_METRIC_OBJECT_LOCATION,
            DSL_METRIC_OBJECT_DIMENSIONS, DSL_METRIC_OBJECT_CONFIDENCE_INFERENCE,
        DSL_METRIC_OBJECT_PERSISTENCE};

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);
        
        std::string defaultLabel("Person 123");

        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  // required to process
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 10;
        objectMeta.rect_params.top = 10;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        objectMeta.text_params.display_text = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
        defaultLabel.copy(objectMeta.text_params.display_text, defaultLabel.size(), 0);

        WHEN( "A the Action is created with content types" )
        {
            DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR pAction = DSL_ODE_ACTION_LABEL_CUSTOMIZE_NEW(
                actionName.c_str(), label_types);

            THEN( "The OdeAction can Handle the Occurrence" )
            {
                std::string expectedLabel("L:10,10 | D:200x100 | IC:0.000000 | T:0s");

                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                std::string actualLabel(objectMeta.text_params.display_text);
                REQUIRE( actualLabel == expectedLabel );
            }
        }
        WHEN( "A the Action is created with 0 content_types" )
        {
            std::vector<uint> label_types;
            
            DSL_ODE_ACTION_LABEL_CUSTOMIZE_PTR pAction = DSL_ODE_ACTION_LABEL_CUSTOMIZE_NEW(
                actionName.c_str(), label_types);

            THEN( "The OdeAction can Handle the Occurrence" )
            {
                std::string expectedLabel("");

                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                std::string actualLabel(objectMeta.text_params.display_text);
                REQUIRE( actualLabel == expectedLabel );
            }
        }
    }
}

SCENARIO( "A new OffsetLabelOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new OffsetLabelOdeAction" ) 
    {
        std::string actionName("ode-action");
        int offsetX(-5), offsetY(-5);

        WHEN( "A new OdeAction is created with an array of content types" )
        {
            DSL_ODE_ACTION_LABEL_OFFSET_PTR pAction = DSL_ODE_ACTION_LABEL_OFFSET_NEW(
                actionName.c_str(), offsetX, offsetY);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A OffsetLabelOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new OffsetLabelOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint limit(1);
        
        std::string actionName("ode-action");

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        NvDsObjectMeta objectMeta = {0};
        objectMeta.class_id = classId; // must match Trigger's classId
        objectMeta.object_id = INT64_MAX; 
        objectMeta.rect_params.left = 20;
        objectMeta.rect_params.top = 20;
        objectMeta.rect_params.width = 200;
        objectMeta.rect_params.height = 100;
        
        NvDsFrameMeta frameMeta =  {0};
        frameMeta.bInferDone = true;  // required to process
        frameMeta.frame_num = 444;
        frameMeta.ntp_timestamp = INT64_MAX;
        frameMeta.source_id = 2;
        frameMeta.source_frame_width = DSL_DEFAULT_STREAMMUX_WIDTH;
        frameMeta.source_frame_height = DSL_DEFAULT_STREAMMUX_HEIGHT;

        WHEN( "offsets are defined negative and in-frame" )
        {
            int offsetX(-5), offsetY(-5);

            DSL_ODE_ACTION_LABEL_OFFSET_PTR pAction = DSL_ODE_ACTION_LABEL_OFFSET_NEW(
                actionName.c_str(), offsetX, offsetY);

            objectMeta.text_params.x_offset = 20;
            objectMeta.text_params.y_offset = 10;
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                uint exp_x_offset(15), exp_y_offset(5);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
            }
        }
        WHEN( "offsets are defined negative and out-of-frame" )
        {
            int offsetX(-25), offsetY(-15);

            DSL_ODE_ACTION_LABEL_OFFSET_PTR pAction = DSL_ODE_ACTION_LABEL_OFFSET_NEW(
                actionName.c_str(), offsetX, offsetY);

            objectMeta.text_params.x_offset = 20;
            objectMeta.text_params.y_offset = 10;
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                uint exp_x_offset(0), exp_y_offset(0);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
            }
        }
        WHEN( "offsets are defined positive and in-frame" )
        {
            int offsetX(5), offsetY(5);

            DSL_ODE_ACTION_LABEL_OFFSET_PTR pAction = DSL_ODE_ACTION_LABEL_OFFSET_NEW(
                actionName.c_str(), offsetX, offsetY);

            objectMeta.text_params.x_offset = 20;
            objectMeta.text_params.y_offset = 10;
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                uint exp_x_offset(25), exp_y_offset(15);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
            }
        }
        WHEN( "offsets are defined positive and out-of-frame" )
        {
            int offsetX(25), offsetY(15);

            DSL_ODE_ACTION_LABEL_OFFSET_PTR pAction = DSL_ODE_ACTION_LABEL_OFFSET_NEW(
                actionName.c_str(), offsetX, offsetY);

            objectMeta.text_params.x_offset = DSL_DEFAULT_STREAMMUX_WIDTH-10;
            objectMeta.text_params.y_offset = DSL_DEFAULT_STREAMMUX_HEIGHT-10;;
                
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                uint exp_x_offset(DSL_DEFAULT_STREAMMUX_WIDTH-1), 
                    exp_y_offset(DSL_DEFAULT_STREAMMUX_HEIGHT-1);
                
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                    
                REQUIRE( objectMeta.text_params.x_offset == exp_x_offset );
                REQUIRE( objectMeta.text_params.y_offset == exp_y_offset );
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A new Text FileOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FileOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string filePath("./event-file.txt");
        uint mode(DSL_WRITE_MODE_APPEND);
        bool forceFlush(true);

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_FILE_TEXT_PTR pAction = DSL_ODE_ACTION_FILE_TEXT_NEW(
                actionName.c_str(), filePath.c_str(), mode, forceFlush);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A new CSV FileOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FileOdeAction" ) 
    {
        std::string actionName("ode-action");
        bool forceFlush(true);

        WHEN( "A new CSV FileOdeAction is created in APPEND mode" )
        {
            std::string filePath("./event-file-append.csv");
            uint mode(DSL_WRITE_MODE_APPEND);

            // create the action and CSV file once
            {
                DSL_ODE_ACTION_FILE_CSV_PTR pAction = DSL_ODE_ACTION_FILE_CSV_NEW(
                    actionName.c_str(), filePath.c_str(), mode, forceFlush);
                    
                REQUIRE( pAction != nullptr );
            }
            // create the action and CSV file a second time 
            DSL_ODE_ACTION_FILE_CSV_PTR pAction = DSL_ODE_ACTION_FILE_CSV_NEW(
                actionName.c_str(), filePath.c_str(), mode, forceFlush);

            // ***************************************************
            // NOTE: requires manual verification to ensure header is only added once.
            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
        WHEN( "A new CSV FileOdeAction is created in TRUNCATE mode" )
        {
            std::string filePath("./event-file-truncate.csv");
            uint mode(DSL_WRITE_MODE_TRUNCATE);
            
            DSL_ODE_ACTION_FILE_CSV_PTR pAction = DSL_ODE_ACTION_FILE_CSV_NEW(
                actionName.c_str(), filePath.c_str(), mode, forceFlush);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
            }
        }
    }
}

SCENARIO( "A new MOT Challenge FileOdeAction is created correctly", "[OdeAction]" )
{
    GIVEN( "Attributes for a new FileOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string filePath("./event-file.txt");
        uint mode(DSL_WRITE_MODE_TRUNCATE);
        bool forceFlush(true);

        WHEN( "A new OdeAction is created" )
        {
            DSL_ODE_ACTION_FILE_MOTC_PTR pAction = DSL_ODE_ACTION_FILE_MOTC_NEW(
                actionName.c_str(), filePath.c_str(), mode, forceFlush);

            THEN( "The Action's members are setup and returned correctly" )
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
        uint mode(DSL_WRITE_MODE_TRUNCATE);
        bool forceFlush(false);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_FILE_TEXT_PTR pAction = DSL_ODE_ACTION_FILE_TEXT_NEW(
            actionName.c_str(), filePath.c_str(), mode, forceFlush);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: Action disable other Handler will produce an error message as Handler does not exist
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
    }
}

SCENARIO( "A MotcOdeAction handles an ODE Occurence correctly", "[OdeAction]" )
{
    GIVEN( "A new FileOdeAction" ) 
    {
        std::string triggerName("first-occurence");
        std::string source;
        uint classId(1);
        uint objectId(4);
        uint limit(1);
        
        std::string actionName("action");
        std::string filePath("./my-file.txt");
        uint mode(DSL_WRITE_MODE_TRUNCATE);
        bool forceFlush(false);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_FILE_MOTC_PTR pAction = DSL_ODE_ACTION_FILE_MOTC_NEW(
            actionName.c_str(), filePath.c_str(), mode, forceFlush);

        WHEN( "A new ODE is created" )
        {
            // Frame Meta test data
            NvDsFrameMeta frameMeta =  {0};
            frameMeta.frame_num = 1;

            // Object Meta test data
            NvDsObjectMeta objectMeta1 = {0};
            objectMeta1.class_id = classId; // must match ODE Trigger's classId
            objectMeta1.rect_params.left = 10.123;
            objectMeta1.rect_params.top = 10.456;
            objectMeta1.rect_params.width = 200.789;
            objectMeta1.rect_params.height = 100.1;
            
            // Object Meta test data
            NvDsObjectMeta objectMeta2 = {0};
            objectMeta2.class_id = classId; // must match ODE Trigger's classId
            objectMeta2.rect_params.left = 44.444;
            objectMeta2.rect_params.top = 33.333;
            objectMeta2.rect_params.width = 100.987;
            objectMeta2.rect_params.height = 200.1;
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta1);
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta2);
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
        uint mode(DSL_WRITE_MODE_APPEND);
        uint classId(1);
        uint limit(1);
        
        std::string actionName("action");
        std::string filePath("./my-file.txt");
        bool forceFlush(true);

        DSL_ODE_TRIGGER_OCCURRENCE_PTR pTrigger = 
            DSL_ODE_TRIGGER_OCCURRENCE_NEW(triggerName.c_str(), source.c_str(), classId, limit);

        DSL_ODE_ACTION_FILE_CSV_PTR pAction = DSL_ODE_ACTION_FILE_CSV_NEW(
            actionName.c_str(), filePath.c_str(), mode, forceFlush);

        WHEN( "A new ODE is created" )
        {
            NvDsFrameMeta frameMeta = {0};
            NvDsObjectMeta objectMeta = {0};
            
            THEN( "The OdeAction can Handle the Occurrence" )
            {
                // NOTE:: verification requires visual post inspection of the file.
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
                
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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
            DSL_ODE_ACTION_FILL_SURROUNDINGS_PTR pAction = 
                DSL_ODE_ACTION_FILL_SURROUNDINGS_NEW(actionName.c_str(), pBgColor);

            THEN( "The Action's members are setup and returned correctly" )
            {
                std::string retName = pAction->GetCStrName();
                REQUIRE( actionName == retName );
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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
                DSL_ODE_ACTION_PRINT_NEW(actionName.c_str(), false);

            THEN( "The Action's members are setup and returned correctly" )
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
            DSL_ODE_ACTION_PRINT_NEW(actionName.c_str(), false);

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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, 
                    NULL, displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
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

            THEN( "The Action's members are setup and returned correctly" )
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
                pAction->HandleOccurrence(pTrigger, NULL, 
                    displayMetaData, &frameMeta, &objectMeta);
            }
        }
    }
}

static void enabled_state_change_listener_1(boolean enabled, void* client_data)
{
    std::cout 
        << "Enabled State Change listener 1 callback called with enabled = " 
        << enabled << "\n";
}

static void enabled_state_change_listener_2(boolean enabled, void* client_data)
{
    std::cout 
        << "Enabled State Change listener 2 callback called with enabled = " 
        << enabled << "\n";
}

SCENARIO( "An OdeAction calls all enabled-state-change-listners on SetEnabled", "[OdeAction]" )
{
    GIVEN( "A new CaptureObjectOdeAction" ) 
    {
        std::string actionName("ode-action");
        std::string outdir("./");
        uint userData1(0), userData2(0);
        uint width(1280), height(720);

        DSL_ODE_ACTION_CAPTURE_OBJECT_PTR pAction = 
            DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(actionName.c_str(), 
                outdir.c_str());
        
        REQUIRE( pAction->AddEnabledStateChangeListener(
            enabled_state_change_listener_1, NULL) == true );
        REQUIRE( pAction->AddEnabledStateChangeListener(
            enabled_state_change_listener_2, NULL) == true );
        
        WHEN( "The ODE Action is disabled" )
        {
            pAction->SetEnabled(false);
            
            THEN( "All client callback functions are called" )
            {
                // requires manual/visual verification at this time.
            }
        }
        WHEN( "The ODE Trigger is enabled" )
        {
            pAction->SetEnabled(true);
            
            THEN( "All client callback functions are called" )
            {
                // requires manual/visual verification at this time.
            }
        }
    }
}
