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

#include "Dsl.h"
#include "DslDetectionEvent.h"
#include "DslEventAction.h"

namespace DSL
{
    EventAction::EventAction(const char* name)
        : Base(name)
    {
        LOG_FUNC();

    }

    EventAction::~EventAction()
    {
        LOG_FUNC();

    }

    // ********************************************************************

    CallbackEventAction::CallbackEventAction(const char* name)
        : EventAction(name)
    {
        LOG_FUNC();

    }

    CallbackEventAction::~CallbackEventAction()
    {
        LOG_FUNC();

    }
    
    void CallbackEventAction::HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOG_FUNC();

    }

    // ********************************************************************

    DisplayEventAction::DisplayEventAction(const char* name)
        : EventAction(name)
    {
        LOG_FUNC();

    }

    DisplayEventAction::~DisplayEventAction()
    {
        LOG_FUNC();

    }

    void DisplayEventAction::HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOG_FUNC();

    }

    // ********************************************************************

    LogEventAction::LogEventAction(const char* name)
        : EventAction(name)
    {
        LOG_FUNC();

    }

    LogEventAction::~LogEventAction()
    {
        LOG_FUNC();

    }

    void LogEventAction::HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        LOG_FUNC();

        DSL_DETECTION_EVENT_PTR pDetectionEvent = std::dynamic_pointer_cast<DetectionEvent>(pEvent);
        
        uint minWidth(0), minHeight(0);
        pDetectionEvent->GetMinDimensions(&minWidth, &minHeight);
        uint minFrameCountN(0), minFrameCountD(0); 
        pDetectionEvent->GetMinFrameCount(&minFrameCountN, &minFrameCountD);
        
        LOG_INFO("Event Name      : " << pDetectionEvent->GetName());
        LOG_INFO("  Event Id      : " << eventId);
        LOG_INFO("  Frame Number  : " << pFrameMeta->frame_num );
        LOG_INFO("  NTP Timestamp : " << pFrameMeta->ntp_timestamp );
        LOG_INFO("  Source Id     : " << pFrameMeta->source_id );
        LOG_INFO("  Class Id      : " << pObjectMeta->class_id);
        LOG_INFO("  Object Id     : " << pObjectMeta->object_id);
        LOG_INFO("  Object Label  : " << pObjectMeta->obj_label);
        LOG_INFO("  Object Data   : ------------------------");
        LOG_INFO("    Confidence  : " << pObjectMeta->confidence);
        LOG_INFO("    Left        : " << pObjectMeta->rect_params.left);
        LOG_INFO("    Top         : " << pObjectMeta->rect_params.top);
        LOG_INFO("    Width       : " << pObjectMeta->rect_params.width);
        LOG_INFO("    Height      : " << pObjectMeta->rect_params.height);
        LOG_INFO("  Min Criteria  : ------------------------");
        LOG_INFO("    Confidence  : " << pDetectionEvent->GetMinConfidence());
        LOG_INFO("    Frame Count : " << minFrameCountN << " out of " << minFrameCountD);
        LOG_INFO("    Width       : " << minWidth);
        LOG_INFO("    Height      : " << minHeight);
    }

}    
