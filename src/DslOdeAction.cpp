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
#include "DslOdeType.h"
#include "DslOdeAction.h"

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
    
    void CallbackEventAction::HandleOccurrence(DSL_ODE_OCCURRENCE_PTR pOdeOccurrence)
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

    void DisplayEventAction::HandleOccurrence(DSL_ODE_OCCURRENCE_PTR pOdeOccurrence)
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

    void LogEventAction::HandleOccurrence(DSL_ODE_OCCURRENCE_PTR pOdeOccurrence)
    {
        LOG_FUNC();

        LOG_INFO("Event Name      : " << pOdeOccurrence->event_name);
        LOG_INFO("  Type          : " << pOdeOccurrence->event_type);
        LOG_INFO("  Unique Id     : " << pOdeOccurrence->event_id);
        LOG_INFO("  NTP Timestamp : " << pOdeOccurrence->ntp_timestamp );
        LOG_INFO("  Source Data   : ------------------------");
        LOG_INFO("    Id          : " << pOdeOccurrence->source_id );
        LOG_INFO("    Frame       : " << pOdeOccurrence->frame_num );
        LOG_INFO("    Width       : " << pOdeOccurrence->source_frame_width );
        LOG_INFO("    Heigh       : " << pOdeOccurrence->source_frame_height );
        LOG_INFO("  Object Data   : ------------------------");
        LOG_INFO("    Class Id    : " << pOdeOccurrence->class_id );
        LOG_INFO("    Tracking Id : " << pOdeOccurrence->object_id);
        LOG_INFO("    Label       : " << pOdeOccurrence->object_label);
        LOG_INFO("    Confidence  : " << pOdeOccurrence->confidence);
        LOG_INFO("    Left        : " << pOdeOccurrence->box.top);
        LOG_INFO("    Top         : " << pOdeOccurrence->box.left);
        LOG_INFO("    Width       : " << pOdeOccurrence->box.width);
        LOG_INFO("    Height      : " << pOdeOccurrence->box.height);
        LOG_INFO("  Min Criteria  : ------------------------");
        LOG_INFO("    Confidence  : " << pOdeOccurrence->min_confidence);
        LOG_INFO("    Frame Count : " << pOdeOccurrence->min_frame_count_n
            << " out of " << pOdeOccurrence->min_frame_count_d);
        LOG_INFO("    Width       : " << pOdeOccurrence->box_criteria.width);
        LOG_INFO("    Height      : " << pOdeOccurrence->box_criteria.height);
    }

}    
