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

    EventActionLog::EventActionLog(const char* name)
        : EventAction(name)
    {
        LOG_FUNC();

    }

    EventActionLog::~EventActionLog()
    {
        LOG_FUNC();

    }

    void EventActionLog::HandleOccurrence(const std::string& eventName, uint eventId, NvDsObjectMeta* pObjectMeta)
    {
        LOG_FUNC();

        LOG_INFO("Event Name   : " << eventName);
        LOG_INFO("  event_id   : " << eventId);
        LOG_INFO("  obj_label  : " << pObjectMeta->obj_label);
        LOG_INFO("  class_id   : " << pObjectMeta->class_id);
        LOG_INFO("  object_id  : " << pObjectMeta->object_id);
        LOG_INFO("  confidence : " << pObjectMeta->confidence);
        LOG_INFO("  left       : " << pObjectMeta->rect_params.left);
        LOG_INFO("  top        : " << pObjectMeta->rect_params.top);
        LOG_INFO("  width      : " << pObjectMeta->rect_params.width);
        LOG_INFO("  height     : " << pObjectMeta->rect_params.height);
    }

    // ********************************************************************

    EventActionDisplay::EventActionDisplay(const char* name)
        : EventAction(name)
    {
        LOG_FUNC();

    }

    EventActionDisplay::~EventActionDisplay()
    {
        LOG_FUNC();

    }

    void EventActionDisplay::HandleOccurrence(const std::string& eventName, uint eventId, NvDsObjectMeta* pObjectMeta)
    {
        LOG_FUNC();

    }

    // ********************************************************************

    EventActionCallback::EventActionCallback(const char* name)
        : EventAction(name)
    {
        LOG_FUNC();

    }

    EventActionCallback::~EventActionCallback()
    {
        LOG_FUNC();

    }
    
    void EventActionCallback::HandleOccurrence(const std::string& eventName, uint eventId, NvDsObjectMeta* pObjectMeta)
    {
        LOG_FUNC();

    }
}    
