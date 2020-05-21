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

#ifndef _DSL_EVENT_ACTION_H
#define _DSL_EVENT_ACTION_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */

    #define DSL_EVENT_ACTION_PTR std::shared_ptr<EventAction>

    #define DSL_EVENT_ACTION_CALLBACK_PTR std::shared_ptr<CallbackEventAction>
    #define DSL_EVENT_ACTION_CALLBACK_NEW(name) \
        std::shared_ptr<CallbackEventAction>(new CallbackEventAction(name))
        
    #define DSL_EVENT_ACTION_DISPLAY_PTR std::shared_ptr<DisplayEventAction>
    #define DSL_EVENT_ACTION_DISPLAY_NEW(name) \
        std::shared_ptr<DisplayEventAction>(new DisplayEventAction(name))
        
    #define DSL_EVENT_ACTION_LOG_PTR std::shared_ptr<LogEventAction>
    #define DSL_EVENT_ACTION_LOG_NEW(name) \
        std::shared_ptr<LogEventAction>(new LogEventAction(name))
        
        
    class EventAction : public Base
    {
    public: 
    
        EventAction(const char* name);

        ~EventAction();
        
        virtual void HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta) = 0;
        
    private:

    };

    class CallbackEventAction : public EventAction
    {
    public:
    
        CallbackEventAction(const char* name);
        
        ~CallbackEventAction();

        void HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:

    };
    
    class DisplayEventAction : public EventAction
    {
    public:
    
        DisplayEventAction(const char* name);
        
        ~DisplayEventAction();
        
        void HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
    };
    class LogEventAction : public EventAction
    {
    public:
    
        LogEventAction(const char* name);
        
        ~LogEventAction();
        
        void HandleOccurrence(DSL_BASE_PTR pEvent, uint64_t eventId, 
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
    };

}


#endif // _DSL_EVENT_ACTION_H
