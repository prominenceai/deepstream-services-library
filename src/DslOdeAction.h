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

#ifndef _DSL_ODE_ACTION_H
#define _DSL_ODE_ACTION_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBase.h"
//#include "DslOdeOccurrence.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_ODE_ACTION_PTR std::shared_ptr<OdeAction>

    #define DSL_ODE_ACTION_CALLBACK_PTR std::shared_ptr<CallbackOdeAction>
    #define DSL_ODE_ACTION_CALLBACK_NEW(name, handler, userData) \
        std::shared_ptr<CallbackOdeAction>(new CallbackOdeAction(name, handler, userData))
        
    #define DSL_ODE_ACTION_DISPLAY_PTR std::shared_ptr<DisplayOdeAction>
    #define DSL_ODE_ACTION_DISPLAY_NEW(name) \
        std::shared_ptr<DisplayOdeAction>(new DisplayOdeAction(name))
        
    #define DSL_ODE_ACTION_LOG_PTR std::shared_ptr<LogOdeAction>
    #define DSL_ODE_ACTION_LOG_NEW(name) \
        std::shared_ptr<LogOdeAction>(new LogOdeAction(name))
        
    #define DSL_ODE_ACTION_PRINT_PTR std::shared_ptr<PrintOdeAction>
    #define DSL_ODE_ACTION_PRINT_NEW(name) \
        std::shared_ptr<PrintOdeAction>(new PrintOdeAction(name))
        
    #define DSL_ODE_ACTION_REDACT_PTR std::shared_ptr<RedactOdeAction>
    #define DSL_ODE_ACTION_REDACT_NEW(name, red, green, blue, alpha) \
        std::shared_ptr<RedactOdeAction>(new RedactOdeAction(name, red, green, blue, alpha))
        
    #define DSL_ODE_ACTION_QUEUE_PTR std::shared_ptr<QueueOdeAction>
    #define DSL_ODE_ACTION_QUEUE_NEW(name, maxSize) \
        std::shared_ptr<QueueOdeAction>(new QueueOdeAction(name, maxSize))
        
        
    class OdeAction : public Base
    {
    public: 
    
        OdeAction(const char* name);

        ~OdeAction();
        
        virtual void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta) = 0;
        
    private:

    };

    /**
     * @class CallbackOdeAction
     * @brief ODE Callback Ode Action class
     */
    class CallbackOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Callback Action class
         * @param handler client callback function to call on ODE
         */
        CallbackOdeAction(const char* name, dsl_ode_occurrence_handler_cb clientHandler, void* userData);
        
        /**
         * @brief dtor for the ODE Callback Action class
         */
        ~CallbackOdeAction();

        /**
         * @brief Handles the ODE occurrence by calling the client handler
         * @param 
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
    private:
    
        /**
         * @brief Client Callback function to call on ODE occurrence
         */
        dsl_ode_occurrence_handler_cb m_clientHandler;
        
        /**
         * @brief pointer to client's data returned on callback
         */ 
        void* m_clientData;

    };
    
    /**
     * @class DisplayOdeAction
     * @brief ODE Display Ode Action class
     */
    class DisplayOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Display Action class
         */
        DisplayOdeAction(const char* name);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~DisplayOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by adding display info
         * using OSD text overlay
         * @param pOdeOccurrence ODE occurrence data
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
    };

    /**
     * @class LogOdeAction
     * @brief ODE Log Ode Action class
     */
    class LogOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Log Action class
         */
        LogOdeAction(const char* name);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~LogOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by adding/calling LOG_INFO 
         * with the ODE occurrence data data
         * @param pOdeOccurrence ODE occurrence data
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
    };
        
    /**
     * @class PrintOdeAction
     * @brief ODE Print Ode Action class
     */
    class PrintOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Print Action class
         */
        PrintOdeAction(const char* name);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~PrintOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by printing the  
         * the occurrence data to the console
         * @param pOdeOccurrence ODE occurrence data
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);

    private:
    
    };
        
    /**
     * @class RedactOdeAction
     * @brief ODE Redact Ode Action class
     */
    class RedactOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Log Action class
         * @param[in] red red level for the redaction background color [0..1]
         * @param[in] blue blue level for the redaction background color [0..1]
         * @param[in] green green level for the redaction background color [0..1]
         * @param[in] alpha alpha level for the redaction background color [0..1]
         */
        RedactOdeAction(const char* name, double red, double green, double blue, double alpha);
        
        /**
         * @brief dtor for the ODE Display Action class
         */
        ~RedactOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by redacting the object 
         * with the boox coordinates in the ODE occurrence data
         * @param pOdeOccurrence ODE occurrence data
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
            
    private:
    
        NvOSD_ColorParams m_backgroundColor;
    
    };

    /**
     * @class QueueOdeAction
     * @brief ODE Queue Ode Action class
     */
    class QueueOdeAction : public OdeAction
    {
    public:
    
        /**
         * @brief ctor for the ODE Queue Action class
         * @param handler client callback function to call on ODE
         */
        QueueOdeAction(const char* name, uint maxSize);
        
        /**
         * @brief dtor for the ODE Queue Action class
         */
        ~QueueOdeAction();
        
        /**
         * @brief Handles the ODE occurrence by queuing the data 
         * for subsequent, asynchronous reading/polling
         * @param pOdeOccurrence ODE occurrence data
         */
        void HandleOccurrence(DSL_BASE_PTR pBaseType,
            NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta);
        
        /**
         * @brief Dequeues the oldest OSD occurrence from the queue
         * @return shared pointer to oldest OSD occurrence
         */
//        DSL_ODE_OCCURRENCE_PTR Dequeue();
        
        /**
         * @brief returns the maximum size the queue is allowed to grow
         * @return the current max queue size in use
         */
        uint GetMaxSize();
        
        /**
         * @brief returns the current size of the ODE occurence queue
         * @return current size of the ODE occurence queue
         */
        uint GetCurrentSize();
        
    private:
    
        /**
         * @brief queue of shared pointers to ODE occurrence data
         */
 //       std::queue<DSL_ODE_OCCURRENCE_PTR> m_odeQueue;
        
        /**
         * @brief maximum size the queue is allowed to grow before popping events
         */
        uint m_maxSize;
    
    };
}


#endif // _DSL_ODE_ACTION_H
