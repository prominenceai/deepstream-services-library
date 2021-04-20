/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#ifndef _DSL_PIPELINE_BUS_MGR_H
#define _DSL_PIPELINE_BUS_MGR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"

namespace DSL
{

    class PipelineStateMgr
    {
    public: 
    
        PipelineStateMgr(const GstObject* pGstPipeline);

        ~PipelineStateMgr();

        /**
         * @brief adds a callback to be notified on change of Pipeline state
         * @param[in] listener pointer to the client's function to call on state change
         * @param[in] clientData opaque pointer to client data passed into the listner function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool AddStateChangeListener(dsl_state_change_listener_cb listener, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool RemoveStateChangeListener(dsl_state_change_listener_cb listener);
            
        /**
         * @brief adds a callback to be notified on change of Pipeline state
         * @param[in] listener pointer to the client's function to call on state change
         * @param[in] clientData opaque pointer to client data passed into the listner function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool AddEosListener(dsl_eos_listener_cb listener, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool RemoveEosListener(dsl_eos_listener_cb listener);
            
        /**
         * @brief adds a callback to be notified on the event an error message is recieved on the bus
         * @param[in] handler pointer to the client's function to call on error message
         * @param[in] clientData opaque pointer to client data passed into the handler function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool AddErrorMessageHandler(dsl_error_message_handler_cb handler, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool RemoveErrorMessageHandler(dsl_error_message_handler_cb handler);
            
        /**
         * @brief handles incoming Message Packets received
         * by the bus watcher callback function
         * @return true if the message was handled correctly 
         */
        bool HandleBusWatchMessage(GstMessage* pMessage);

        /**
         * @brief Gets the last error message recieved by the bus watch error handler
         * @param[out] source name of gst object that sent the error mess
         * @param[out] message error/warning message sent by the source
         */
        void GetLastErrorMessage(std::wstring& source, std::wstring& message);

        /**
         * @brief Sets the last error message recieved by the bus watch error handler.
         * this service will schedule a timer thread to notify all client handlers. 
         * @param[out] source name of gst object that sent the error mess
         * @param[out] message error/warning message sent by the source
         */
        void SetLastErrorMessage(std::wstring& source, std::wstring& message);
        
        /**
         * @brief Timer experation callback function to notify all error-message-handlers of a new error
         * recieved by the bus-watch. Allows notifications to be sent out from the main-loop context.
         * @return false always to destroy the one-shot timer calling this callback. 
         */
        int NotifyErrorMessageHandlers();
    
    private:
    
        /**
         * GST Pipeline Object, provided on construction by the derived parent Pipeline
         */
        const GstObject* m_pGstPipeline;
        
        bool HandleStateChanged(GstMessage* pMessage);
        
        void HandleEosMessage(GstMessage* pMessage);
        
        void HandleErrorMessage(GstMessage* pMessage);
    
        /**
         * @brief map of all currently registered state-change-listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_state_change_listener_cb, void*>m_stateChangeListeners;
        
        /**
         * @brief map of all currently registered end-of-stream-listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_eos_listener_cb, void*>m_eosListeners;
        
        /**
         * @brief map of all currently registered error-message-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_error_message_handler_cb, void*>m_errorMessageHandlers;

        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_busWatchMutex;

        /**
         * @brief handle to the installed Bus Watch function.
         */
        guint m_gstBusWatch;
        
        /**
         * @brief mutex to protect multiple threads from accessing/updating last error message information
         */
        GMutex m_lastErrorMutex;
        
        /**
         * @brief timer used to execute the error notification thread
         */
        uint m_errorNotificationTimerId;
        
        /**
         * @brief name of the gst object that was the source of the last error message
         * Note: in wchar format for client handlers
         */
        std::wstring m_lastErrorSource;

        /**
         * @brief the last error message received by the bus watch. 
         * Note: in wchar format for client handlers
         */
        std::wstring m_lastErrorMessage;
        
        /**
         * @brief maps a GstState constant value to a string for logging
         */
        std::map<GstState, std::string> m_mapPipelineStates;
        
        /**
         * @brief initializes the "constant-value-to-string" maps
         */
        void _initMaps();
    };

    /**
     * @brief callback function to watch a pipeline's bus for messages
     * @param[in] bus instance pointer
     * @param[in] message incoming message packet to process
     * @param[in] pData pipeline instance pointer
     * @return true if the message was handled correctly 
     */
    static gboolean bus_watch(
        GstBus* bus, GstMessage* pMessage, gpointer pData);

    /**
     * @brief Timer thread Notification Handler to invoke a Pipelines 
     * NotifyErrorMessageHandlers() function
     * @param pPipeline shared pointer to the Pipeline that started the timer so that 
     * clients can be notified in the timer's experation callback running from the 
     * main the loop, instead of the bus-watch callback
     * @return false always to self destroy the one-shot timer.
     */
    static int ErrorMessageHandlersNotificationHandler(gpointer pPipeline);
}

#endif //  DSL_PIPELINE_BUS_MGR_H
