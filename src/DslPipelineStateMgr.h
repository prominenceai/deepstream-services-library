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
    
        PipelineStateMgr(GstObject* pGstPipeline);

        ~PipelineStateMgr();

        /**
         * @brief Creates a new g_main_context and g_main_loop for the Pipeline. 
         * @return true on successful creation, false otherwise.
         */
        bool NewMainLoop();
        
        /**
         * @brief Runs and joins with the Pipeline's main-loop. 
         * @return true on successful return from the main-loop, false otherwise.
         * Note: the service will block on success, but return immediately on failure.
         */
        bool RunMainLoop();
        
        /**
         * @brief Quits the Pipeline's mainloop causing RunMainLoop() to return.
         * @return true on successful quit, false otherwise.
         */
        bool QuitMainLoop();
        
        /**
         * @brief Deletes the Pipeline's own g_main_loop and g_main_context.
         * @return true on successful delete, false otherwise.
         */
        bool DeleteMainLoop();
        
        virtual void HandleStop() = 0;

        /**
         * @brief Adds a callback to be notified on change of Pipeline state
         * @param[in] listener pointer to the client's function to call on state change
         * @param[in] clientData opaque pointer to client data passed into the listner function.
         * @return true on successful listener add, false otherwise.
         */
        bool AddStateChangeListener(dsl_state_change_listener_cb listener, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful listener remove, false otherwise.
         */
        bool RemoveStateChangeListener(dsl_state_change_listener_cb listener);
            
        /**
         * @brief adds a callback to be notified on change of Pipeline state
         * @param[in] listener pointer to the client's function to call on state change
         * @param[in] clientData opaque pointer to client data passed into the listner function.
         * @return true on successful listener add, false otherwise.
         */
        bool AddEosListener(dsl_eos_listener_cb listener, void* clientData);

        /**
         * @brief queries if a callback function is currently added as an EOS listener
         * @param[in] listener pointer to the client's function to query for
         * @return true if the calback is an EOS listener, false otherwise.
         */
        bool IsEosListener(dsl_eos_listener_cb listener);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful listener remove, false otherwise.
         */
        bool RemoveEosListener(dsl_eos_listener_cb listener);
            
        /**
         * @brief adds a callback to be notified on the event an error message is recieved on the bus
         * @param[in] handler pointer to the client's function to call on error message
         * @param[in] clientData opaque pointer to client data passed into the handler function.
         * @return true on successful handler add, false otherwise.
         */
        bool AddErrorMessageHandler(dsl_error_message_handler_cb handler, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return true on successful handler remove, false otherwise.
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

    protected:

        /**
         * pointer to the Pipelines GST Bus
         */
        GstBus* m_pGstBus;

        /**
         * @brief Mutex to prevent bus-watch callback re-entry
         */
        GMutex m_busWatchMutex;

        /**
         * @brief set to true by the bus-watch function on EOS
         * and cleared by the Pipeline on transition to NULL state.
         */
        bool m_eosFlag;

        /**
         * @brief Pointer to the Pipelines own g_main_context if one has 
         * been created, NULL otherwise.
         */
        GMainContext* m_pMainContext;
        
        /**
         * @brief Pointer to the Pipeline's own g_main_loop if one has 
         * been created, NULL otherwise.
         */
        GMainLoop* m_pMainLoop;
    
    private:

        /**
         * @brief Private helper function to handle a Pipeline state-change message.
         * @param[in] pointer to the state-change message.
         */
        bool HandleStateChanged(GstMessage* pMessage);
        
        /**
         * @brief private helper function to handle a Pipeline end-of-stream (EOS) message.
         * @param[in] pointer to the eos message to handle.
         */
        void HandleEosMessage(GstMessage* pMessage);
        
        /**
         * @brief private helper function to handle a Pipeline error message.
         * @param[in] pointer to the error message to handle.
         */
        void HandleErrorMessage(GstMessage* pMessage);
        
        /**
         * @brief private helper function to handle an Application message
         * @param[in] pointer to the Application message to handle.
         */
        void HandleApplicationMessage(GstMessage* pMessage);
    
        /**
         * GST Pipeline Object, provided on construction by the derived parent Pipeline.
         */
        GstObject* m_pGstPipeline;
        
        /**
         * @brief unique id of the installed Bus Watch function. This handle is
         * used by default, until if/when the client calls on the Pipeline
         * to create its own main-context and main-loop. At that point the
         * default bus-watch will be removed and a new bus-watch created as a 
         * GSource to be attached to the new Pipeline's main-loop.
         */
        guint m_busWatchId;
        
        /**
         * @brief point to a bus-watch created as a GSource to be attached
         * to the Pipelines main-loop if one is created.
         */
        GSource* m_pBusWatch;

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
     * @param[in] pPipeline pipeline instance pointer
     * @return true if the message was handled correctly 
     */
    static gboolean bus_watch(
        GstBus* bus, GstMessage* pMessage, gpointer pPipeline);

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
