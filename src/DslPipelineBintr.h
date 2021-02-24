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

#ifndef _DSL_PIPELINE_H
#define _DSL_PIPELINE_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBranchBintr.h"
#include "DslSourceBintr.h"
#include "DslDewarperBintr.h"
#include "DslPipelineSourcesBintr.h"
    
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PIPELINE_PTR std::shared_ptr<PipelineBintr>
    #define DSL_PIPELINE_NEW(name) \
        std::shared_ptr<PipelineBintr>(new PipelineBintr(name))

    /**
     * @class PipelineBintr
     * @brief 
     */
    class PipelineBintr : public BranchBintr
    {
    public:
    
        /** 
         * 
         */
        PipelineBintr(const char* pipeline);
        ~PipelineBintr();

        /**
         * @brief Links all Child Elementrs owned by this Source Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();

        /**
         * @brief Attempts to link all and play the Pipeline
         * @return true if able to play, false otherwise
         */
        bool Play();

        /**
         * @brief Attempts to pause the Pipeline, if non-live, and currently playing
         * @return true if able to pause, false otherwise 
         */
        bool Pause();
        
        /**
         * @brief Attempts to stop the Pipeline currently playing
         * @return true if able to stop, false otherwise 
         */
        bool Stop();
        
        /**
         * @brief returns whether the Pipeline has all live sources or not.
         * @return true if all sources are live, false otherwise (default when no sources).
         */
        bool IsLive();

        /**
         * @brief adds a single Source Bintr to this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        bool AddSourceBintr(DSL_BASE_PTR pSourceBintr);

        bool IsSourceBintrChild(DSL_BASE_PTR pSourceBintr);

        /**
         * @brief returns the number of Sources currently in use by
         * this Pipeline
         */
        uint GetNumSourcesInUse()
        {
            if (!m_pPipelineSourcesBintr)
            {
                return 0;
            }
            return m_pPipelineSourcesBintr->GetNumChildren();
        } 
        
        /**
         * @brief removes a single Source Bintr from this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        bool RemoveSourceBintr(DSL_BASE_PTR pSourceBintr);

        /**
         * @brief Gets the current batch settings for the Pipeline's Stream Muxer
         * @param[out] batchSize current batchSize, default == the number of source
         * @param[out] batchTimeout current batch timeout
         * @return true if the batch properties could be read, false otherwise
         */
        void GetStreamMuxBatchProperties(uint* batchSize, uint* batchTimeout);

        /**
         * @brief Sets the current batch settings for the Pipeline's Stream Muxer
         * @param[in] batchSize new batchSize to set, default == the number of sources
         * @param[in] batchTimeout timeout value to set in ms
         * @return true if the batch properties could be set, false otherwise
         */
        bool SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout);

        /**
         * @brief Gets the current dimensions for the Pipeline's Stream Muxer
         * @param[out] width width in pixels for the current setting
         * @param[out] height height in pixels for the curren setting
         * @return true if the output dimensions could be read, false otherwise
         */
        bool GetStreamMuxDimensions(uint* width, uint* height);

        /**
         * @brief Set the dimensions for the Pipeline's Stream Muxer
         * @param width width in pixels to set the streamMux Output
         * @param height height in pixels to set the StreamMux output
         * @return true if the output dimensions could be set, false otherwise
         */
        bool SetStreamMuxDimensions(uint width, uint height);
        
        /**
         * @brief Gets the current setting for the Pipeline's Muxer padding
         * @param enable true if enabled, false otherwise.
         * @return true if the Padding enabled setting could be read, false otherwisee
         */
        bool GetStreamMuxPadding(bool* enabled);

        /**
         * @brief Sets, enables/disables the Pipeline's Stream Muxer padding
         * @param enabled set to true to enable padding
         * @return true if the Padding enabled setting could be set, false otherwise.
         */
        bool SetStreamMuxPadding(bool enabled);
        
        /**
         * @brief Gets the current setting for the Pipeline's StreamMuxer
         * num-surfaces-per-frame seting
         * @param[out] num current setting for the number of surfaces [1..4].
         * @return true if the number setting could be read, false otherwisee
         */
        bool GetStreamMuxNumSurfacesPerFrame(uint* num);

        /**
         * @brief Sets the current setting for the PipelineSourcesBintr's StreamMuxer
         * num-surfaces-per-frame seting
         * @param[in] num new value for the number of surfaces [1..4].
         * @return true if the number setting could be set, false otherwisee
         */
        bool SetStreamMuxNumSurfacesPerFrame(uint num);
        
        /**
         * @brief Gets the current x and y offsets for the Pipeline's XWindow
         * @param[out] xOffset x directional offset from left window edge in pixels
         * @param[out] yOffset y directional offset from top window edge in pixels
         */
        void GetXWindowOffsets(uint* xOffset, uint* yOffset);

        /**
         * @brief Sets the current x and y offsets for the Pipeline's XWindow
         * Note: this function is used for XWindow unit testing only. Offsets are set by the Window Sink
         * @param[in] xOffset x directional offset from left window edge in pixels
         * @param[in] yOffset y directional offset from top window edge in pixels
         */
        void SetXWindowOffsets(uint xOffset, uint yOffset);

        /**
         * @brief Gets the current dimensions for the Pipeline's XWindow
         * @param[out] width width in pixels for the current setting
         * @param[out] height height in pixels for the current setting
         */
        void GetXWindowDimensions(uint* width, uint* height);

        /**
         * @brief Sets the dimensions for the Pipeline's XWindow
         * Note: this function is used for XWindow unit testing only. Dimensions are set by the Window Sink
         * @param width width in pixels to set the XWindow on creation
         * @param height height in pixels to set the XWindow on creation
         * @return true if the output dimensions could be set, false otherwise
         */
        void SetXWindowDimensions(uint width, uint height);
        
        /**
         * @brief Gets the current full-screen-enabled setting for the Pipeline's XWindow
         * @retrun true if full-screen-mode is currently enabled, false otherwise
         */
        bool GetXWindowFullScreenEnabled();
        
        /**
         * @brief Sets the full-screen-enabled setting for the Pipeline's XWindow
         * @param enabled if true, sets the XWindow to full-screen on creation
         * @return true if the full-screen-enabled could be set, false if called after XWindow creation
         */
        bool SetXWindowFullScreenEnabled(bool enabled);

        /**
         * @brief dumps a Pipeline's graph to dot file.
         * @param[in] filename name of the file without extention.
         * The caller is responsible for providing a correctly formated filename
         * The diretory location is specified by the GStreamer debug 
         * environment variable GST_DEBUG_DUMP_DOT_DIR
         */ 
        void DumpToDot(char* filename);
        
        /**
         * @brief dumps a Pipeline's graph to dot file prefixed
         * with the current timestamp.  
         * @param[in] filename name of the file without extention.
         * The caller is responsible for providing a correctly formated filename
         * The diretory location is specified by the GStreamer debug 
         * environment variable GST_DEBUG_DUMP_DOT_DIR
         */ 
        void DumpToDotWithTs(char* filename);
        
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
         * @brief adds a callback to be notified on display/window event [ButtonPress|KeyRelease]
         * @param[in] handler pointer to the client's function to call on XWindow event
         * @param[in] clientData opaque pointer to client data passed into the handler function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool AddXWindowKeyEventHandler(dsl_xwindow_key_event_handler_cb handler, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool RemoveXWindowKeyEventHandler(dsl_xwindow_key_event_handler_cb handler);
            
        /**
         * @brief adds a callback to be notified on display/window event [ButtonPress|KeyRelease]
         * @param[in] handler pointer to the client's function to call on XWindow event
         * @param[in] clientData opaque pointer to client data passed into the handler function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool AddXWindowButtonEventHandler(dsl_xwindow_button_event_handler_cb handler, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool RemoveXWindowButtonEventHandler(dsl_xwindow_button_event_handler_cb handler);
            
        /**
         * @brief adds a callback to be notified on XWindow Delete message event
         * @param[in] handler pointer to the client's function to call on XWindow Delete event
         * @param[in] clientData opaque pointer to client data passed into the handler function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool AddXWindowDeleteEventHandler(dsl_xwindow_delete_event_handler_cb handler, void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        bool RemoveXWindowDeleteEventHandler(dsl_xwindow_delete_event_handler_cb handler);
            
        /**
         * @brief handles incoming Message Packets received
         * by the bus watcher callback function
         * @return true if the message was handled correctly 
         */
        bool HandleBusWatchMessage(GstMessage* pMessage);

        /**
         * @brief handles incoming sync messages
         * @param[in] message incoming message to process
         * @return [GST_BUS_PASS|GST_BUS_FAIL]
         */
        GstBusSyncReply HandleBusSyncMessage(GstMessage* pMessage);

        /**
         * @brief handles incoming window KEY & BUTTON events by calling
         * all client installed event handlers for each queued event.
         */
        void HandleXWindowEvents();

        bool CreateXWindow();
        
        /**
         * @brief returns a handle to this PipelineBintr's XWindow
         * @return XWindow handle, NULL untill created
         */
        const Window GetXWindow()
        {
            LOG_FUNC();
            
            return m_pXWindow;
        }
        
        bool ClearXWindow();
        
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

        bool HandleStateChanged(GstMessage* pMessage);
        
        void HandleEosMessage(GstMessage* pMessage);
        
        void HandleErrorMessage(GstMessage* pMessage);
        
        /**
         * @brief parent bin for all Source bins in this Pipeline
         */
        DSL_PIPELINE_SOURCES_PTR m_pPipelineSourcesBintr;
        
        /**
         * @brief x-offset setting to use on XWindow creation in pixels
         */
        uint m_xWindowOffsetX;
        
        /**
         * @brief y-offset setting to use on XWindow creation in pixels
         */
        uint m_xWindowOffsetY;
        
        /**
         * @brief width setting to use on XWindow creation in pixels
         */
        uint m_xWindowWidth;
        
        /**
         * @brief height setting to use on XWindow creation in pixels
         */
        uint m_xWindowHeight;
        
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
         * @brief map of all currently registered XWindow-key-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_xwindow_key_event_handler_cb, void*>m_xWindowKeyEventHandlers;

        /**
         * @brief map of all currently registered XWindow-button-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_xwindow_button_event_handler_cb, void*>m_xWindowButtonEventHandlers;

        /**
         * @brief map of all currently registered XWindow-delete-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_xwindow_delete_event_handler_cb, void*>m_xWindowDeleteEventHandlers;

        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_busWatchMutex;

        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_busSyncMutex;

        /**
         * @brief Bus used to receive GstMessage packets.
         */
        GstBus* m_pGstBus;
        
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
         * @brief a single display for each Pipeline
        */
        Display* m_pXDisplay;

        /**
         * @brief mutex for display thread
        */
        GMutex m_displayMutex;
                
        /**
         * @brief handle to X Window
         */
        Window m_pXWindow;
        
        /**
         * @brief handle to the X Window event thread, 
         * active for the life of the Pipeline
        */
        GThread* m_pXWindowEventThread;        
        
        /**
         * @brief if true, the Pipeline will set its XWindow to full-screen if one is created
         * A Pipeline requires a Window Sink to create an XWindow on Play
         */
        bool m_xWindowfullScreenEnabled;
        
        /**
         * @brief maps a GstMessage constant value to a string for logging
         */
        std::map<GstMessageType, std::string> m_mapMessageTypes;

        /**
         * @brief initializes the "constant-value-to-string" maps
         */
        void _initMaps();
        
        
    }; // Pipeline
    
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
     * @brief 
     * @param[in] bus instance pointer
     * @param[in] message incoming message packet to process
     * @param[in] pData pipeline instance pointer
     * @return [GST_BUS_PASS|GST_BUS_FAIL]
     */
    static GstBusSyncReply bus_sync_handler(
        GstBus* bus, GstMessage* pMessage, gpointer pData);

    /**
     * @brief Timer thread Notification Handler to invoke a Pipelines NotifyErrorMessageHandlers() function
     * @param pPipeline shared pointer to the Pipeline that started the timer so that clients can be notified
     * in the timer's experation callback running from the main the loop, instead of the bus-watch callback
     * @return false always to self destroy the one-shot timer.
     */
    static int ErrorMessageHandlersNotificationHandler(gpointer pPipeline);
    
    static gpointer XWindowEventThread(gpointer pData);

    
} // Namespace

#endif // _DSL_PIPELINE_H

