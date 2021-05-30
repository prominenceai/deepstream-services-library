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

#ifndef _DSL_PIPELINE_XWIN_MGR_H
#define _DSL_PIPELINE_XWIN_MGR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslSourceBintr.h"
#include "DslSinkBintr.h"

namespace DSL
{

    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PIPELINE_XWIN_MGR_PTR std::shared_ptr<PipelineXWinMgr>
    
    class PipelineXWinMgr
    {
    public: 
    
        PipelineXWinMgr(const GstObject* pGstPipeline);

        ~PipelineXWinMgr();
        
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
         * @param width width in pixels to set the XWindow
         * @param height height in pixels to set the XWindow
         * @return true if the XWindow dimensions could be set, false otherwise
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
        
        void PrepareXWindow(GstMessage* pMessage);

        /**
         * @brief handles incoming window KEY & BUTTON events by calling
         * all client installed event handlers for each queued event.
         */
        void HandleXWindowEvents();

        bool CreateXWindow();
        
        /**
         * @brief queries the player to determine if it owns an xwindow
         * @return true if the Player has ownership of an xwindow, false otherwise
         */
        bool OwnsXWindow();
        
        /**
         * @brief returns a handle to this PipelineBintr's XWindow
         * @return XWindow handle, NULL untill created
         */
        Window GetXWindow();
		
		/**
		 * @brief Sets the PipelineBintr's XWindow handle. The Pipeline
		 * must be in an unlinked state to change XWindow handles. 
		 * @return true on successful clear, false otherwise
		 */
		bool SetXWindow(Window xWindow);
        
		/**
		 * @brief Clears the PipelineBintr's XWindow buffer
		 * @return true on successful clear, false otherwise
		 */
        bool ClearXWindow();
        
		/**
		 * @brief Destroys the PipelineBintr's XWindow
		 * @return true on successful destruction, false otherwise
		 */
        bool DestroyXWindow();
        
        /**
         * @brief handles incoming sync messages
         * @param[in] message incoming message to process
         * @return [GST_BUS_PASS|GST_BUS_FAIL]
         */
        GstBusSyncReply HandleBusSyncMessage(GstMessage* pMessage);
        
    private:

        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_busSyncMutex;
    
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
		 * @brief flag to determine if the XWindow was created or provided by the client.
		 * The Pipeline needs to delete the XWindow if created, but not the client's
		 */
		bool m_pXWindowCreated;
        
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
    };
    
    static gpointer XWindowEventThread(gpointer pPipeline);

    /**
     * @brief 
     * @param[in] bus instance pointer
     * @param[in] message incoming message packet to process
     * @param[in] pData pipeline instance pointer
     * @return [GST_BUS_PASS|GST_BUS_FAIL]
     */
    static GstBusSyncReply bus_sync_handler(
        GstBus* bus, GstMessage* pMessage, gpointer pData);
}

#endif //  DSL_PIPELINE_XWIN_MGR_H
