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

#include "DslApi.h"
#include "DslSourceBintr.h"
#include "DslDewarperBintr.h"
#include "DslGieBintr.h"
#include "DslTrackerBintr.h"
#include "DslOsdBintr.h"
#include "DslDisplayBintr.h"
#include "DslPipelineSourcesBintr.h"
#include "DslPipelineSGiesBintr.h"
#include "DslPipelineSinksBintr.h"
#include "DslSinkBintr.h"
    
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
    class PipelineBintr : public Bintr
    {
    public:
    
        /** 
         * 
         */
        PipelineBintr(const char* pipeline);
        ~PipelineBintr();

        bool Play();
        
        bool Stop();

        /**
         * @brief adds a single Source Bintr to this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        bool AddSourceBintr(DSL_NODETR_PTR pSourceBintr);

        bool IsSourceBintrChild(DSL_NODETR_PTR pSourceBintr);

        /**
         * @brief returns the number of Sources currently in use by
         * this Pipeline
         */
        uint GetNumSourceInUse()
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
        bool RemoveSourceBintr(DSL_NODETR_PTR pSourceBintr);

        /**
         * @brief adds a single DewarperBintr to this Pipeline 
         * @param[in] pDewarperBintr shared pointer to DewarperBintr to add
         */
        bool AddDewarperBintr(DSL_NODETR_PTR pDewarperBintr);

        /**
         * @brief adds a single GIE Bintr to this Pipeline 
         * @param[in] pGieBintr shared pointer to GIE Bintr to add
         */
        bool AddPrimaryGieBintr(DSL_NODETR_PTR pPrmaryGieBintr);

        /**
         * @brief adds a single Secondary GIE Nodetr to this Pipeline 
         * @param[in] pSecondaryGieNodetr shared pointer to SGIE Nodetr to add
         */
        bool AddSecondaryGieBintr(DSL_NODETR_PTR pSecondaryGieBintr);

        /**
         * @brief adds a single Display Bintr to this Pipeline 
         * @param[in] pDisplayBintr shared pointer to Display Bintr to add
         */
        bool AddTrackerBintr(DSL_NODETR_PTR pTrackerBintr);
        
        /**
         * @brief adds a single Display Bintr to this Pipeline 
         * @param[in] pDisplayBintr shared pointer to Display Bintr to add
         */
        bool AddDisplayBintr(DSL_NODETR_PTR pDisplayBintr);
        
        /**
         * @brief adds a single OSD Bintr to this Pipeline 
         * @param[in] pOsdBintr shared pointer to OSD Bintr to add
         */
        bool AddOsdBintr(DSL_NODETR_PTR pOsdBintr);
        
        /**
         * @brief adds a single Sink Bintr to this Pipeline 
         * @param[in] pSinkBintr shared pointer to Sink Bintr to add
         */
        bool AddSinkBintr(DSL_NODETR_PTR pSinkBintr);

        bool IsSinkBintrChild(DSL_NODETR_PTR pSinkBintr);

        /**
         * @brief removes a single Sink Bintr from this Pipeline 
         * @param[in] pSinkBintr shared pointer to Sink Bintr to add
         */
        bool RemoveSinkBintr(DSL_NODETR_PTR pSinkBintr);

        /**
         * @brief Gets the current batch settings for the Pipeline's Stream Muxer
         * @param[out] batchSize current batchSize, default == the number of source
         * @param[out] batchTimeout current batch timeout
         * @return true if the batch properties could be read, false otherwise
         */
        bool GetStreamMuxBatchProperties(uint* batchSize, uint* batchTimeout);

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
         * @return true if the Padding enable setting could be set, false otherwise.
         */
        bool SetStreamMuxPadding(bool enabled);
        
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
         * @param[in] userdata opaque pointer to client data passed into the listner function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        DslReturnType AddStateChangeListener(dsl_state_change_listener_cb listener, void* userdata);

        /**
         * @brief called to determine if a CB is currently a child (in-ues) by the Pipeline
         * @param listener calback to check if in use
         * @return true if currently a child in use
         */
        bool IsChildStateChangeListener(dsl_state_change_listener_cb listener);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        DslReturnType RemoveStateChangeListener(dsl_state_change_listener_cb listener);
            
        /**
         * @brief adds a callback to be notified on display/window event [ButtonPress|KeyRelease]
         * @param[in] handler pointer to the client's function to call on Display event
         * @param[in] userdata opaque pointer to client data passed into the handler function.
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        DslReturnType AddDisplayEventHandler(dsl_display_event_handler_cb handler, void* userdata);

        /**
         * @brief called to determine if a CB is currently a child (in-ues) by the Pipeline
         * @param handler calback to check if in use
         * @return true if currently a child in use
         */
        bool IsChildDisplayEventHandler(dsl_display_event_handler_cb handler);

        /**
         * @brief removes a previously added callback
         * @param[in] handler pointer to the client's function to remove
         * @return DSL_RESULT_PIPELINE_RESULT
         */
        DslReturnType RemoveDisplayEventHandler(dsl_display_event_handler_cb handler);
            
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
        
        bool LinkAll();
        
        void UnlinkAll();
        
        /**
         * @brief returns a handle to this PipelineBintr's XWindow
         * @return XWindow handle, NULL untill created
         */
        const Window GetXWindow()
        {
            LOG_FUNC();
            
            return m_pXWindow;
        }

    private:

        std::vector<DSL_BINTR_PTR> m_linkedComponents;
        
        /**
         * @brief parent bin for all Source bins in this Pipeline
         */
        DSL_PIPELINE_SOURCES_PTR m_pPipelineSourcesBintr;
        
        /**
         * @brief optional, one at most Dewarper for this Pipeline
         */
        DSL_DEWARPER_PTR m_pDewarperBintr;
        
        /**
         * @brief optional, one at most Primary GIE for this Pipeline
         */
        DSL_PRIMARY_GIE_PTR m_pPrimaryGieBintr;
        
        /**
         * @brief optional, one or more Secondary GIEs for this Pipeline
         */
        DSL_PIPELINE_SGIES_PTR m_pSecondaryGiesBintr;
        
        /**
         * @brief optional, one at most Tracker for this Pipeline
         */
        DSL_TRACKER_PTR m_pTrackerBintr;

        /**
         * @brief optional, one at most OSD for this Pipeline
         */
        DSL_OSD_PTR m_pOsdBintr;
        
        /**
         * @brief optional one and only optional Tiled Display for this Pipeline
         */
        DSL_DISPLAY_PTR m_pDisplayBintr;
                        
        /**
         * @brief parent bin for all Sink bins in this Pipeline
         */
        DSL_PIPELINE_SINKS_PTR m_pPipelineSinksBintr;
        
        /**
         * @brief map of all currently registered state-change-listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_state_change_listener_cb, void*>m_stateChangeListeners;
        
        /**
         * @brief map of all currently registered display-event-handlers
         * callback functions mapped with the user provided data
         */
        std::map<dsl_display_event_handler_cb, void*>m_displayEventHandlers;

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
         * @brief maps a GstMessage constant value to a string for logging
         */
        std::map<GstMessageType, std::string> m_mapMessageTypes;

        bool HandleStateChanged(GstMessage* pMessage);
        
        void HandleErrorMessage(GstMessage* pMessage);

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

    static gpointer XWindowEventThread(gpointer pData);

    
} // Namespace

#endif // _DSL_PIPELINE_H

