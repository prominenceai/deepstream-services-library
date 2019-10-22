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
#include "DslSinkBintr.h"
#include "DslOsdBintr.h"
#include "DslGieBintr.h"
#include "DslDisplayBintr.h"
    
namespace DSL 
{
    /**
     * @class ProcessBintr
     * @brief 
     */
    class ProcessBintr : public Bintr
    {
    public:
    
        /** 
         * 
         */
        ProcessBintr(const char* name);
        ~ProcessBintr();

        /**
         * @brief Adds a Sink Bintr to this Process Bintr
         * @param[in] pSinkBintr
         */
        void AddSinkBintr(std::shared_ptr<Bintr> pSinkBintr);

        /**
         * @brief 
         * @param[in] pOsdBintr
         */
        void AddOsdBintr(std::shared_ptr<Bintr> pOsdBintr);
        
        /**
         * @brief 
         */
        void AddSinkGhostPad();

        
    private:
    
        /**
         * @brief one or more Sinks for this Process bintr
         */
        std::shared_ptr<SinksBintr> m_pSinksBintr;
        
        /**
         * @brief optional OSD for this Process bintr
         */
        std::shared_ptr<OsdBintr> m_pOsdBintr;
        
    };

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

        void AddChild(std::shared_ptr<Bintr> pChildBintr);
        
        bool Pause();
        bool Play();
        
        /**
         * @brief adds a single Source Bintr to this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        void AddSourceBintr(std::shared_ptr<Bintr> pSourceBintr);

        /**
         * @brief removes a single Source Bintr from this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        void RemoveSourceBintr(std::shared_ptr<Bintr> pSourceBintr);

        /**
         * @brief adds a single Sink Bintr to this Pipeline 
         * @param[in] pSinkBintr shared pointer to Sink Bintr to add
         */
        void AddSinkBintr(std::shared_ptr<Bintr> pSinkBintr)
        {
            m_pProcessBintr->AddSinkBintr(pSinkBintr);
        }

        /**
         * @brief adds a single OSD Bintr to this Pipeline 
         * @param[in] pOsdBintr shared pointer to OSD Bintr to add
         */
        void AddOsdBintr(std::shared_ptr<Bintr> pOsdBintr)
        {
            m_pProcessBintr->AddOsdBintr(pOsdBintr);
        }
        
        /**
         * @brief adds a single GIE Bintr to this Pipeline 
         * @param[in] pGieBintr shared pointer to GIE Bintr to add
         */
        void AddPrimaryGieBintr(std::shared_ptr<Bintr> pGieBintr);

        /**
         * @brief adds a single Display Bintr to this Pipeline 
         * @param[in] pDisplayBintr shared pointer to Display Bintr to add
         */
        void AddDisplayBintr(std::shared_ptr<Bintr> pDisplayBintr);
        
        /**
         * @brief 
         * @param[in] m_areSourcesLive
         * @param[in] batchSize
         * @param[in] batchTimeout
         * @param[in] width
         * @param[in] height
         */
        void SetStreamMuxProperties(gboolean areSourcesLive, guint batchSize, guint batchTimeout, 
            guint width, guint height)
        {
            m_pSourcesBintr->SetStreamMuxProperties(areSourcesLive, 
                batchSize, batchTimeout, width, height);
        }
        
        void LinkComponents();
        
        void UnlinkComponents();
        
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
    
    private:

        /**
         * @brief GStreamer Pipeline contained by this pipeline bintr
         */
        GstElement* m_pGstPipeline; 

        /**
         * @brief parent bin for all Source bins in this Pipeline
         */
        std::shared_ptr<SourcesBintr> m_pSourcesBintr;
        
        /**
         * @brief processing bin for all Sink and OSD bins in this Pipeline
         */
        std::shared_ptr<ProcessBintr> m_pProcessBintr;
        
        /**
         * @brief the one and only Display for this Pipeline
         */
        std::shared_ptr<Bintr> m_pPrimaryGieBintr;
        
        /**
         * @brief the one and only Display for this Pipeline
         */
        std::shared_ptr<Bintr> m_pDisplayBintr;
                
        /**
         * @brief true if the components in the Pipeline are in a state 
         * of linked, false if unlinked. 
         */
        bool m_areComponentsLinked;
        
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
         * @brief mutex to protect critical pipeline code
         */
        GMutex m_pipelineMutex;

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
         * @brief maps a GstMessage constant value to a string for logging
         */
        std::map<GstMessageType, std::string> m_mapMessageTypes;

        bool HandleStateChanged(GstMessage* pMessage);
        
        void _handleErrorMessage(GstMessage* pMessage);        
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
        
    
} // Namespace

#endif // _DSL_PIPELINE_H

