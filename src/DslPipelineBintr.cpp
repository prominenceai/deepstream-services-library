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
#include "DslServices.h"
#include "DslPipelineBintr.h"

#include <gst/gst.h>

namespace DSL
{
    PipelineBintr::PipelineBintr(const char* name)
        : Bintr(name)
        , m_pPipelineSourcesBintr(nullptr)
        , m_pGstBus(NULL)
        , m_gstBusWatch(0)
        , m_pXWindowEventThread(NULL)
        , m_pXDisplay(XOpenDisplay(NULL))
{
        LOG_FUNC();

        m_pGstObj = GST_OBJECT(gst_pipeline_new(name));
        if (!m_pGstObj)
        {
            LOG_ERROR("Failed to create new GST Pipeline for '" << name << "'");
            throw;
        }
                
        // Initialize "constant-to-string" maps
        _initMaps();
        
        g_mutex_init(&m_busSyncMutex);
        g_mutex_init(&m_busWatchMutex);
        g_mutex_init(&m_displayMutex);

        // get the GST message bus - one per GST pipeline
        m_pGstBus = gst_pipeline_get_bus(GST_PIPELINE(m_pGstObj));
        
        // install the watch function for the message bus
        m_gstBusWatch = gst_bus_add_watch(m_pGstBus, bus_watch, this);
        
        // install the sync handler for the message bus
        gst_bus_set_sync_handler(m_pGstBus, bus_sync_handler, this, NULL);        
    }

    PipelineBintr::~PipelineBintr()
    {
        LOG_FUNC();
        
        Stop();

        if (m_pXWindow)
        {
            XDestroyWindow(m_pXDisplay, m_pXWindow);
        }
        // cleanup all resources
        gst_bus_remove_watch (m_pGstBus);
        gst_object_unref(m_pGstBus);

        g_mutex_clear(&m_busSyncMutex);
        g_mutex_clear(&m_busWatchMutex);
        g_mutex_clear(&m_displayMutex);
    }
    
    bool PipelineBintr::AddSourceBintr(DSL_NODETR_PTR pSourceBintr)
    {
        LOG_FUNC();
        
        // Create the shared sources bintr if it doesn't exist
        if (!m_pPipelineSourcesBintr)
        {
            m_pPipelineSourcesBintr = DSL_PIPELINE_SOURCES_NEW("sources-bin");
            AddChild(m_pPipelineSourcesBintr);
        }

        if (!m_pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)))
        {
            return false;
        }
        return true;
    }

    bool PipelineBintr::IsSourceBintrChild(DSL_NODETR_PTR pSourceBintr)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no Sources");
            return false;
        }
        return (m_pPipelineSourcesBintr->IsChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)));
    }

    bool PipelineBintr::RemoveSourceBintr(DSL_NODETR_PTR pSourceBintr)
    {
        LOG_FUNC();

        // Must cast to SourceBintr first so that correct Instance of RemoveChild is called
        return m_pPipelineSourcesBintr->RemoveChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
    }

    bool PipelineBintr::AddOsdBintr(DSL_NODETR_PTR pOsdBintr)
    {
        LOG_FUNC();
        
        if (m_pOsdBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has an exisiting OSD '" 
                << m_pOsdBintr->GetName());
            return false;
        }
        m_pOsdBintr = std::dynamic_pointer_cast<OsdBintr>(pOsdBintr);
        
        return AddChild(pOsdBintr);
    }

    bool PipelineBintr::AddPrimaryGieBintr(DSL_NODETR_PTR pPrmaryGieBintr)
    {
        LOG_FUNC();
        
        if (m_pPrimaryGieBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has an exisiting Primary GIE '" 
                << m_pPrimaryGieBintr->GetName());
            return false;
        }
        m_pPrimaryGieBintr = std::dynamic_pointer_cast<PrimaryGieBintr>(pPrmaryGieBintr);
        
        return AddChild(pPrmaryGieBintr);
    }

    bool PipelineBintr::AddTrackerBintr(DSL_NODETR_PTR pTrackerBintr)
    {
        LOG_FUNC();

        if (m_pTrackerBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' allready has a Tracker");
            return false;
        }
        m_pTrackerBintr = std::dynamic_pointer_cast<TrackerBintr>(pTrackerBintr);
        
        return AddChild(pTrackerBintr);
    }

    bool PipelineBintr::AddSecondaryGieBintr(DSL_NODETR_PTR pSecondaryGieBintr)
    {
        LOG_FUNC();
        
        // Create the optional Secondary GIEs bintr 
        if (!m_pSecondaryGiesBintr)
        {
            m_pSecondaryGiesBintr = DSL_PIPELINE_SGIES_NEW("sgies-bin");
            AddChild(m_pSecondaryGiesBintr);
        }
        return m_pSecondaryGiesBintr->AddChild(std::dynamic_pointer_cast<SecondaryGieBintr>(pSecondaryGieBintr));
    }

    bool PipelineBintr::AddSinkBintr(DSL_NODETR_PTR pSinkBintr)
    {
        LOG_FUNC();
        
        // Create the shared Sinks bintr if it doesn't exist
        if (!m_pPipelineSinksBintr)
        {
            m_pPipelineSinksBintr = DSL_PIPELINE_SINKS_NEW("sinks-bin");
            AddChild(m_pPipelineSinksBintr);
        }
        return m_pPipelineSinksBintr->AddChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr));
    }

    bool PipelineBintr::IsSinkBintrChild(DSL_NODETR_PTR pSinkBintr)
    {
        LOG_FUNC();

        if (!m_pPipelineSinksBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no Sinks");
            return false;
        }
        return (m_pPipelineSinksBintr->IsChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr)));
    }

    bool PipelineBintr::RemoveSinkBintr(DSL_NODETR_PTR pSinkBintr)
    {
        LOG_FUNC();

        if (!m_pPipelineSinksBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no Sinks");
            return false;
        }

        // Must cast to SourceBintr first so that correct Instance of RemoveChild is called
        return m_pPipelineSinksBintr->RemoveChild(std::dynamic_pointer_cast<SinkBintr>(pSinkBintr));
    }

    bool PipelineBintr::AddDisplayBintr(DSL_NODETR_PTR pDisplayBintr)
    {
        LOG_FUNC();

        if (m_pDisplayBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' allready has a Tiled Display");
            return false;
        }
        m_pDisplayBintr = std::dynamic_pointer_cast<DisplayBintr>(pDisplayBintr);
        
        return AddChild(pDisplayBintr);
    }

    bool PipelineBintr::GetStreamMuxBatchProperties(guint* batchSize, uint* batchTimeout)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxBatchProperties(batchSize, batchTimeout);
        
        return true;
    }

    bool PipelineBintr::SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxBatchProperties(batchSize, batchTimeout);
        
        return true;
    }

    bool PipelineBintr::GetStreamMuxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxDimensions(width, height);
        return true;
    }

    bool PipelineBintr::SetStreamMuxDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxDimensions(width, height);
        return true;
    }
    
    bool PipelineBintr::GetStreamMuxPadding(bool* enabled)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxPadding(enabled);
        return true;
    }
    
    bool PipelineBintr::SetStreamMuxPadding(bool enabled)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxPadding(enabled);
        return true;
    }
    
    bool PipelineBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_INFO("Components for Pipeline '" << GetName() << "' are already assembled");
            return false;
        }
        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipline has no required Source component - and is unable to link");
            return false;
        }
        if (!m_pPipelineSinksBintr)
        {
            LOG_ERROR("Pipline has no required Sink component - and is unable to link");
            return false;
        }
        if (m_pSecondaryGiesBintr and !m_pPrimaryGieBintr)
        {
            LOG_ERROR("Pipline has a Seconday GIE and no Primary GIE - and is unable to linke");
            return false;
        }
        
        // Start with an empty list of linked components
        m_linkedComponents.clear();

        // Link all Source Elementrs, and all Sources to the StreamMux
        // then add the PipelineSourcesBintr as the Source (head) component for this Pipeline
        if (!m_pPipelineSourcesBintr->LinkAll())
        {
            return false;
        }
        m_linkedComponents.push_back(m_pPipelineSourcesBintr);

        if (m_pTrackerBintr)
        {
            // then LinkAll Tracker Elementrs and add as the next component in the Pipeline
            if (!m_pTrackerBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pTrackerBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pTrackerBintr);
        }
        
        if (m_pPrimaryGieBintr)
        {
            // Set the GIE's batch size to the number of active sources, 
            // then LinkAll PrimaryGie Elementrs and add as the next component in the Pipeline
            m_pPrimaryGieBintr->SetBatchSize(m_pPipelineSourcesBintr->GetNumChildren());
            if (!m_pPrimaryGieBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pPrimaryGieBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pPrimaryGieBintr);
        }
        
        if (m_pSecondaryGiesBintr)
        {
            m_pSecondaryGiesBintr->SetInterval(m_pPrimaryGieBintr->GetInterval());

            // Set the Secondary GIE's batch size and interval to the same as the Primary GIE's
            m_pSecondaryGiesBintr->SetPrimaryGieName(m_pPrimaryGieBintr->GetCStrName());
            m_pSecondaryGiesBintr->SetBatchSize(m_pPrimaryGieBintr->GetBatchSize());
            
            // LinkAll SecondaryGie Elementrs and add the Bintr as next component in the Pipeline
            if (!m_pSecondaryGiesBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pSecondaryGiesBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pSecondaryGiesBintr);
        }
        
        if (m_pDisplayBintr)
        {
            // Link All Tiled Display Elementrs and add as the next component in the Pipeline
            if (!m_pDisplayBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pDisplayBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pDisplayBintr);
        }

        if (m_pOsdBintr)
        {
            // LinkAll Osd Elementrs and add as next component in the Pipeline
            if (!m_pOsdBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pOsdBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pOsdBintr);
        }

        if (!m_pPipelineSinksBintr->LinkAll() or
            !m_linkedComponents.back()->LinkToSink(m_pPipelineSinksBintr))
        {
            return false;
        }
        m_linkedComponents.push_back(m_pPipelineSinksBintr);
        
        m_isLinked = true;
        return true;
    }
    
    void PipelineBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            return;
        }
        // iterate through the list of Linked Components, unlinking each
        for (auto const& ivector: m_linkedComponents)
        {
            // all but the tail m_pPipelineSinksBintr will be Linked to Sink
            if (ivector->IsLinkedToSink())
            {
                ivector->UnlinkFromSink();
            }
            ivector->UnlinkAll();
        }

        m_isLinked = false;
    }

    bool PipelineBintr::Play()
    {
        LOG_FUNC();
        
        if (GetState() == GST_STATE_NULL)
        {
            if (!LinkAll() or !Pause())
            {
                LOG_ERROR("Unable to prepare for Play");
                return false;
            }
        }
                
        // flush the output buffer and then wait until all requests have been 
        // received and processed by the X server. TRUE = Discard all queued events
        XSync(m_pXDisplay, FALSE);       

        // Call the base class to complete the Play process
        return Bintr::Play();
    }
    
    bool PipelineBintr::Stop()
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            if (!m_linkedComponents.back()->SendEos())
            {
                LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
                return false;
            }
            Bintr::Stop();
            UnlinkAll();
        }
        
        return true;
        
    }

    void PipelineBintr::DumpToDot(char* filename)
    {
        LOG_FUNC();
        
        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pGstObj), 
            GST_DEBUG_GRAPH_SHOW_ALL, filename);
    }
    
    void PipelineBintr::DumpToDotWithTs(char* filename)
    {
        LOG_FUNC();
        
        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(m_pGstObj), 
            GST_DEBUG_GRAPH_SHOW_ALL, filename);
    }

    DslReturnType PipelineBintr::AddStateChangeListener(dsl_state_change_listener_cb listener, void* userdata)
    {
        LOG_FUNC();
        
        if (m_stateChangeListeners[listener])
        {   
            LOG_ERROR("Pipeline listener is not unique");
            return DSL_RESULT_PIPELINE_LISTENER_NOT_UNIQUE;
        }
        m_stateChangeListeners[listener] = userdata;
        
        return DSL_RESULT_SUCCESS;
    }

    bool PipelineBintr::IsChildStateChangeListener(dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        
        return (bool)m_stateChangeListeners[listener];
    }
    
    DslReturnType PipelineBintr::RemoveStateChangeListener(dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        
        if (!m_stateChangeListeners[listener])
        {   
            LOG_ERROR("Pipeline listener was not found");
            return DSL_RESULT_PIPELINE_LISTENER_NOT_FOUND;
        }
        m_stateChangeListeners.erase(listener);
        
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType PipelineBintr::AddDisplayEventHandler(dsl_display_event_handler_cb handler, void* userdata)
    {
        LOG_FUNC();

        if (m_displayEventHandlers[handler])
        {   
            LOG_ERROR("Pipeline handler is not unique");
            return DSL_RESULT_PIPELINE_HANDLER_NOT_UNIQUE;
        }
        m_displayEventHandlers[handler] = userdata;
        
        return DSL_RESULT_SUCCESS;
    }

    bool PipelineBintr::IsChildDisplayEventHandler(dsl_state_change_listener_cb handler)
    {
        LOG_FUNC();

        return (bool)m_displayEventHandlers[handler];
        
    }

    DslReturnType PipelineBintr::RemoveDisplayEventHandler(dsl_display_event_handler_cb handler)
    {
        LOG_FUNC();

        if (!m_displayEventHandlers[handler])
        {   
            LOG_ERROR("Pipeline handler was not found");
            return DSL_RESULT_PIPELINE_HANDLER_NOT_FOUND;
        }
        m_displayEventHandlers.erase(handler);
        
        return DSL_RESULT_SUCCESS;
    }
    
    bool PipelineBintr::HandleBusWatchMessage(GstMessage* pMessage)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busWatchMutex);
        
        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
        case GST_MESSAGE_STREAM_STATUS:
        case GST_MESSAGE_DURATION_CHANGED:
        case GST_MESSAGE_QOS:
        case GST_MESSAGE_NEW_CLOCK:
        case GST_MESSAGE_ASYNC_DONE:
        case GST_MESSAGE_TAG:
            LOG_INFO("Message type:: " << m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)]);
            return true;
        case GST_MESSAGE_INFO:
            return true;
        case GST_MESSAGE_WARNING:
            return true;
        case GST_MESSAGE_ERROR:
            HandleErrorMessage(pMessage);            
            return true;
        case GST_MESSAGE_STATE_CHANGED:
            HandleStateChanged(pMessage);
            return true;
        case GST_MESSAGE_EOS:
            return false;
        default:
            LOG_INFO("Unhandled message type:: " << GST_MESSAGE_TYPE(pMessage));
        }
        return true;
    }

    GstBusSyncReply PipelineBintr::HandleBusSyncMessage(GstMessage* pMessage)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busSyncMutex);

        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
            if (gst_is_video_overlay_prepare_window_handle_message(pMessage))
            {
                if (CreateXWindow())
                {
                    gst_video_overlay_set_window_handle(
                        GST_VIDEO_OVERLAY(GST_MESSAGE_SRC(pMessage)), m_pXWindow);                
                }
            }
            if (GST_MESSAGE_SRC(pMessage) == GST_OBJECT(m_pGstObj))
            {
                const GstStructure *structure;
                structure = gst_message_get_structure(pMessage);
            }
            return GST_BUS_DROP;

        default:
            break;
        }
        return GST_BUS_PASS;
    }
    
    bool PipelineBintr::HandleStateChanged(GstMessage* pMessage)
    {

        if (GST_ELEMENT(GST_MESSAGE_SRC(pMessage)) != GST_ELEMENT(m_pGstObj))
        {
            return false;
        }

        GstState oldstate, newstate;

        gst_message_parse_state_changed(pMessage, &oldstate, &newstate, NULL);
        
        LOG_INFO(m_mapPipelineStates[oldstate] << " => " << m_mapPipelineStates[newstate]);

        // iterate through the map of state-change-listeners calling each
        for(auto const& imap: m_stateChangeListeners)
        {
            imap.first((uint)oldstate, (uint)newstate, imap.second);
        }
        return true;
    }
    
    void PipelineBintr::HandleXWindowEvents()
    {
        XEvent xEvent;

        while (XPending (m_pXDisplay)) 
        {
            
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);
            
            XNextEvent(m_pXDisplay, &xEvent);
            switch (xEvent.type) 
            {
            case ButtonPress:                
                LOG_INFO("Button pressed");
                break;
                
            case KeyPress:
                LOG_INFO("Key pressed"); 
                
                // wait for key release to process
                break;

            case KeyRelease:
                LOG_INFO("Key released");
                
                // TEMP using any key to quit the main loop - TODO - handle termination
                Stop();
                g_main_loop_quit(Services::GetServices()->GetMainLoopHandle());
                break;
            }
        }
    }

    bool PipelineBintr::CreateXWindow()
    {
        if (!m_pDisplayBintr)
        {
            LOG_ERROR("_createWindow error: Miissing Display Bintr for Pipeline '" << GetName() << '"');
            return false;
        }
        uint width(0), height(0);
        m_pDisplayBintr->GetDimensions(&width, &height);
        
        m_pXWindow = XCreateSimpleWindow(m_pXDisplay, 
            RootWindow(m_pXDisplay, DefaultScreen(m_pXDisplay)), 
            0, 0, width, height, 2, 0x00000000, 0x00000000);            
        LOG_WARN("Setting Window: " << width << ":" << height);

        if (!m_pXWindow)
        {
            LOG_ERROR("Failed to create new X Window for Pipeline '" << GetName() << "' ");
            return false;
        }
        XSetWindowAttributes attr = {0};
        
        attr.event_mask = ButtonPress | KeyRelease;
        XChangeWindowAttributes(m_pXDisplay, m_pXWindow, CWEventMask, &attr);

        Atom wmDeleteMessage = XInternAtom(m_pXDisplay, "WM_DELETE_WINDOW", False);
        if (wmDeleteMessage != None)
        {
            XSetWMProtocols(m_pXDisplay, m_pXWindow, &wmDeleteMessage, 1);
        }
        XMapRaised(m_pXDisplay, m_pXWindow);

        // Start the X window event thread
        std::string threadName = GetName() + std::string("-x-window-event-thread");
        m_pXWindowEventThread = g_thread_new(threadName.c_str(), XWindowEventThread, this);
        
        return true;
    }
    
    
    void PipelineBintr::HandleErrorMessage(GstMessage* pMessage)
    {
        LOG_FUNC();
        
        GError* error = NULL;
        gchar* debugInfo = NULL;
        gst_message_parse_error(pMessage, &error, &debugInfo);

        LOG_ERROR("Error message '" << error->message << "' received from '" 
            << GST_OBJECT_NAME(pMessage->src) << "'");
            
        if (debugInfo)
        {
            LOG_DEBUG("Debug info: " << debugInfo);
        }

        g_error_free(error);
        g_free (debugInfo);
    }    
    
    void PipelineBintr::_initMaps()
    {
        m_mapMessageTypes[GST_MESSAGE_UNKNOWN] = "GST_MESSAGE_UNKNOWN";
        m_mapMessageTypes[GST_MESSAGE_ELEMENT] = "GST_MESSAGE_ELEMENT";
        m_mapMessageTypes[GST_MESSAGE_EOS] = "GST_MESSAGE_EOS";
        m_mapMessageTypes[GST_MESSAGE_INFO] = "GST_MESSAGE_INFO";
        m_mapMessageTypes[GST_MESSAGE_WARNING] = "GST_MESSAGE_WARNING";
        m_mapMessageTypes[GST_MESSAGE_ERROR] = "GST_MESSAGE_ERROR";
        m_mapMessageTypes[GST_MESSAGE_TAG] = "GST_MESSAGE_TAG";
        m_mapMessageTypes[GST_MESSAGE_BUFFERING] = "GST_MESSAGE_BUFFERING";
        m_mapMessageTypes[GST_MESSAGE_STATE_CHANGED] = "GST_MESSAGE_STATE_CHANGED";
        m_mapMessageTypes[GST_MESSAGE_STEP_DONE] = "GST_MESSAGE_STEP_DONE";
        m_mapMessageTypes[GST_MESSAGE_CLOCK_LOST] = "GST_MESSAGE_CLOCK_LOST";
        m_mapMessageTypes[GST_MESSAGE_NEW_CLOCK] = "GST_MESSAGE_NEW_CLOCK";
        m_mapMessageTypes[GST_MESSAGE_STREAM_STATUS] = "GST_MESSAGE_STREAM_STATUS";
        m_mapMessageTypes[GST_MESSAGE_DURATION_CHANGED] = "GST_MESSAGE_DURATION_CHANGED";
        m_mapMessageTypes[GST_MESSAGE_QOS] = "GST_MESSAGE_QOS";

        m_mapPipelineStates[GST_STATE_READY] = "GST_STATE_READY";
        m_mapPipelineStates[GST_STATE_PLAYING] = "GST_STATE_PLAYING";
        m_mapPipelineStates[GST_STATE_PAUSED] = "GST_STATE_PAUSED";
        m_mapPipelineStates[GST_STATE_NULL] = "GST_STATE_NULL";
    }

    static gboolean bus_watch(GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<PipelineBintr*>(pData)->HandleBusWatchMessage(pMessage);
    }    
    
    static GstBusSyncReply bus_sync_handler(GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<PipelineBintr*>(pData)->HandleBusSyncMessage(pMessage);
    }

    static gpointer XWindowEventThread(gpointer pData)
    {
        static_cast<PipelineBintr*>(pData)->HandleXWindowEvents();
       
        return NULL;
    }
    

} // DSL