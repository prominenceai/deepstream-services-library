/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslSurfaceTransform.h"

#include "DslServices.h"
#include "DslPipelineBintr.h"

namespace DSL
{
    PipelineBintr::PipelineBintr(const char* name)
        : BranchBintr(name)
        , m_pGstBus(NULL)
        , m_gstBusWatch(0)
        , m_pXWindowEventThread(NULL)
        , m_pXDisplay(0)
        , m_pXWindow(0)
		, m_pXWindowCreated(false)
        , m_xWindowOffsetX(0)
        , m_xWindowOffsetY(0)
        , m_xWindowWidth(0)
        , m_xWindowHeight(0)
        , m_errorNotificationTimerId(0)
        , m_xWindowfullScreenEnabled(false)
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
        g_mutex_init(&m_lastErrorMutex);

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
        
        // cleanup all resources
        if (m_pXWindow and m_pXWindowCreated)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);
            XDestroyWindow(m_pXDisplay, m_pXWindow);
        }
        if (m_pXDisplay)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);
            XCloseDisplay(m_pXDisplay);
            // Setting the display handle to NULL will terminate the XWindow Event Thread.
            m_pXDisplay = NULL;
        }
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busWatchMutex);
            gst_bus_remove_watch(m_pGstBus);
            gst_object_unref(m_pGstBus);
        }
        
        g_mutex_clear(&m_busSyncMutex);
        g_mutex_clear(&m_busWatchMutex);
        g_mutex_clear(&m_displayMutex);
        g_mutex_clear(&m_lastErrorMutex);
    }
    
    bool PipelineBintr::AddSourceBintr(DSL_BASE_PTR pSourceBintr)
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

    bool PipelineBintr::IsSourceBintrChild(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no Sources");
            return false;
        }
        return (m_pPipelineSourcesBintr->IsChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)));
    }

    bool PipelineBintr::RemoveSourceBintr(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        // Must cast to SourceBintr first so that correct Instance of RemoveChild is called
        return m_pPipelineSourcesBintr->RemoveChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
    }


    void PipelineBintr::GetStreamMuxBatchProperties(guint* batchSize, uint* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    bool PipelineBintr::SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout)
    {
        LOG_FUNC();

        m_batchSize = batchSize;
        m_batchTimeout = batchTimeout;

        if (IsLinked())
        {
            LOG_ERROR("Pipeline '" << GetName() << "' is currently Linked - batch properties can not be updated");
            return false;
            
        }
        if (m_pPipelineSourcesBintr)
        {
            m_pPipelineSourcesBintr->SetStreamMuxBatchProperties(m_batchSize, m_batchTimeout);
        }
        
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
    
    bool PipelineBintr::GetStreamMuxNumSurfacesPerFrame(uint* num)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxNumSurfacesPerFrame(num);
        return true;
    }
    
    bool PipelineBintr::SetStreamMuxNumSurfacesPerFrame(uint num)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxNumSurfacesPerFrame(num);
        return true;
    }
    
    void PipelineBintr::GetXWindowOffsets(uint* xOffset, uint* yOffset)
    {
        LOG_FUNC();

        if (m_pXWindow)
        {
            XWindowAttributes attrs;
            XGetWindowAttributes(m_pXDisplay, m_pXWindow, &attrs);
            m_xWindowOffsetX = attrs.x;
            m_xWindowOffsetY = attrs.y;
        }
        *xOffset = m_xWindowOffsetX;
        *yOffset = m_xWindowOffsetY;
    }

    void PipelineBintr::SetXWindowOffsets(uint xOffset, uint yOffset)
    {
        LOG_FUNC();

        m_xWindowOffsetX = xOffset;
        m_xWindowOffsetY = yOffset;
    }

    void PipelineBintr::GetXWindowDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        if (m_pXWindow)
        {
            XWindowAttributes attrs;
            XGetWindowAttributes(m_pXDisplay, m_pXWindow, &attrs);
            m_xWindowWidth = attrs.width;
            m_xWindowHeight = attrs.height;
        }
        *width = m_xWindowWidth;
        *height = m_xWindowHeight;
    }

   void PipelineBintr::SetXWindowDimensions(uint width, uint height)
    {
        LOG_FUNC();

        m_xWindowWidth = width;
        m_xWindowHeight = height;
    }
    
    bool PipelineBintr::GetXWindowFullScreenEnabled()
    {
        LOG_FUNC();
        
        return m_xWindowfullScreenEnabled;
    }
    
    bool PipelineBintr::SetXWindowFullScreenEnabled(bool enabled)
    {
        LOG_FUNC();
        
        // TODO verify dimensions before setting.
        if (m_pXWindow)
        {
            LOG_ERROR("Can not set full-screen-enable for Pipeline '" << GetName() << "' once XWindow has been created.");
            return false;
        }
        m_xWindowfullScreenEnabled = enabled;
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
            LOG_ERROR("Pipline '" << GetName() << "' has no required Source component - and is unable to link");
            return false;
        }

        // If the batch size has not been explicitely set, use the number of sources.
        if (m_batchSize < m_pPipelineSourcesBintr->GetNumChildren())
        {
            SetStreamMuxBatchProperties(m_pPipelineSourcesBintr->GetNumChildren(), m_batchTimeout);
        }
        
        // Start with an empty list of linked components
        m_linkedComponents.clear();

        // Link all Source Elementrs (required component), and all Sources to the StreamMuxer
        // then add the PipelineSourcesBintr as the Source (head) component for this Pipeline
        if (!m_pPipelineSourcesBintr->LinkAll())
        {
            return false;
        }
        m_linkedComponents.push_back(m_pPipelineSourcesBintr);
        
        LOG_INFO("Pipeline '" << GetName() << "' Linked up all Source '" << 
            m_pPipelineSourcesBintr->GetName() << "' successfully");

        // call the base class to Link all remaining components.
        return BranchBintr::LinkAll();
    }

    bool PipelineBintr::Play()
    {
        LOG_FUNC();
        
        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_NULL)
        {
            if (!LinkAll())
            {
                LOG_ERROR("Unable to prepare Pipeline '" << GetName() << "' for Play");
                return false;
            }
            // For non-live sources we Pause to preroll before we play
            if (!m_pPipelineSourcesBintr->StreamMuxPlayTypeIsLiveGet())
            {
                if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
                {
                    LOG_ERROR("Failed to Pause non-live soures before playing Pipeline '" << GetName() << "'");
                    return false;
                }
            }
        }
                
        // Call the base class to complete the Play process
        return SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
    }

    bool PipelineBintr::Pause()
    {
        LOG_FUNC();
        
        GstState state;
        GetState(state, 0);
        if (state != GST_STATE_PLAYING)
        {
            LOG_WARN("Pipeline '" << GetName() << "' is not in a state of Playing");
            return false;
        }
        // Call the base class to Pause
        if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Pause Pipeline '" << GetName() << "'");
            return false;
        }
        return true;
    }

    bool PipelineBintr::Stop()
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            GstState state;
            GetState(state, 0);
            if (state == GST_STATE_PAUSED or state == GST_STATE_PLAYING)
            {
                SendEos();
                sleep(1);
            }
            if (!SetState(GST_STATE_NULL, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
                return false;
            }
            UnlinkAll();
        }
        return true;
    }

    bool PipelineBintr::IsLive()
    {
        LOG_FUNC();
        
        if (!m_pPipelineSourcesBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no sources, therefore is-live = false");
            return false;
        }
        return m_pPipelineSourcesBintr->StreamMuxPlayTypeIsLiveGet();
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

    bool PipelineBintr::AddStateChangeListener(dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        
        if (m_stateChangeListeners.find(listener) != m_stateChangeListeners.end())
        {   
            LOG_ERROR("Pipeline listener is not unique");
            return false;
        }
        m_stateChangeListeners[listener] = clientData;
        
        return true;
    }

    bool PipelineBintr::RemoveStateChangeListener(dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_stateChangeListeners.find(listener) == m_stateChangeListeners.end())
        {   
            LOG_ERROR("Pipeline listener was not found");
            return false;
        }
        m_stateChangeListeners.erase(listener);
        
        return true;
    }

    bool PipelineBintr::AddEosListener(dsl_eos_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        
        if (m_eosListeners.find(listener) != m_eosListeners.end())
        {   
            LOG_ERROR("Pipeline listener is not unique");
            return false;
        }
        m_eosListeners[listener] = clientData;
        
        return true;
    }

    bool PipelineBintr::RemoveEosListener(dsl_eos_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_eosListeners.find(listener) == m_eosListeners.end())
        {   
            LOG_ERROR("Pipeline listener was not found");
            return false;
        }
        m_eosListeners.erase(listener);
        
        return true;
    }

    bool PipelineBintr::AddErrorMessageHandler(dsl_error_message_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        
        if (m_errorMessageHandlers.find(handler) != m_errorMessageHandlers.end())
        {   
            LOG_ERROR("Pipeline handler is not unique");
            return false;
        }
        m_errorMessageHandlers[handler] = clientData;
        
        return true;
    }

    bool PipelineBintr::RemoveErrorMessageHandler(dsl_error_message_handler_cb handler)
    {
        LOG_FUNC();
        
        if (m_errorMessageHandlers.find(handler) == m_errorMessageHandlers.end())
        {   
            LOG_ERROR("Pipeline handler was not found");
            return false;
        }
        m_errorMessageHandlers.erase(handler);
        
        return true;
    }

    bool PipelineBintr::AddXWindowKeyEventHandler(dsl_xwindow_key_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();

        if (m_xWindowKeyEventHandlers.find(handler) != m_xWindowKeyEventHandlers.end())
        {   
            LOG_ERROR("Pipeline handler is not unique");
            return false;
        }
        m_xWindowKeyEventHandlers[handler] = clientData;
        
        return true;
    }

    bool PipelineBintr::RemoveXWindowKeyEventHandler(dsl_xwindow_key_event_handler_cb handler)
    {
        LOG_FUNC();

        if (m_xWindowKeyEventHandlers.find(handler) == m_xWindowKeyEventHandlers.end())
        {   
            LOG_ERROR("Pipeline handler was not found");
            return false;
        }
        m_xWindowKeyEventHandlers.erase(handler);
        
        return true;
    }
    
    bool PipelineBintr::AddXWindowButtonEventHandler(dsl_xwindow_button_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();

        if (m_xWindowButtonEventHandlers.find(handler) != m_xWindowButtonEventHandlers.end())
        {   
            LOG_ERROR("Pipeline handler is not unique");
            return false;
        }
        m_xWindowButtonEventHandlers[handler] = clientData;
        
        return true;
    }

    bool PipelineBintr::RemoveXWindowButtonEventHandler(dsl_xwindow_button_event_handler_cb handler)
    {
        LOG_FUNC();

        if (m_xWindowButtonEventHandlers.find(handler) == m_xWindowButtonEventHandlers.end())
        {   
            LOG_ERROR("Pipeline handler was not found");
            return false;
        }
        m_xWindowButtonEventHandlers.erase(handler);
        
        return true;
    }
    
    bool PipelineBintr::AddXWindowDeleteEventHandler(dsl_xwindow_delete_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();

        if (m_xWindowDeleteEventHandlers.find(handler) != m_xWindowDeleteEventHandlers.end())
        {   
            LOG_ERROR("Pipeline handler is not unique");
            return false;
        }
        m_xWindowDeleteEventHandlers[handler] = clientData;
        
        return true;
    }

    bool PipelineBintr::RemoveXWindowDeleteEventHandler(dsl_xwindow_delete_event_handler_cb handler)
    {
        LOG_FUNC();

        if (m_xWindowDeleteEventHandlers.find(handler) == m_xWindowDeleteEventHandlers.end())
        {   
            LOG_ERROR("Pipeline handler was not found");
            return false;
        }
        m_xWindowDeleteEventHandlers.erase(handler);
        
        return true;
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
            LOG_INFO("Message type:: " << m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)]);
            return true;
        case GST_MESSAGE_TAG:
            LOG_DEBUG("Message type:: " << m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)]);
            return true;
        case GST_MESSAGE_EOS:
            HandleEosMessage(pMessage);
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
                // Window Sink component is signaling to prepare window handle
                // perform single creation of XWindow if not provided by the client

                LOG_INFO("Prepare window handle received from source " << GST_MESSAGE_SRC_NAME(pMessage));

                if (!m_pXWindow)
                {
                    g_object_get(GST_MESSAGE_SRC(pMessage), "window-x", &m_xWindowOffsetX, NULL);
                    g_object_get(GST_MESSAGE_SRC(pMessage), "window-y", &m_xWindowOffsetY, NULL);
                    g_object_get(GST_MESSAGE_SRC(pMessage), "window-width", &m_xWindowWidth, NULL);
                    g_object_get(GST_MESSAGE_SRC(pMessage), "window-height", &m_xWindowHeight, NULL);
                    
                    CreateXWindow();
                }
                
                gst_video_overlay_set_window_handle(
                    GST_VIDEO_OVERLAY(GST_MESSAGE_SRC(pMessage)), m_pXWindow);
                gst_video_overlay_expose(
                    GST_VIDEO_OVERLAY(GST_MESSAGE_SRC(pMessage)));
                UNREF_MESSAGE_ON_RETURN(pMessage);
                return GST_BUS_DROP;
            }
            break;
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
            try
            {
                imap.first((uint)oldstate, (uint)newstate, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Pipeline '" << GetName() << "' threw exception calling Client State-Change-Lister");
            }
        }
        return true;
    }
    
    void PipelineBintr::HandleEosMessage(GstMessage* pMessage)
    {
        LOG_INFO("EOS message recieved");
        
        // iterate through the map of EOS-listeners calling each
        for(auto const& imap: m_eosListeners)
        {
            try
            {
                imap.first(imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Pipeline '" << GetName() << "' threw exception calling Client EOS-Lister");
            }
        }
    }
    
    void PipelineBintr::HandleXWindowEvents()
    {
        while (m_pXDisplay)
        {
            {
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);
                while (m_pXDisplay and XPending(m_pXDisplay)) 
                {
                    XEvent xEvent;
                    XNextEvent(m_pXDisplay, &xEvent);
                    XButtonEvent buttonEvent = xEvent.xbutton;
                    switch (xEvent.type) 
                    {
                    case ButtonPress:
                        LOG_INFO("Button '" << buttonEvent.button << "' pressed: xpos = " 
                            << buttonEvent.x << ": ypos = " << buttonEvent.y);
                        
                        // iterate through the map of XWindow Button Event handlers calling each
                        for(auto const& imap: m_xWindowButtonEventHandlers)
                        {
                            imap.first(buttonEvent.button, buttonEvent.x, buttonEvent.y, imap.second);
                        }
                        break;
                        
                    case KeyRelease:
                        KeySym key;
                        char keyString[255];
                        if (XLookupString(&xEvent.xkey, keyString, 255, &key,0))
                        {   
                            keyString[1] = 0;
                            std::string cstrKeyString(keyString);
                            std::wstring wstrKeyString(cstrKeyString.begin(), cstrKeyString.end());
                            LOG_INFO("Key released = '" << cstrKeyString << "'"); 
                            
                            // iterate through the map of XWindow Key Event handlers calling each
                            for(auto const& imap: m_xWindowKeyEventHandlers)
                            {
                                imap.first(wstrKeyString.c_str(), imap.second);
                            }
                        }
                        break;
                        
                    case ClientMessage:
                        LOG_INFO("Client message");

                        if (XInternAtom(m_pXDisplay, "WM_DELETE_WINDOW", True) != None)
                        {
                            LOG_INFO("WM_DELETE_WINDOW message received");
                            Stop();
                            // iterate through the map of XWindow Delete Event handlers calling each
                            for(auto const& imap: m_xWindowDeleteEventHandlers)
                            {
                                imap.first(imap.second);
                            }
                        }
                        break;
                        
                    default:
                        break;
                    }
                }
            }
            g_usleep(G_USEC_PER_SEC / 20);
        }
    }

    bool PipelineBintr::CreateXWindow()
    {
        LOG_FUNC();
        
        if (!m_xWindowWidth or !m_xWindowHeight)
        {
            LOG_ERROR("Failed to create new X Display for Pipeline '" << GetName() << "' with invalid width or height");
            return false;
        }

        // create new XDisplay first
        m_pXDisplay = XOpenDisplay(NULL);
        if (!m_pXDisplay)
        {
            LOG_ERROR("Failed to create new X Display for Pipeline '" << GetName() << "' ");
            return false;
        }
        
        // create new simple XWindow either in 'full-screen-enabled' or using the Window Sink offsets and dimensions
        if (m_xWindowfullScreenEnabled)
        {
            LOG_INFO("Creating new XWindow in 'full-screen-mode'");

            m_pXWindow = XCreateSimpleWindow(m_pXDisplay, 
                RootWindow(m_pXDisplay, DefaultScreen(m_pXDisplay)), 
                0, 0, 10, 10, 0, BlackPixel(m_pXDisplay, 0), BlackPixel(m_pXDisplay, 0));
        } 
        else
        {
            LOG_INFO("Creating new XWindow: x-offset = " << m_xWindowOffsetX << ", y-offset = " << m_xWindowOffsetY << 
                ", width = " << m_xWindowWidth << ", height = " << m_xWindowHeight);
        
            m_pXWindow = XCreateSimpleWindow(m_pXDisplay, 
                RootWindow(m_pXDisplay, DefaultScreen(m_pXDisplay)), 
                m_xWindowOffsetX, m_xWindowOffsetY, m_xWindowWidth, m_xWindowHeight, 2, 0, 0);
        } 
        
            
        if (!m_pXWindow)
        {
            LOG_ERROR("Failed to create new X Window for Pipeline '" << GetName() << "' ");
            return false;
        }
		// Flag used to cleanup the handle - pipeline created vs. client created.
		m_pXWindowCreated = True;
        XSetWindowAttributes attr{0};
        
        attr.event_mask = ButtonPress | KeyRelease;
        XChangeWindowAttributes(m_pXDisplay, m_pXWindow, CWEventMask, &attr);

        Atom wmDeleteMessage = XInternAtom(m_pXDisplay, "WM_DELETE_WINDOW", False);
        if (wmDeleteMessage != None)
        {
            XSetWMProtocols(m_pXDisplay, m_pXWindow, &wmDeleteMessage, 1);
        }
        
        XMapRaised(m_pXDisplay, m_pXWindow);
        if (m_xWindowfullScreenEnabled)
        {
            Atom wmState = XInternAtom(m_pXDisplay, "_NET_WM_STATE", False);
            Atom fullscreen = XInternAtom(m_pXDisplay, "_NET_WM_STATE_FULLSCREEN", False);
            XEvent xev{0};
            xev.type = ClientMessage;
            xev.xclient.window = m_pXWindow;
            xev.xclient.message_type = wmState;
            xev.xclient.format = 32;
            xev.xclient.data.l[0] = 1;
            xev.xclient.data.l[1] = fullscreen;
            xev.xclient.data.l[2] = 0;        

            XSendEvent(m_pXDisplay, DefaultRootWindow(m_pXDisplay), False,
                SubstructureRedirectMask | SubstructureNotifyMask, &xev);
        }
        // flush the XWindow output buffer and then wait until all requests have been 
        // received and processed by the X server. TRUE = Discard all queued events
        XSync(m_pXDisplay, TRUE);

        // Start the X window event thread
        std::string threadName = GetName() + std::string("-x-window-event-thread");
        m_pXWindowEventThread = g_thread_new(threadName.c_str(), XWindowEventThread, this);
        
        return true;
    }
    
	bool PipelineBintr::SetXWindow(Window xWindow)
	{
        LOG_FUNC();
		
		if (IsLinked())
		{
            LOG_ERROR("Pipeline '" << GetName() 
				<< "' failed to set XWindow handle as it is currently linked");
            return false;
		}
		if (m_pXWindowCreated)
		{
            LOG_ERROR("Pipeline '" << GetName() 
				<< "' failed to set XWindow handle as one has already been created");
            return false;
		}
		m_pXWindow = xWindow;
		return true;
	}
	
    bool PipelineBintr::ClearXWindow()
    {
        LOG_FUNC();
        
        if (!m_pXWindow)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has now XWindow to clear");
            return false;
        }
        XClearWindow(m_pXDisplay, m_pXWindow);
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

        // persist the last error information
        std::string cstrSource(GST_OBJECT_NAME(pMessage->src));
        std::string cstrMessage(error->message);

        std::wstring wstrSource(cstrSource.begin(), cstrSource.end());
        std::wstring wstrMessage(cstrMessage.begin(), cstrMessage.end());
        
        // Setting the last error message will invoke a timer thread to notify all client handlers.
        SetLastErrorMessage(wstrSource, wstrMessage);
        
        g_error_free(error);
        g_free(debugInfo);
    }    

    void PipelineBintr::GetLastErrorMessage(std::wstring& source, std::wstring& message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_lastErrorMutex);
        
        source = m_lastErrorSource;
        message = m_lastErrorMessage;
    }

    void PipelineBintr::SetLastErrorMessage(std::wstring& source, std::wstring& message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_lastErrorMutex);

        m_lastErrorSource = source;
        m_lastErrorMessage = message;
        
        if (m_errorMessageHandlers.size())
        {
            m_errorNotificationTimerId = g_timeout_add(1, ErrorMessageHandlersNotificationHandler, this);
        }
    }
    
    int PipelineBintr::NotifyErrorMessageHandlers()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_lastErrorMutex);

        // iterate through the map of state-change-listeners calling each
        for(auto const& imap: m_errorMessageHandlers)
        {
            try
            {
                imap.first(m_lastErrorSource.c_str(), m_lastErrorMessage.c_str(), imap.second);
            }
            catch(...)
            {
                LOG_ERROR("RTSP Source '" << GetName() << "' threw exception calling Client State-Change-Lister");
            }
        }
        // clear the timer id and return false to self remove
        m_errorNotificationTimerId = 0;
        return false;
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

    static int ErrorMessageHandlersNotificationHandler(gpointer pPipeline)
    {
        return static_cast<PipelineBintr*>(pPipeline)->
            NotifyErrorMessageHandlers();
    }

    static gpointer XWindowEventThread(gpointer pData)
    {
        static_cast<PipelineBintr*>(pData)->HandleXWindowEvents();
       
        return NULL;
    }
    

} // DSL
