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

#include "Dsl.h"
#include "DslPipelineXWinMgr.h"
#include "DslServices.h"

namespace DSL
{
    PipelineXWinMgr::PipelineXWinMgr(const GstObject* pGstPipeline)
        : m_pXWindowEventThread(NULL)
        , m_pXDisplay(0)
        , m_pXWindow(0)
        , m_pXWindowCreated(false)
        , m_xWindowOffsetX(0)
        , m_xWindowOffsetY(0)
        , m_xWindowWidth(0)
        , m_xWindowHeight(0)
        , m_xWindowfullScreenEnabled(false)
    {
        LOG_FUNC();

        GstBus* pGstBus = gst_pipeline_get_bus(GST_PIPELINE(pGstPipeline));

        // install the sync handler for the message bus
        gst_bus_set_sync_handler(pGstBus, bus_sync_handler, this, NULL);        

        gst_object_unref(pGstBus);

        g_mutex_init(&m_busSyncMutex);
        g_mutex_init(&m_displayMutex);
    }

    PipelineXWinMgr::~PipelineXWinMgr()
    {
        LOG_FUNC();
        
        // cleanup all resources
        if (m_pXDisplay)
        {
            // create scope for the mutex
            {
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);

                if (m_pXWindow and m_pXWindowCreated)
                {
                    XDestroyWindow(m_pXDisplay, m_pXWindow);
                }
                
                XCloseDisplay(m_pXDisplay);
                // Setting the display handle to NULL will terminate the XWindow Event Thread.
                m_pXDisplay = NULL;
            }
            g_thread_join(m_pXWindowEventThread);
        }

        g_mutex_clear(&m_displayMutex);
        g_mutex_clear(&m_busSyncMutex);
    }

    void PipelineXWinMgr::GetXWindowOffsets(uint* xOffset, uint* yOffset)
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

    void PipelineXWinMgr::SetXWindowOffsets(uint xOffset, uint yOffset)
    {
        LOG_FUNC();

        m_xWindowOffsetX = xOffset;
        m_xWindowOffsetY = yOffset;
    }

    void PipelineXWinMgr::GetXWindowDimensions(uint* width, uint* height)
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

   void PipelineXWinMgr::SetXWindowDimensions(uint width, uint height)
    {
        LOG_FUNC();

        m_xWindowWidth = width;
        m_xWindowHeight = height;
        if (m_pXWindow)
        {
            XMoveResizeWindow(m_pXDisplay, m_pXWindow, 
                m_xWindowOffsetX, m_xWindowOffsetY, m_xWindowWidth, m_xWindowHeight);
        }
    }
    
    bool PipelineXWinMgr::GetXWindowFullScreenEnabled()
    {
        LOG_FUNC();
        
        return m_xWindowfullScreenEnabled;
    }
    
    bool PipelineXWinMgr::SetXWindowFullScreenEnabled(bool enabled)
    {
        LOG_FUNC();
        
        if (m_pXWindow)
        {
            LOG_ERROR("Can not set full-screen-enabled once XWindow has been created.");
            return false;
        }
        m_xWindowfullScreenEnabled = enabled;
        return true;
    }

    bool PipelineXWinMgr::AddXWindowKeyEventHandler(dsl_xwindow_key_event_handler_cb handler, void* clientData)
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

    bool PipelineXWinMgr::RemoveXWindowKeyEventHandler(dsl_xwindow_key_event_handler_cb handler)
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
    
    bool PipelineXWinMgr::AddXWindowButtonEventHandler(dsl_xwindow_button_event_handler_cb handler, void* clientData)
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

    bool PipelineXWinMgr::RemoveXWindowButtonEventHandler(dsl_xwindow_button_event_handler_cb handler)
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
    
    bool PipelineXWinMgr::AddXWindowDeleteEventHandler(dsl_xwindow_delete_event_handler_cb handler, void* clientData)
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

    bool PipelineXWinMgr::RemoveXWindowDeleteEventHandler(dsl_xwindow_delete_event_handler_cb handler)
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
    
    GstBusSyncReply PipelineXWinMgr::HandleBusSyncMessage(GstMessage* pMessage)
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
                    
                    DSL_BASE_PTR pWindowSink =
                        DSL::Services::GetServices()->_sinkWindowGet(
                            GST_MESSAGE_SRC(pMessage));
                    if (pWindowSink)
                    {
                        LOG_WARN("Creating Window for WindowSinkBintr '" 
                            << pWindowSink->GetName() << "'");
                    }
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

    void PipelineXWinMgr::HandleXWindowEvents()
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

    bool PipelineXWinMgr::CreateXWindow()
    {
        LOG_FUNC();
        
        if (!m_xWindowWidth or !m_xWindowHeight)
        {
            LOG_ERROR("Failed to create new X Display with invalid width or height");
            return false;
        }

        // create new XDisplay first
        m_pXDisplay = XOpenDisplay(NULL);
        if (!m_pXDisplay)
        {
            LOG_ERROR("Failed to create new X Display");
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
            LOG_ERROR("Failed to create new X Window");
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
        m_pXWindowEventThread = g_thread_new(NULL, XWindowEventThread, this);
        
        return true;
    }

    bool PipelineXWinMgr::OwnsXWindow()
    {
        LOG_FUNC();
        
        return (m_pXWindow and m_pXWindowCreated);
    }
    
    Window PipelineXWinMgr::GetXWindow()
    {
        LOG_FUNC();
        
        return m_pXWindow;
    }
    
    bool PipelineXWinMgr::SetXWindow(Window xWindow)
    {
        LOG_FUNC();
        
//        if (IsLinked())
//        {
//            LOG_ERROR("Pipeline '" << GetName() 
//                << "' failed to set XWindow handle as it is currently linked");
//            return false;
//        }
        if (m_pXWindowCreated)
        {
            DestroyXWindow();
            LOG_INFO("PipelineXWinMgr destroyed its own XWindow to use the client's");
            m_pXWindowCreated = false;
        }
        m_pXWindow = xWindow;
        return true;
    }
    
    bool PipelineXWinMgr::ClearXWindow()
    {
        LOG_FUNC();
        
        if (!m_pXWindow or !m_pXWindowCreated)
        {
            LOG_ERROR("PipelineXWinMgr does not own a XWindow to clear");
            return false;
        }
        XClearWindow(m_pXDisplay, m_pXWindow);
        return true;
    }
    
    bool PipelineXWinMgr::DestroyXWindow()
    {
        LOG_FUNC();
        
        if (!m_pXWindow or !m_pXWindowCreated)
        {
            LOG_INFO("PipelineXWinMgr does not own an XWindow to distroy");
            return false;
        }
        XDestroyWindow(m_pXDisplay, m_pXWindow);
        m_pXWindow = 0;
        m_pXWindowCreated = False;
        return true;
    }

    static GstBusSyncReply bus_sync_handler(GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<PipelineXWinMgr*>(pData)->HandleBusSyncMessage(pMessage);
    }

    static gpointer XWindowEventThread(gpointer pPipeline)
    {
        static_cast<PipelineXWinMgr*>(pPipeline)->HandleXWindowEvents();
       
        return NULL;
    }
    
} // DSL   