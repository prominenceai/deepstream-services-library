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

#include "Dss.h"
#include "DssDriver.h"

namespace DSS
{
    
    
    // Initialize the Driver's single instance pointer
    Driver* Driver::m_pInstatnce = NULL;
    
    Driver* Driver::GetDriver()
    {
        
        if (!m_pInstatnce)
        {
            LOG_INFO("Driver Initialization");
            
            m_pInstatnce = new Driver;
        }
        return m_pInstatnce;
    }
        
    Driver::Driver()
        : m_pDisplay(XOpenDisplay(NULL))
        , m_pMainLoop(g_main_loop_new(NULL, FALSE))
        , m_pXWindowEventThread(NULL)
        , m_pAppContext(NULL)
    {
        LOG_FUNC();
        
        // Initialize all 
        g_mutex_init(&m_driverMutex);
        g_mutex_init(&m_displayMutex);

        // Add the event thread
        g_timeout_add(40, EventThread, NULL);

        // Start the X window event thread
        m_pXWindowEventThread = g_thread_new("dss-x-window-event-thread",
            XWindowEventThread, NULL);
        
    }

    Driver::~Driver()
    {
        LOG_FUNC();
        
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

            if (m_pDisplay) 
            { 
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);
                
                // The X window event thread will exit on display close
                XCloseDisplay(m_pDisplay); 
            }
            
            if (m_pMainLoop)
            {
                LOG_WARN("Main loop is still running!");
                g_main_loop_quit(m_pMainLoop);
            }

            if (m_pAppContext)
            {   
                delete m_pAppContext;
            }
        }
        
        g_mutex_clear(&m_displayMutex);
        g_mutex_clear(&m_driverMutex);
    }

    bool Driver::Configure(const std::string& cfgFilePathSpec)
    {
        LOG_FUNC();
        
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        Config* pAppConfig = new Config();
        if (!(pAppConfig->LoadFile(cfgFilePathSpec)))
        {
            return false;
        }    
        m_pAppContext = new AppContext(*pAppConfig);
        
        return m_pAppContext->Update(m_pDisplay);
    }
        
    bool Driver::HandleXWindowEvents()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);
        
        XEvent xEvent;

        while (XPending (m_pDisplay)) 
        {
            
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_displayMutex);
            
            XNextEvent(m_pDisplay, &xEvent);
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
                
                break;
            }
        }
        return true;
    }
    
    static gboolean EventThread(gpointer arg)
    {
        Driver* pDrv = Driver::GetDriver();
        
        return true;
    }
    

    static gpointer XWindowEventThread(gpointer arg)
    {
        Driver* pDrv = Driver::GetDriver();

        while (pDrv->IsDisplayActive())
        {
            pDrv->HandleXWindowEvents();
            
        }
        
        return NULL;
    }

} // namespace 