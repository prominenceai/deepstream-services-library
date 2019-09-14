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

#include "Dsd.h"
#include "DsdDriver.h"

namespace DSD
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
    {
        LOG_FUNC();
        
        // Initialize all 
        g_mutex_init(&m_driverMutex);
        g_mutex_init(&m_displayMutex);

        // Add the event thread
        g_timeout_add(40, EventThread, NULL);

        // Start the X window event thread
        m_pXWindowEventThread = g_thread_new("dsd-x-window-event-thread",
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
        }
        
        g_mutex_clear(&m_displayMutex);
        g_mutex_clear(&m_driverMutex);
    }

    DsdReturnType Driver::SourceNew(const std::string& source, guint type, 
        gboolean live, guint width, guint height, guint fps_n, guint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allSources[source])
        {   
            LOG_ERROR("Source name '" << source << "' is not unique");
            return DSD_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allSources[source] = new SourceBintr(
                source, type, live, width, height, fps_n, fps_d);
        }
        catch(...)
        {
            LOG_ERROR("New Source '" << source << "' threw exception on create");
            return DSD_RESULT_SOURCE_NEW_EXCEPTION;
        }
        LOG_INFO("new source '" << source << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::SourceDelete(const std::string& source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allSources[source])
        {   
            LOG_ERROR("Source name '" << source << "' was not found");
            return DSD_RESULT_SOURCE_NAME_NOT_FOUND;
        }
        try
        {
            delete m_allSources[source];
            m_allSources.erase(source);
        }
        catch(...)
        {
            LOG_ERROR("Source delete '" << source << "' threw exception on create");
            return DSD_RESULT_SOURCE_NEW_EXCEPTION;
        }
        LOG_INFO("Source '" << source << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::StreamMuxNew(const std::string& streammux, 
        gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allStreamMuxs[streammux])
        {   
            LOG_ERROR("Stream Mux name '" << streammux << "' is not unique");
            return DSD_RESULT_STREAMMUX_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allStreamMuxs[streammux] = new StreamMuxBintr(
                streammux, live, batchSize, batchTimeout, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New StreamMux '" << streammux << "' threw exception on create");
            return DSD_RESULT_STREAMMUX_NEW_EXCEPTION;
        }
        LOG_INFO("new Stream Mux '" << streammux << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::StreamMuxDelete(const std::string& streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allStreamMuxs[streammux])
        {   
            LOG_ERROR("Streammux name '" << streammux << "' was not found");
            return DSD_RESULT_STREAMMUX_NAME_NOT_FOUND;
        }
        try
        {
            delete m_allStreamMuxs[streammux];
            m_allStreamMuxs.erase(streammux);
        }
        catch(...)
        {
            LOG_ERROR("Stream Mux delete '" << streammux << "' threw exception on create");
            return DSD_RESULT_SOURCE_NEW_EXCEPTION;
        }
        
        LOG_INFO("Streammux '" << streammux << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::DisplayNew(const std::string& display, 
        guint rows, guint columns, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allDisplays[display])
        {   
            LOG_ERROR("Display name '" << display << "' is not unique");
            return DSD_RESULT_DISPLAY_NAME_NOT_UNIQUE;
        }
        m_allDisplays[display] = 1;
        LOG_INFO("new Display '" << display << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::DisplayDelete(const std::string& display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allDisplays[display])
        {   
            LOG_ERROR("Display name '" << display << "' was not found");
            return DSD_RESULT_DISPLAY_NAME_NOT_FOUND;
        }
        m_allDisplays.erase(display);
        LOG_INFO("Display '" << display << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::GieNew(const std::string& gie, 
        const std::string& model,const std::string& infer, 
        guint batchSize, guint bc1, guint bc2, guint bc3, guint bc4)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allGies[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' is not unique");
            return DSD_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        m_allGies[gie] = 1;
        LOG_INFO("new GIE '" << gie << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::GieDelete(const std::string& gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allGies[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' was not found");
            return DSD_RESULT_GIE_NAME_NOT_FOUND;
        }
        m_allGies.erase(gie);
        LOG_INFO("GIE '" << gie << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::ConfigNew(const std::string& config)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allConfigs[config])
        {   
            LOG_ERROR("Config name '" << config << "' is not unique");
            return DSD_RESULT_CONFIG_NAME_NOT_UNIQUE;
        }
        m_allConfigs[config] = 1;
        LOG_INFO("new Config '" << config << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::ConfigDelete(const std::string& config)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allConfigs[config])
        {   
            LOG_ERROR("Config name '" << config << "' was not found");
            return DSD_RESULT_CONFIG_NAME_NOT_FOUND;
        }
        m_allConfigs.erase(config);
        LOG_INFO("Config '" << config << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::ConfigFileLoad(const std::string& config, const std::string& file)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigFileSave(const std::string& config, const std::string& file)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigFileOverWrite(const std::string& config, const std::string& file)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigSourceAdd(const std::string& config, const std::string& source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigSourceRemove(const std::string& config, const std::string& source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigOsdAdd(const std::string& config, const std::string& osd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigOsdRemove(const std::string& config, const std::string& osd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigGieAdd(const std::string& config, const std::string& gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigGieRemove(const std::string& config, const std::string& gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigDisplayAdd(const std::string& config, const std::string& display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::ConfigDisplayRemove(const std::string& config, const std::string& display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineNew(const std::string& pipeline, const std::string& config)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineDelete(const std::string& pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelinePause(const std::string& pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelinePlay(const std::string& pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineGetState(const std::string& pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
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

        pDrv->HandleXWindowEvents();
       
        return NULL;
    }

} // namespace 