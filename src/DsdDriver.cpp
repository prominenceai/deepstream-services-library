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
#include "DsdApi.h"

DsdReturnType dsd_source_new(const std::string& source, guint type, 
        gboolean live, guint width, guint height, guint fps_n, guint fps_d)
{
    return DSD::Driver::GetDriver()->SourceNew(source, type, live, 
        width, height, fps_n, fps_d);
}

DsdReturnType dsd_source_delete(const std::string& source)
{
    return DSD::Driver::GetDriver()->SourceDelete(source);
};

DsdReturnType dsd_streammux_new(const std::string& streammux, 
        gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
{
    return DSD::Driver::GetDriver()->StreamMuxNew(streammux, live, 
        batchSize, batchTimeout, width, height);
}

DsdReturnType dsd_streammux_delete(const std::string& streammux)
{
    return DSD::Driver::GetDriver()->StreamMuxDelete(streammux);
}

DsdReturnType dsd_display_new(const std::string& display, 
        guint rows, guint columns, guint width, guint height)
{
    return DSD::Driver::GetDriver()->DisplayNew(display, rows, columns, width, height);
}

DsdReturnType dsd_display_delete(const std::string& display)
{
    return DSD::Driver::GetDriver()->DisplayDelete(display);
}

DsdReturnType dsd_gie_new(const std::string& gie, 
        const std::string& model,const std::string& infer, 
        guint batchSize, guint bc1, guint bc2, guint bc3, guint bc4)
{
    return DSD::Driver::GetDriver()->GieNew(gie, model, infer, 
        batchSize, bc1, bc2, bc3, bc4);
}

DsdReturnType dsd_gie_delete(const std::string& gie)
{
    return DSD::Driver::GetDriver()->GieDelete(gie);
}

DsdReturnType dsd_pipeline_new(const std::string& pipeline)
{
    return DSD::Driver::GetDriver()->PipelineNew(pipeline);
}

DsdReturnType dsd_pipeline_delete(const std::string& pipeline)
{
    return DSD::Driver::GetDriver()->PipelineDelete(pipeline);
}

DsdReturnType dsd_pipeline_source_add(const std::string& pipeline, 
    const std::string& source)
{
    return DSD::Driver::GetDriver()->PipelineSourceAdd(pipeline, source);
};

DsdReturnType dsd_pipeline_source_remove(const std::string& pipeline, 
    const std::string& source)
{
    return DSD::Driver::GetDriver()->PipelineSourceRemove(pipeline, source);
};

DsdReturnType dsd_pipeline_streammux_add(const std::string& pipeline, 
    const std::string& streammux)
{
    return DSD::Driver::GetDriver()->PipelineStreamMuxAdd(pipeline, streammux);
};

DsdReturnType dsd_pipeline_streammux_remove(const std::string& pipeline, 
    const std::string& streammux)
{
    return DSD::Driver::GetDriver()->PipelineStreamMuxRemove(pipeline, streammux);
};

DsdReturnType dsd_pipeline_osd_add(const std::string& pipeline, 
    const std::string& osd)
{
    return DSD::Driver::GetDriver()->PipelineOsdAdd(pipeline, osd);
};

DsdReturnType dsd_pipeline_osd_remove(const std::string& pipeline, 
    const std::string& osd)
{
    return DSD::Driver::GetDriver()->PipelineOsdRemove(pipeline, osd);
};

DsdReturnType dsd_pipeline_gie_add(const std::string& pipeline, 
    const std::string& gie)
{
    return DSD::Driver::GetDriver()->PipelineGieAdd(pipeline, gie);
};

DsdReturnType dsd_pipeline_gie_remove(const std::string& pipeline, 
    const std::string& gie)
{
    return DSD::Driver::GetDriver()->PipelineGieRemove(pipeline, gie);
};

DsdReturnType dsd_pipeline_display_add(const std::string& pipeline, 
    const std::string& display)
{
    DSD::Driver::GetDriver()->PipelineDisplayAdd(pipeline, display);
};

DsdReturnType dsd_pipeline_display_remove(const std::string& pipeline, 
    const std::string& display)
{
    return DSD::Driver::GetDriver()->PipelineDisplayRemove(pipeline, display);
};

DsdReturnType dsd_pipeline_pause(const std::string& pipeline)
{
    return DSD::Driver::GetDriver()->PipelineGetState(pipeline);
};

DsdReturnType dsd_pipeline_play(const std::string& pipeline)
{
    return DSD::Driver::GetDriver()->PipelinePlay(pipeline);
};

DsdReturnType dsd_pipeline_get_state(const std::string& pipeline)
{
    return DSD::Driver::GetDriver()->PipelineGetState(pipeline);
};

void dsd_main_loop_run()
{
    DSD::Driver::GetDriver()->MainLoopRun();
};

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
        : m_pMainLoop(g_main_loop_new(NULL, FALSE))
        , m_pXDisplay(XOpenDisplay(NULL))
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
            LOG_ERROR("Source Delete '" << source << "' threw exception");
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
            LOG_ERROR("Stream Mux delete '" << streammux << "' threw exception");
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
        try
        {
            m_allDisplays[display] = new DisplayBintr(
                display, m_pXDisplay, rows, columns, width, height);
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display New'" << display << "' threw exception on create");
            return DSD_RESULT_DISPLAY_NEW_EXCEPTION;
        }
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
        try
        {
            delete m_allDisplays[display];
            m_allDisplays.erase(display);
        }
        catch(...)
        {
            LOG_ERROR("Display delete '" << display << "' threw exception");
            return DSD_RESULT_DISPLAY_NEW_EXCEPTION;
        }
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
        try
        {
            m_allGies[gie] = new PrimaryGieBintr(
                gie, model, infer, batchSize, bc1, bc2, bc3, bc4);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << gie << "' threw exception on create");
            return DSD_RESULT_GIE_NEW_EXCEPTION;
        }
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
        try
        {
            delete m_allGies[gie];
            m_allGies.erase(gie);
        }
        catch(...)
        {
            LOG_ERROR("Delete Primary GIE '" << gie << "' threw exception on create");
            return DSD_RESULT_DISPLAY_NEW_EXCEPTION;
        }
        LOG_INFO("GIE '" << gie << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineNew(const std::string& pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allPipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' is not unique");
            return DSD_RESULT_PIPELINE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allPipelines[pipeline] = new PipelineBintr(pipeline);
        }
        catch(...)
        {
            LOG_ERROR("New Pipeline '" << pipeline << "' threw exception on create");
            return DSD_RESULT_PIPELINE_NEW_EXCEPTION;
        }
        LOG_INFO("new PIPELINE '" << pipeline << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineDelete(const std::string& pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allPipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        try
        {
            delete m_allPipelines[pipeline];
            m_allPipelines.erase(pipeline);
        }
        catch(...)
        {
            LOG_ERROR("Delete Pipeline '" << pipeline << "' threw exception on create");
            return DSD_RESULT_PIPELINE_NEW_EXCEPTION;
        }
        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineSourceAdd(const std::string& pipeline, 
        const std::string& source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allPipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        if (!m_allSources[source])
        {   
            LOG_ERROR("Source name '" << source << "' was not found");
            return DSD_RESULT_SOURCE_NAME_NOT_FOUND;
        }
        if (!m_allPipelines[pipeline]->AddSourceBintr(m_allSources[source]))
        {   
            LOG_ERROR("Failed to add '" << source << "' to Pipeline '" << pipeline);
            return DSD_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
        }
        
        LOG_INFO("Source '" << source << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineSourceRemove(const std::string& pipeline, 
        const std::string& source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineStreamMuxAdd(const std::string& pipeline, 
        const std::string& streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allPipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        if (!m_allStreamMuxs[streammux])
        {   
            LOG_ERROR("Stream Mux name '" << streammux << "' was not found");
            return DSD_RESULT_STREAMMUX_NAME_NOT_FOUND;
        }
        if (!m_allPipelines[pipeline]->AddStreamMuxBintr(m_allStreamMuxs[streammux]))
        {   
            LOG_ERROR("Failed to add '" << streammux << "' to Pipeline '" << pipeline);
            return DSD_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
        }
        
        LOG_INFO("Stream Mux '" << streammux << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineStreamMuxRemove(const std::string& pipeline, 
        const std::string& streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineOsdAdd(const std::string& pipeline, 
        const std::string& osd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineOsdRemove(const std::string& pipeline, 
        const std::string& osd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineGieAdd(const std::string& pipeline, 
        const std::string& gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineGieRemove(const std::string& pipeline, 
        const std::string& gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineDisplayAdd(const std::string& pipeline, 
        const std::string& display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allPipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        if (!m_allDisplays[display])
        {   
            LOG_ERROR("Display name '" << display << "' was not found");
            return DSD_RESULT_DISPLAY_NAME_NOT_FOUND;
        }
        if (!m_allPipelines[pipeline]->AddDisplayBintr(m_allDisplays[display]))
        {   
            LOG_ERROR("Failed to add '" << display << "' to Pipeline '" << pipeline);
            return DSD_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
        }
        
        LOG_INFO("Display '" << display << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineDisplayRemove(const std::string& pipeline, 
        const std::string& display)
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
        
    void Driver::MainLoopRun()
    {
        LOG_FUNC();

        g_main_loop_run(m_pMainLoop);
    };
        
    bool Driver::HandleXWindowEvents()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);
        
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