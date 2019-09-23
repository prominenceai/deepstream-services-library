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

DsdReturnType dsd_source_new(const char* source, guint type, 
        gboolean live, guint width, guint height, guint fps_n, guint fps_d)
{
    return DSD::Driver::GetDriver()->SourceNew(source, type, live, 
        width, height, fps_n, fps_d);
}

DsdReturnType dsd_source_delete(const char* source)
{
    return DSD::Driver::GetDriver()->SourceDelete(source);
}

DsdReturnType dsd_sink_new(const char* sink, guint displayId, 
    guint overlayId, guint offsetX, guint offsetY, guint width, guint height)
{
    return DSD::Driver::GetDriver()->SinkNew(sink, 
        displayId, overlayId, offsetX, offsetY, width, height);
}


DsdReturnType dsd_sink_delete(const char* sink)
{
    return DSD::Driver::GetDriver()->SinkDelete(sink);
}


DsdReturnType dsd_streammux_new(const char* streammux, 
    gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
{
    return DSD::Driver::GetDriver()->StreamMuxNew(streammux, live, 
        batchSize, batchTimeout, width, height);
}

DsdReturnType dsd_streammux_delete(const char* streammux)
{
    return DSD::Driver::GetDriver()->StreamMuxDelete(streammux);
}

DsdReturnType dsd_display_new(const char* display, 
        guint rows, guint columns, guint width, guint height)
{
    return DSD::Driver::GetDriver()->DisplayNew(display, rows, columns, width, height);
}

DsdReturnType dsd_display_delete(const char* display)
{
    return DSD::Driver::GetDriver()->DisplayDelete(display);
}

DsdReturnType dsd_gie_new(const char* gie, const char* configFilePath, 
            guint batchSize, guint interval, guint uniqueId, guint gpuId, const 
            std::string& modelEngineFile, const char* rawOutputDir)
{
    return DSD::Driver::GetDriver()->GieNew(gie, configFilePath, batchSize, 
        interval, uniqueId, gpuId, modelEngineFile, rawOutputDir);
}

DsdReturnType dsd_gie_delete(const char* gie)
{
    return DSD::Driver::GetDriver()->GieDelete(gie);
}

DsdReturnType dsd_pipeline_new(const char* pipeline)
{
    return DSD::Driver::GetDriver()->PipelineNew(pipeline);
}

DsdReturnType dsd_pipeline_delete(const char* pipeline)
{
    return DSD::Driver::GetDriver()->PipelineDelete(pipeline);
}

DsdReturnType dsd_pipeline_components_add(const char* pipeline, 
    const char** components)
{
    return DSD::Driver::GetDriver()->PipelineComponentsAdd(pipeline, components);
}

DsdReturnType dsd_pipeline_components_remove(const char* pipeline, 
    const char** components)
{
    return DSD::Driver::GetDriver()->PipelineComponentsRemove(pipeline, components);
}

DsdReturnType dsd_pipeline_pause(const char* pipeline)
{
    return DSD::Driver::GetDriver()->PipelineGetState(pipeline);
}

DsdReturnType dsd_pipeline_play(const char* pipeline)
{
    return DSD::Driver::GetDriver()->PipelinePlay(pipeline);
}

DsdReturnType dsd_pipeline_get_state(const char* pipeline)
{
    return DSD::Driver::GetDriver()->PipelineGetState(pipeline);
}

void dsd_main_loop_run()
{
    g_main_loop_run(DSD::Driver::GetDriver()->m_pMainLoop);
}

namespace DSD
{
    // Initialize the Driver's single instance pointer
    Driver* Driver::m_pInstatnce = NULL;
    
    Driver* Driver::GetDriver()
    {
        
        if (!m_pInstatnce)
        {
            LOG_INFO("Driver Initialization");
            
            m_pInstatnce = new Driver();
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

    DsdReturnType Driver::SourceNew(const char* source, guint type, 
        gboolean live, guint width, guint height, guint fps_n, guint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[source])
        {   
            LOG_ERROR("Source name '" << source << "' is not unique");
            return DSD_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[source] = new SourceBintr(
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
    
    DsdReturnType Driver::SourceDelete(const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[source])
        {   
            LOG_ERROR("Source name '" << source << "' was not found");
            return DSD_RESULT_SOURCE_NAME_NOT_FOUND;
        }
        delete m_allComps[source];
        m_allComps.erase(source);

        LOG_INFO("Source '" << source << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::SinkNew(const char* sink, guint displayId, guint overlayId,
        guint offsetX, guint offsetY, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[sink])
        {   
            LOG_ERROR("Sink name '" << sink << "' is not unique");
            return DSD_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[sink] = new SinkBintr(
                sink, displayId, overlayId, offsetX, offsetY, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << sink << "' threw exception on create");
            return DSD_RESULT_SINK_NEW_EXCEPTION;
        }
        LOG_INFO("new Sink '" << sink << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::SinkDelete(const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[sink])
        {   
            LOG_ERROR("Sink name '" << sink << "' was not found");
            return DSD_RESULT_SINK_NAME_NOT_FOUND;
        }
        delete m_allComps[sink];
        m_allComps.erase(sink);
        
        LOG_INFO("Sink '" << sink << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::StreamMuxNew(const char* streammux, 
        gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[streammux])
        {   
            LOG_ERROR("Stream Mux name '" << streammux << "' is not unique");
            return DSD_RESULT_STREAMMUX_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[streammux] = new StreamMuxBintr(
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
    
    DsdReturnType Driver::StreamMuxDelete(const char* streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[streammux])
        {   
            LOG_ERROR("Streammux name '" << streammux << "' was not found");
            return DSD_RESULT_STREAMMUX_NAME_NOT_FOUND;
        }
        delete m_allComps[streammux];
        m_allComps.erase(streammux);
        
        LOG_INFO("Streammux '" << streammux << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::DisplayNew(const char* display, 
        guint rows, guint columns, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[display])
        {   
            LOG_ERROR("Display name '" << display << "' is not unique");
            return DSD_RESULT_DISPLAY_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[display] = new DisplayBintr(
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
    
    DsdReturnType Driver::DisplayDelete(const char* display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[display])
        {   
            LOG_ERROR("Display name '" << display << "' was not found");
            return DSD_RESULT_DISPLAY_NAME_NOT_FOUND;
        }
        delete m_allComps[display];
        m_allComps.erase(display);

        LOG_INFO("Display '" << display << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::GieNew(const char* gie, 
        const char* configFilePath, guint batchSize, 
        guint interval, guint uniqueId, guint gpuId, const 
        std::string& modelEngineFile, const char* rawOutputDir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' is not unique");
            return DSD_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[gie] = new GieBintr(gie, configFilePath, batchSize, 
                interval, uniqueId, gpuId, modelEngineFile, rawOutputDir);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << gie << "' threw exception on create");
            return DSD_RESULT_GIE_NEW_EXCEPTION;
        }
        LOG_INFO("new GIE '" << gie << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::GieDelete(const char* gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' was not found");
            return DSD_RESULT_GIE_NAME_NOT_FOUND;
        }
        delete m_allComps[gie];
        m_allComps.erase(gie);

        LOG_INFO("GIE '" << gie << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineNew(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' is not unique");
            return DSD_RESULT_PIPELINE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[pipeline] = new PipelineBintr(pipeline);
        }
        catch(...)
        {
            LOG_ERROR("New Pipeline '" << pipeline << "' threw exception on create");
            return DSD_RESULT_PIPELINE_NEW_EXCEPTION;
        }
        LOG_INFO("new PIPELINE '" << pipeline << "' created successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineDelete(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        delete m_allComps[pipeline];
        m_allComps.erase(pipeline);

        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineSourceAdd(const char* pipeline, 
        const char* source)
    {
        LOG_FUNC();

        ((PipelineBintr*)m_allComps[pipeline])->AddSourceBintr((SourceBintr*)m_allComps[source]);
        
        LOG_INFO("Source '" << source << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineSourceRemove(const char* pipeline, 
        const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineStreamMuxAdd(const char* pipeline, 
        const char* streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        if (!m_allComps[streammux])
        {   
            LOG_ERROR("Stream Mux name '" << streammux << "' was not found");
            return DSD_RESULT_STREAMMUX_NAME_NOT_FOUND;
        }
        ((PipelineBintr*)m_allComps[pipeline])->AddStreamMuxBintr((StreamMuxBintr*)m_allComps[streammux]);
        
        LOG_INFO("Stream Mux '" << streammux << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }

    
    DsdReturnType Driver::PipelineStreamMuxRemove(const char* pipeline, 
        const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineSinkAdd(const char* pipeline, 
        const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        if (!m_allComps[sink])
        {   
            LOG_ERROR("Sink name '" << sink << "' was not found");
            return DSD_RESULT_SOURCE_NAME_NOT_FOUND;
        }
        ((PipelineBintr*)m_allComps[pipeline])->AddSinkBintr((SinkBintr*)m_allComps[sink]);
        
        LOG_INFO("Sink '" << sink << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineSinkRemove(const char* pipeline, 
        const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    
    DsdReturnType Driver::PipelineOsdAdd(const char* pipeline, 
        const char* osd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineOsdRemove(const char* pipeline, 
        const char* osd)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineGieAdd(const char* pipeline, 
        const char* gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineGieRemove(const char* pipeline, 
        const char* gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelineDisplayAdd(const char* pipeline, 
        const char* display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        ((PipelineBintr*)m_allComps[pipeline])->AddDisplayBintr((DisplayBintr*)m_allComps[display]);
        
        LOG_INFO("Display '" << display << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineDisplayRemove(const char* pipeline, 
        const char* display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
        
    DsdReturnType Driver::PipelinePause(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DsdReturnType Driver::PipelinePlay(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        // flush the output buffer and then wait until all requests have been 
        // received and processed by the X server. TRUE = Discard all queued events
        XSync(m_pXDisplay, TRUE);       

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSD_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        
        if (!((PipelineBintr*)m_allComps[pipeline])->Play())
        {
            return DSD_RESULT_PIPELINE_FAILED_TO_PLAY;
        }

        return DSD_RESULT_SUCCESS;
    }
    
    DsdReturnType Driver::PipelineGetState(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSD_RESULT_API_NOT_IMPLEMENTED;
    }
        
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