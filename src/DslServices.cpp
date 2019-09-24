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
#include "DslApi.h"

DslReturnType dsl_source_new(const char* source, guint type, 
        gboolean live, guint width, guint height, guint fps_n, guint fps_d)
{
    return DSL::Services::GetServices()->SourceNew(source, type, live, 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_delete(const char* source)
{
    return DSL::Services::GetServices()->SourceDelete(source);
}

DslReturnType dsl_sink_new(const char* sink, guint displayId, 
    guint overlayId, guint offsetX, guint offsetY, guint width, guint height)
{
    return DSL::Services::GetServices()->SinkNew(sink, 
        displayId, overlayId, offsetX, offsetY, width, height);
}


DslReturnType dsl_sink_delete(const char* sink)
{
    return DSL::Services::GetServices()->SinkDelete(sink);
}


DslReturnType dsl_streammux_new(const char* streammux, 
    gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
{
    return DSL::Services::GetServices()->StreamMuxNew(streammux, live, 
        batchSize, batchTimeout, width, height);
}

DslReturnType dsl_streammux_delete(const char* streammux)
{
    return DSL::Services::GetServices()->StreamMuxDelete(streammux);
}

DslReturnType dsl_display_new(const char* display, 
        guint rows, guint columns, guint width, guint height)
{
    return DSL::Services::GetServices()->DisplayNew(display, rows, columns, width, height);
}

DslReturnType dsl_display_delete(const char* display)
{
    return DSL::Services::GetServices()->DisplayDelete(display);
}

DslReturnType dsl_gie_new(const char* gie, const char* configFilePath, 
            guint batchSize, guint interval, guint uniqueId, guint gpuId, const 
            std::string& modelEngineFile, const char* rawOutputDir)
{
    return DSL::Services::GetServices()->GieNew(gie, configFilePath, batchSize, 
        interval, uniqueId, gpuId, modelEngineFile, rawOutputDir);
}

DslReturnType dsl_gie_delete(const char* gie)
{
    return DSL::Services::GetServices()->GieDelete(gie);
}

DslReturnType dsl_pipeline_new(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineNew(pipeline);
}

DslReturnType dsl_pipeline_delete(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineDelete(pipeline);
}

DslReturnType dsl_pipeline_components_add(const char* pipeline, 
    const char** components)
{
    return DSL::Services::GetServices()->PipelineComponentsAdd(pipeline, components);
}

DslReturnType dsl_pipeline_components_remove(const char* pipeline, 
    const char** components)
{
    return DSL::Services::GetServices()->PipelineComponentsRemove(pipeline, components);
}

DslReturnType dsl_pipeline_pause(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineGetState(pipeline);
}

DslReturnType dsl_pipeline_play(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelinePlay(pipeline);
}

DslReturnType dsl_pipeline_get_state(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineGetState(pipeline);
}

void dsl_main_loop_run()
{
    g_main_loop_run(DSL::Services::GetServices()->m_pMainLoop);
}

namespace DSL
{
    // Initialize the Services's single instance pointer
    Services* Services::m_pInstatnce = NULL;
    
    Services* Services::GetServices()
    {
        
        if (!m_pInstatnce)
        {
            LOG_INFO("Services Initialization");
            
            m_pInstatnce = new Services();
        }
        return m_pInstatnce;
    }
        
    Services::Services()
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

    Services::~Services()
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

    DslReturnType Services::SourceNew(const char* source, guint type, 
        gboolean live, guint width, guint height, guint fps_n, guint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[source])
        {   
            LOG_ERROR("Source name '" << source << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[source] = new SourceBintr(
                source, type, live, width, height, fps_n, fps_d);
        }
        catch(...)
        {
            LOG_ERROR("New Source '" << source << "' threw exception on create");
            return DSL_RESULT_SOURCE_NEW_EXCEPTION;
        }
        LOG_INFO("new source '" << source << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceDelete(const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[source])
        {   
            LOG_ERROR("Source name '" << source << "' was not found");
            return DSL_RESULT_SOURCE_NAME_NOT_FOUND;
        }
        delete m_allComps[source];
        m_allComps.erase(source);

        LOG_INFO("Source '" << source << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkNew(const char* sink, guint displayId, guint overlayId,
        guint offsetX, guint offsetY, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[sink])
        {   
            LOG_ERROR("Sink name '" << sink << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[sink] = new SinkBintr(
                sink, displayId, overlayId, offsetX, offsetY, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << sink << "' threw exception on create");
            return DSL_RESULT_SINK_NEW_EXCEPTION;
        }
        LOG_INFO("new Sink '" << sink << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkDelete(const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[sink])
        {   
            LOG_ERROR("Sink name '" << sink << "' was not found");
            return DSL_RESULT_SINK_NAME_NOT_FOUND;
        }
        delete m_allComps[sink];
        m_allComps.erase(sink);
        
        LOG_INFO("Sink '" << sink << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::StreamMuxNew(const char* streammux, 
        gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[streammux])
        {   
            LOG_ERROR("Stream Mux name '" << streammux << "' is not unique");
            return DSL_RESULT_STREAMMUX_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[streammux] = new StreamMuxBintr(
                streammux, live, batchSize, batchTimeout, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New StreamMux '" << streammux << "' threw exception on create");
            return DSL_RESULT_STREAMMUX_NEW_EXCEPTION;
        }
        LOG_INFO("new Stream Mux '" << streammux << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::StreamMuxDelete(const char* streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[streammux])
        {   
            LOG_ERROR("Streammux name '" << streammux << "' was not found");
            return DSL_RESULT_STREAMMUX_NAME_NOT_FOUND;
        }
        delete m_allComps[streammux];
        m_allComps.erase(streammux);
        
        LOG_INFO("Streammux '" << streammux << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::DisplayNew(const char* display, 
        guint rows, guint columns, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[display])
        {   
            LOG_ERROR("Display name '" << display << "' is not unique");
            return DSL_RESULT_DISPLAY_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[display] = new DisplayBintr(
                display, m_pXDisplay, rows, columns, width, height);
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display New'" << display << "' threw exception on create");
            return DSL_RESULT_DISPLAY_NEW_EXCEPTION;
        }
        LOG_INFO("new Display '" << display << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::DisplayDelete(const char* display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[display])
        {   
            LOG_ERROR("Display name '" << display << "' was not found");
            return DSL_RESULT_DISPLAY_NAME_NOT_FOUND;
        }
        delete m_allComps[display];
        m_allComps.erase(display);

        LOG_INFO("Display '" << display << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::GieNew(const char* gie, 
        const char* configFilePath, guint batchSize, 
        guint interval, guint uniqueId, guint gpuId, const 
        std::string& modelEngineFile, const char* rawOutputDir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' is not unique");
            return DSL_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[gie] = new GieBintr(gie, configFilePath, batchSize, 
                interval, uniqueId, gpuId, modelEngineFile, rawOutputDir);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << gie << "' threw exception on create");
            return DSL_RESULT_GIE_NEW_EXCEPTION;
        }
        LOG_INFO("new GIE '" << gie << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::GieDelete(const char* gie)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' was not found");
            return DSL_RESULT_GIE_NAME_NOT_FOUND;
        }
        delete m_allComps[gie];
        m_allComps.erase(gie);

        LOG_INFO("GIE '" << gie << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineNew(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' is not unique");
            return DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_allComps[pipeline] = new PipelineBintr(pipeline);
        }
        catch(...)
        {
            LOG_ERROR("New Pipeline '" << pipeline << "' threw exception on create");
            return DSL_RESULT_PIPELINE_NEW_EXCEPTION;
        }
        LOG_INFO("new PIPELINE '" << pipeline << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineDelete(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        delete m_allComps[pipeline];
        m_allComps.erase(pipeline);

        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineSourceAdd(const char* pipeline, 
        const char* source)
    {
        LOG_FUNC();

        ((PipelineBintr*)m_allComps[pipeline])->AddSourceBintr((SourceBintr*)m_allComps[source]);
        
        LOG_INFO("Source '" << source << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineSourceRemove(const char* pipeline, 
        const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DslReturnType Services::PipelineStreamMuxAdd(const char* pipeline, 
        const char* streammux)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        if (!m_allComps[streammux])
        {   
            LOG_ERROR("Stream Mux name '" << streammux << "' was not found");
            return DSL_RESULT_STREAMMUX_NAME_NOT_FOUND;
        }
        ((PipelineBintr*)m_allComps[pipeline])->AddStreamMuxBintr((StreamMuxBintr*)m_allComps[streammux]);
        
        LOG_INFO("Stream Mux '" << streammux << "'was added to Pipeline '" << pipeline << "' successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelinePause(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }

    DslReturnType Services::PipelinePlay(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        // flush the output buffer and then wait until all requests have been 
        // received and processed by the X server. TRUE = Discard all queued events
        XSync(m_pXDisplay, TRUE);       

        if (!m_allComps[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }

        if (!((PipelineBintr*)m_allComps[pipeline])->Play())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_PLAY;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineGetState(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
        
    bool Services::HandleXWindowEvents()
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
        Services* pServices = Services::GetServices();
        
        return true;
    }
    

    static gpointer XWindowEventThread(gpointer arg)
    {
        Services* pServices = Services::GetServices();

        pServices->HandleXWindowEvents();
       
        return NULL;
    }

} // namespace 