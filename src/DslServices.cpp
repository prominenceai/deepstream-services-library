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

DslReturnType dsl_source_new(const char* source, gboolean live, 
    guint width, guint height, guint fps_n, guint fps_d)
{
    return DSL::Services::GetServices()->SourceNew(source, live, 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_sink_new(const char* sink, guint displayId, 
    guint overlayId, guint offsetX, guint offsetY, guint width, guint height)
{
    return DSL::Services::GetServices()->SinkNew(sink, 
        displayId, overlayId, offsetX, offsetY, width, height);
}

DslReturnType dsl_osd_new(const char* osd, gboolean isClockEnabled)
{
    return DSL::Services::GetServices()->OsdNew(osd, isClockEnabled);
}

DslReturnType dsl_display_new(const char* display, 
        guint rows, guint columns, guint width, guint height)
{
    return DSL::Services::GetServices()->DisplayNew(display, rows, columns, width, height);
}

DslReturnType dsl_gie_new(const char* gie, const char* inferConfigFile, 
    guint batchSize, guint interval, guint uniqueId, guint gpuId, 
    const char* modelEngineFile, const char* rawOutputDir)
{
    return DSL::Services::GetServices()->GieNew(gie, inferConfigFile, batchSize, 
        interval, uniqueId, gpuId, modelEngineFile, rawOutputDir);
}

DslReturnType dsl_component_delete(const char* component)
{
    return DSL::Services::GetServices()->ComponentDelete(component);
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

DslReturnType dsl_pipeline_streammux_properties_set(const char* pipeline,
    gboolean areSourcesLive, guint batchSize, guint batchTimeout, guint width, guint height)
{
    return DSL::Services::GetServices()->PipelineStreamMuxPropertiesSet(pipeline,
        areSourcesLive, batchSize, batchTimeout, width, height);
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

    std::string Services::m_configFileDir = DS_CONFIG_DIR;
    std::string Services::m_modelFileDir = DS_MODEL_DIR;
    
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
        m_pXWindowEventThread = g_thread_new("dsl-x-window-event-thread",
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

    DslReturnType Services::SourceNew(const char* source, gboolean live, 
        guint width, guint height, guint fps_n, guint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_components[source])
        {   
            LOG_ERROR("Source name '" << source << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[source] = std::shared_ptr<SourceBintr>(new SourceBintr(
                source, live, width, height, fps_n, fps_d));
        }
        catch(...)
        {
            LOG_ERROR("New Source '" << source << "' threw exception on create");
            return DSL_RESULT_SOURCE_NEW_EXCEPTION;
        }
        LOG_INFO("new source '" << source << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
                
    DslReturnType Services::SinkNew(const char* sink, guint displayId, guint overlayId,
        guint offsetX, guint offsetY, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_components[sink])
        {   
            LOG_ERROR("Sink name '" << sink << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[sink] = std::shared_ptr<Bintr>(new SinkBintr(
                sink, displayId, overlayId, offsetX, offsetY, width, height));
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << sink << "' threw exception on create");
            return DSL_RESULT_SINK_NEW_EXCEPTION;
        }
        LOG_INFO("new Sink '" << sink << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdNew(const char* osd, gboolean isClockEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_components[osd])
        {   
            LOG_ERROR("OSD name '" << osd << "' is not unique");
            return DSL_RESULT_OSD_NAME_NOT_UNIQUE;
        }
        try
        {   
            m_components[osd] = std::shared_ptr<Bintr>(new OsdBintr(
                osd, isClockEnabled));
        }
        catch(...)
        {
            LOG_ERROR("New OSD '" << osd << "' threw exception on create");
            return DSL_RESULT_SINK_NEW_EXCEPTION;
        }
        LOG_INFO("new OSD '" << osd << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::DisplayNew(const char* display, 
        guint rows, guint columns, guint width, guint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_components[display])
        {   
            LOG_ERROR("Display name '" << display << "' is not unique");
            return DSL_RESULT_DISPLAY_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[display] = std::shared_ptr<Bintr>(new DisplayBintr(
                display, m_pXDisplay, rows, columns, width, height));
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display New'" << display << "' threw exception on create");
            return DSL_RESULT_DISPLAY_NEW_EXCEPTION;
        }
        LOG_INFO("new Display '" << display << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::GieNew(const char* gie, 
        const char* inferConfigFile, guint batchSize, 
        guint interval, guint uniqueId, guint gpuId, 
        const char* modelEngineFile, const char* rawOutputDir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_components[gie])
        {   
            LOG_ERROR("GIE name '" << gie << "' is not unique");
            return DSL_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        
        std::string configFilePathSpec = m_configFileDir;
        configFilePathSpec.append("/");
        configFilePathSpec.append(inferConfigFile);
        LOG_INFO("Infer config file: " << configFilePathSpec);
        
        std::ifstream configFile(configFilePathSpec.c_str());
        if (!configFile.good())
        {
            LOG_ERROR("Infer Config File not found");
            return DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND;
        }
        
        std::string modelFilePathSpec = m_modelFileDir;
        modelFilePathSpec.append("/");
        modelFilePathSpec.append(modelEngineFile);
        LOG_INFO("Model engine file: " << modelFilePathSpec);
        
        std::ifstream modelFile(modelFilePathSpec.c_str());
        if (!modelFile.good())
        {
            LOG_ERROR("Model Engine File not found");
            return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
        }

        try
        {
            m_components[gie] = std::shared_ptr<Bintr>(new GieBintr(gie, 
                configFilePathSpec.c_str(), batchSize, interval, uniqueId, 
                gpuId, modelFilePathSpec.c_str(), rawOutputDir));
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << gie << "' threw exception on create");
            return DSL_RESULT_GIE_NEW_EXCEPTION;
        }
        LOG_INFO("new GIE '" << gie << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentDelete(const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_components[component])
        {   
            LOG_ERROR("Component name '" << component << "' was not found");
            return DSL_RESULT_COMPONENT_NAME_NOT_FOUND;
        }
        m_components.erase(component);

        LOG_INFO("Component '" << component << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineNew(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (m_pipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' is not unique");
            return DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_pipelines[pipeline] = std::shared_ptr<PipelineBintr>(new PipelineBintr(pipeline));
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

        if (!m_pipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        m_pipelines.erase(pipeline);

        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
        
    
    DslReturnType Services::PipelineComponentsAdd(const char* pipeline, 
        const char** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        if (!m_pipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        
        // iterate through the list of components to verifiy the existence
        //  of each... before making any updates to the pipeline.
        for (const char** component = components; *component; component++)
        {

            if (!m_components[*component])
            {   
                LOG_ERROR("Component name '" << *component << "' was not found");
                return DSL_RESULT_COMPONENT_NAME_NOT_FOUND;
            }
        }
        LOG_INFO("All listed components found");

        // iterate through the list of components a second time
        // adding each to the named pipeline individually.
        for (const char** component = components; *component; component++)
        {
            try
            {
                m_components[*component]->AddToParent(m_pipelines[pipeline]);
                LOG_INFO("Component '" << *component 
                    << "' was added to Pipeline '" << pipeline << "' successfully");
            }
            catch(...)
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' threw exception adding component '" << *component << "'");
                return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
            }
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineComponentsRemove(const char* pipeline, 
        const char** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_driverMutex);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DslReturnType Services::PipelineStreamMuxPropertiesSet(const char* pipeline,
        gboolean areSourcesLive, guint batchSize, guint batchTimeout, guint width, guint height)    
    {
        if (!m_pipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        try
        {
            m_pipelines[pipeline]->SetStreamMuxProperties(areSourcesLive, 
                batchSize, batchTimeout, width, height);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer properties");
            return DSL_RESULT_PIPELINE_STREAMMUX_SETUP_FAILED;
        }
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

        if (!m_components[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_components[pipeline])->Play())
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