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
#include "DslServices.h"

GST_DEBUG_CATEGORY(GST_CAT_DSL);

#define RETURN_IF_PIPELINE_NAME_NOT_FOUND(pipelines, name) do \
{ \
    if (!pipelines[name]) \
    { \
        LOG_ERROR("Pipeline name '" << pipeline << "' was not found"); \
        return DSL_RESULT_PIPELINE_NAME_NOT_FOUND; \
    } \
}while(0); 
    
#define RETURN_IF_COMPONENT_NAME_NOT_FOUND(components, name) do \
{ \
    if (!components[name]) \
    { \
        LOG_ERROR("Component name '" << component << "' was not found"); \
        return DSL_RESULT_COMPONENT_NAME_NOT_FOUND; \
    } \
}while(0); 
    

DslReturnType dsl_source_csi_new(const char* source, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    return DSL::Services::GetServices()->SourceCsiNew(source, 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_uri_new(const char* source, 
    const char* uri, uint cudadec_mem_type, uint intra_decode)
{
    return DSL::Services::GetServices()->SourceUriNew(source,
        uri, cudadec_mem_type, intra_decode);
}

DslReturnType dsl_sink_new(const char* sink, uint displayId, 
    uint overlayId, uint offsetX, uint offsetY, uint width, uint height)
{
    return DSL::Services::GetServices()->SinkNew(sink, 
        displayId, overlayId, offsetX, offsetY, width, height);
}

DslReturnType dsl_osd_new(const char* osd, boolean isClockEnabled)
{
    return DSL::Services::GetServices()->OsdNew(osd, isClockEnabled);
}

DslReturnType dsl_display_new(const char* display, 
        uint rows, uint columns, uint width, uint height)
{
    return DSL::Services::GetServices()->DisplayNew(display, rows, columns, width, height);
}

DslReturnType dsl_gie_new(const char* gie, const char* inferConfigFile, 
    uint batchSize, uint interval, uint uniqueId, uint gpuId, 
    const char* modelEngineFile, const char* rawOutputDir)
{
    return DSL::Services::GetServices()->GieNew(gie, inferConfigFile, batchSize, 
        interval, uniqueId, gpuId, modelEngineFile, rawOutputDir);
}

DslReturnType dsl_component_delete(const char* component)
{
    return DSL::Services::GetServices()->ComponentDelete(component);
}

DslReturnType dsl_component_delete_many(const char** names)
{
    return DSL::Services::GetServices()->ComponentDeleteMany(names);
}

DslReturnType dsl_component_delete_all()
{
    return DSL::Services::GetServices()->ComponentDeleteAll();
}

uint dsl_component_list_size()
{
    return DSL::Services::GetServices()->ComponentListSize();
}

const char** dsl_component_list_all()
{
    return DSL::Services::GetServices()->ComponentListAll();
}

DslReturnType dsl_pipeline_new(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineNew(pipeline);
}

DslReturnType dsl_pipeline_delete(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineDelete(pipeline);
}

DslReturnType dsl_pipeline_delete_many(const char** pipelines)
{
    return DSL::Services::GetServices()->PipelineDeleteMany(pipelines);
}

DslReturnType dsl_pipeline_delete_all()
{
    return DSL::Services::GetServices()->PipelineDeleteAll();
}

uint dsl_pipeline_list_size()
{
    return DSL::Services::GetServices()->PipelineListSize();
}

const char** dsl_pipeline_list_all()
{
    return DSL::Services::GetServices()->PipelineListAll();
}

DslReturnType dsl_pipeline_component_add(const char* pipeline, 
    const char* component)
{
    return DSL::Services::GetServices()->PipelineComponentAdd(pipeline, component);
}

DslReturnType dsl_pipeline_component_add_many(const char* pipeline, 
    const char** components)
{
    return DSL::Services::GetServices()->PipelineComponentAddMany(pipeline, components);
}

DslReturnType dsl_pipeline_component_remove(const char* pipeline, 
    const char* component)
{
    return DSL::Services::GetServices()->PipelineComponentRemove(pipeline, component);
}

DslReturnType dsl_pipeline_component_remove_many(const char* pipeline, 
    const char** components)
{
    return DSL::Services::GetServices()->PipelineComponentRemoveMany(pipeline, components);
}

DslReturnType dsl_pipeline_streammux_properties_set(const char* pipeline,
    boolean areSourcesLive, uint batchSize, uint batchTimeout, uint width, uint height)
{
    return DSL::Services::GetServices()->PipelineStreamMuxPropertiesSet(pipeline,
        areSourcesLive, batchSize, batchTimeout, width, height);
}
 
DslReturnType dsl_pipeline_pause(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelinePause(pipeline);
}

DslReturnType dsl_pipeline_play(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelinePlay(pipeline);
}

DslReturnType dsl_pipeline_get_state(const char* pipeline)
{
    return DSL::Services::GetServices()->PipelineGetState(pipeline);
}

DslReturnType dsl_pipeline_dump_to_dot(const char* pipeline, char* filename)
{
    return DSL::Services::GetServices()->PipelineDumpToDot(pipeline, filename);
}

DslReturnType dsl_pipeline_dump_to_dot_with_ts(const char* pipeline, char* filename)
{
    return DSL::Services::GetServices()->PipelineDumpToDotWithTs(pipeline, filename);    
}

DslReturnType dsl_pipeline_state_change_listener_add(const char* pipeline, 
    dsl_state_change_listener_cb listener, void* userdata)
{
    return DSL::Services::GetServices()->
        PipelineStateChangeListenerAdd(pipeline, listener, userdata);
}

DslReturnType dsl_pipeline_state_change_listener_remove(const char* pipeline, 
    dsl_state_change_listener_cb listener)
{
    return DSL::Services::GetServices()->
        PipelineStateChangeListenerRemove(pipeline, listener);
}

DslReturnType dsl_pipeline_display_event_handler_add(const char* pipeline, 
    dsl_display_event_handler_cb handler, void* userdata)
{
    return DSL::Services::GetServices()->
        PipelineDisplayEventHandlerAdd(pipeline, handler, userdata);
}    

DslReturnType dsl_pipeline_display_event_handler_remove(const char* pipeline, 
    dsl_display_event_handler_cb handler)
{
    return DSL::Services::GetServices()->
        PipelineDisplayEventHandlerRemove(pipeline, handler);
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
    std::string Services::m_modelFileDir = DS_MODELS_DIR;
    std::string Services::m_streamFileDir = DS_STREAMS_DIR;
    
    Services* Services::GetServices()
    {
        
        // one time initialization of the single instance pointer
        if (!m_pInstatnce)
        {
            // If gst has not been initialized by the client software
            if (!gst_is_initialized())
            {
                int argc = 0;
                char** argv = NULL;
                
                // initialize the GStreamer library
                gst_init(&argc, &argv);
            }

            // Initialize the single debug category used by the lib
            GST_DEBUG_CATEGORY_INIT(GST_CAT_DSL, "DSL", 0, "DeepStream Services");
            
            // Safe to start logging
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
        g_mutex_init(&m_servicesMutex);
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
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
            
            if (m_pMainLoop)
            {
                LOG_WARN("Main loop is still running!");
                g_main_loop_quit(m_pMainLoop);
            }
        }
        
        g_mutex_clear(&m_displayMutex);
        g_mutex_clear(&m_servicesMutex);
    }
    
    DslReturnType Services::SourceCsiNew(const char* source,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[source])
        {   
            LOG_ERROR("Source name '" << source << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[source] = std::shared_ptr<CsiSourceBintr>(new CsiSourceBintr(
                source, width, height, fps_n, fps_d));
        }
        catch(...)
        {
            LOG_ERROR("New CSI Source '" << source << "' threw exception on create");
            return DSL_RESULT_SOURCE_NEW_EXCEPTION;
        }
        LOG_INFO("new CSI Source '" << source << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceUriNew(const char* source,
        const char* uri, uint cudadecMemType, uint intraDecode)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[source])
        {   
            LOG_ERROR("Source name '" << source << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }

        std::string streamFilePathSpec = m_streamFileDir;
        streamFilePathSpec.append("/");
        streamFilePathSpec.append(uri);
        
        std::ifstream streamFile(streamFilePathSpec.c_str());
        if (!streamFile.good())
        {
            LOG_ERROR("URI stream file not found");
            return DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND;
        }
        streamFilePathSpec.insert(0,"file:");
        LOG_INFO("URI stream file: " << streamFilePathSpec);
        try
        {
            m_components[source] = std::shared_ptr<UriSourceBintr>(new UriSourceBintr(
                source, streamFilePathSpec.c_str(), cudadecMemType, intraDecode));
        }
        catch(...)
        {
            LOG_ERROR("New URI Source '" << source << "' threw exception on create");
            return DSL_RESULT_SOURCE_NEW_EXCEPTION;
        }
        LOG_INFO("new URI Source '" << source << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkNew(const char* sink, uint displayId, uint overlayId,
        uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
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
    
    DslReturnType Services::OsdNew(const char* osd, boolean isClockEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
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
        uint rows, uint columns, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
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
        const char* inferConfigFile, uint batchSize, 
        uint interval, uint uniqueId, uint gpuId, 
        const char* modelEngineFile, const char* rawOutputDir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
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
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
        
        if (m_components[component]->IsInUse())
        {
            LOG_INFO("Component '" << component << "' is in use");
            return DSL_RESULT_COMPONENT_IN_USE;
        }

        m_components.erase(component);

        LOG_INFO("Component '" << component << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentDeleteMany(const char** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // iterate through the list of provided components to verifiy the
        // existence of each... AND that each is not owned by a pipeline 
        // before making any updates to the list of components.
        for (const char** component = components; *component; component++)
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, *component);

            if (m_components[*component]->IsInUse())
            {   
                LOG_ERROR("Component '" << *component << "' is currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
        }
        LOG_DEBUG("All listed components found and un-owned");
        
        // iterate through the list a second time erasing each
        for (const char** component = components; *component; component++)
        {
            m_components.erase(*component);
        }

        LOG_INFO("All Components deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::ComponentDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        LOG_INFO("All Components deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::ComponentListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_components.size();
    }
    
    const char** Services::ComponentListAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        m_componentNames.clear();
        
        // reserve to avoid resizing - plus 1 for the NULL terminator
        m_componentNames.reserve(m_components.size() + 1);
        for(auto const& imap: m_components)
        {
            m_componentNames.push_back(imap.first.c_str());
        }
        m_componentNames.push_back(NULL);
        
        return (const char**)&m_componentNames[0];
    }
    
    DslReturnType Services::PipelineNew(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

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
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        m_pipelines.erase(pipeline);

        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineDeleteMany(const char** pipelines)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // iterate through the list of provided pipelines to verifiy the
        // existence of each before making any updates to the list of pipelines.
        for (const char** pipeline = pipelines; *pipeline; pipeline++)
        {
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, *pipeline);
        }
        LOG_DEBUG("All listed pipelines found");
        
        // iterate through the list a second time erasing each
        for (const char** pipeline = pipelines; *pipeline; pipeline++)
        {
            m_pipelines.erase(*pipeline);
        }

        LOG_INFO("All Pipelines deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        m_pipelines.clear();

        return DSL_RESULT_SUCCESS;
    }

    uint Services::PipelineListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_pipelines.size();
    }
    
    const char** Services::PipelineListAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        m_pipelineNames.clear();
        
        // reserve to avoid resizing - plus 1 for the NULL terminator
        m_pipelineNames.reserve(m_pipelines.size() + 1);
        for(auto const& imap: m_pipelines)
        {
            m_pipelineNames.push_back(imap.first.c_str());
        }
        m_pipelineNames.push_back(NULL);
        
        return (const char**)&m_pipelineNames[0];
    }
    
    DslReturnType Services::PipelineComponentAdd(const char* pipeline, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

        try
        {
            m_components[component]->AddToParent(m_pipelines[pipeline]);
            LOG_INFO("Component '" << component 
                << "' was added to Pipeline '" << pipeline << "' successfully");
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw exception adding component '" << component << "'");
            return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
        }
    }    
        
    
    DslReturnType Services::PipelineComponentAddMany(const char* pipeline, 
        const char** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        // iterate through the list of provided components to verifiy the
        //  existence of each - before making any updates to the pipeline.
        for (const char** component = components; *component; component++)
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, *component);
        }
        LOG_DEBUG("All listed components found");
        
        // ensure that all current commponents are unlinked first
        m_pipelines[pipeline]->UnlinkComponents();

        // iterate through the list of provided components a second time
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
        
        // link all components, previous and those just added
        m_pipelines[pipeline]->LinkComponents();

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineComponentRemove(const char* pipeline, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DslReturnType Services::PipelineComponentRemoveMany(const char* pipeline, 
        const char** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DslReturnType Services::PipelineStreamMuxPropertiesSet(const char* pipeline,
        boolean areSourcesLive, uint batchSize, uint batchTimeout, uint width, uint height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }

    DslReturnType Services::PipelinePlay(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        // flush the output buffer and then wait until all requests have been 
        // received and processed by the X server. TRUE = Discard all queued events
        XSync(m_pXDisplay, TRUE);       


        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->Play())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_PLAY;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineGetState(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
        
    DslReturnType Services::PipelineDumpToDot(const char* pipeline, char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        if (!m_pipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[pipeline]->DumpToDot(filename);
        
        return DSL_RESULT_SUCCESS;
    }   
    
    DslReturnType Services::PipelineDumpToDotWithTs(const char* pipeline, char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        if (!m_pipelines[pipeline])
        {   
            LOG_ERROR("Pipeline name '" << pipeline << "' was not found");
            return DSL_RESULT_PIPELINE_NAME_NOT_FOUND;
        }
        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[pipeline]->DumpToDot(filename);

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStateChangeListenerAdd(const char* pipeline, 
        dsl_state_change_listener_cb listener, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        return m_pipelines[pipeline]->AddStateChangeListener(listener, userdata);
    }
    
    DslReturnType Services::PipelineStateChangeListenerRemove(const char* pipeline, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        return m_pipelines[pipeline]->RemoveStateChangeListener(listener);
    }
    
    DslReturnType Services::PipelineDisplayEventHandlerAdd(const char* pipeline, 
        dsl_display_event_handler_cb handler, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        return m_pipelines[pipeline]->AddDisplayEventHandler(handler, userdata);
    }
        

    DslReturnType Services::PipelineDisplayEventHandlerRemove(const char* pipeline, 
        dsl_display_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        return m_pipelines[pipeline]->RemoveDisplayEventHandler(handler);
    }
    
    bool Services::HandleXWindowEvents()
    {
//        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
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