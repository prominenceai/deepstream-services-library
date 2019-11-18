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

DslReturnType dsl_source_csi_new(const wchar_t* name, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    return DSL::Services::GetServices()->SourceCsiNew(name, 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_uri_new(const wchar_t* name, 
    const wchar_t* uri, uint cudadec_mem_type, uint intra_decode, uint dropFrameInterval)
{
    return DSL::Services::GetServices()->SourceUriNew(name,
        uri, cudadec_mem_type, intra_decode, dropFrameInterval);
}

boolean dsl_source_is_live(const wchar_t* name)
{
    return DSL::Services::GetServices()->SourceIsLive(name);
}

uint dsl_source_get_num_in_use()
{
    return DSL::Services::GetServices()->GetNumSourceInUse();
}

uint dsl_source_get_num_in_use_max()
{
    return DSL::Services::GetServices()->GetNumSourceInUseMax();
}

void dsl_source_set_num_in_use_max(uint max)
{
    return DSL::Services::GetServices()->SetNumSourceInUseMax(max);
}

DslReturnType dsl_sink_overlay_new(const wchar_t* name,
    uint offsetX, uint offsetY, uint width, uint height)
{
    return DSL::Services::GetServices()->OverlaySinkNew(name, 
        offsetX, offsetY, width, height);
}

DslReturnType dsl_osd_new(const wchar_t* name, boolean isClockEnabled)
{
    return DSL::Services::GetServices()->OsdNew(name, isClockEnabled);
}

DslReturnType dsl_display_new(const wchar_t* name, uint width, uint height)
{
    return DSL::Services::GetServices()->DisplayNew(name, width, height);
}

DslReturnType dsl_gie_primary_new(const wchar_t* name, const wchar_t* inferConfigFile,
    const wchar_t* modelEngineFile, uint interval, uint uniqueId)
{
    return DSL::Services::GetServices()->PrimaryGieNew(name, inferConfigFile,
        modelEngineFile, interval, uniqueId);
}

DslReturnType dsl_component_delete(const wchar_t* component)
{
    return DSL::Services::GetServices()->ComponentDelete(component);
}

DslReturnType dsl_component_delete_many(const wchar_t** names)
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

const wchar_t** dsl_component_list_all()
{
    return DSL::Services::GetServices()->ComponentListAll();
}

DslReturnType dsl_pipeline_new(const wchar_t* pipeline)
{
    return DSL::Services::GetServices()->PipelineNew(pipeline);
}

DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines)
{
    return DSL::Services::GetServices()->PipelineNewMany(pipelines);
}

DslReturnType dsl_pipeline_delete(const wchar_t* pipeline)
{
    return DSL::Services::GetServices()->PipelineDelete(pipeline);
}

DslReturnType dsl_pipeline_delete_many(const wchar_t** pipelines)
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

const wchar_t** dsl_pipeline_list_all()
{
    return DSL::Services::GetServices()->PipelineListAll();
}

DslReturnType dsl_pipeline_component_add(const wchar_t* pipeline, 
    const wchar_t* component)
{
    return DSL::Services::GetServices()->PipelineComponentAdd(pipeline, component);
}

DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    return DSL::Services::GetServices()->PipelineComponentAddMany(pipeline, components);
}

DslReturnType dsl_pipeline_component_remove(const wchar_t* pipeline, 
    const wchar_t* component)
{
    return DSL::Services::GetServices()->PipelineComponentRemove(pipeline, component);
}

DslReturnType dsl_pipeline_component_remove_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    return DSL::Services::GetServices()->PipelineComponentRemoveMany(pipeline, components);
}

DslReturnType dsl_pipeline_streammux_set_batch_properties(const wchar_t* pipeline, 
    uint batchSize, uint batchTimeout)
{
    return DSL::Services::GetServices()->PipelineStreamMuxSetBatchProperties(pipeline,
        batchSize, batchTimeout);
}

DslReturnType dsl_pipeline_streammux_set_output_size(const wchar_t* pipeline, 
    uint width, uint height)
{
    return DSL::Services::GetServices()->PipelineStreamMuxSetOutputSize(pipeline,
        width, height);
}
 
DslReturnType dsl_pipeline_pause(const wchar_t* pipeline)
{
    return DSL::Services::GetServices()->PipelinePause(pipeline);
}

DslReturnType dsl_pipeline_play(const wchar_t* pipeline)
{
    return DSL::Services::GetServices()->PipelinePlay(pipeline);
}

DslReturnType dsl_pipeline_stop(const wchar_t* pipeline)
{
    return DSL::Services::GetServices()->PipelineStop(pipeline);
}

DslReturnType dsl_pipeline_get_state(const wchar_t* pipeline)
{
    return DSL::Services::GetServices()->PipelineGetState(pipeline);
}

DslReturnType dsl_pipeline_dump_to_dot(const wchar_t* pipeline, wchar_t* filename)
{
    return DSL::Services::GetServices()->PipelineDumpToDot(pipeline, filename);
}

DslReturnType dsl_pipeline_dump_to_dot_with_ts(const wchar_t* pipeline, wchar_t* filename)
{
    return DSL::Services::GetServices()->PipelineDumpToDotWithTs(pipeline, filename);    
}

DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener, void* userdata)
{
    return DSL::Services::GetServices()->
        PipelineStateChangeListenerAdd(pipeline, listener, userdata);
}

DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener)
{
    return DSL::Services::GetServices()->
        PipelineStateChangeListenerRemove(pipeline, listener);
}

DslReturnType dsl_pipeline_display_event_handler_add(const wchar_t* pipeline, 
    dsl_display_event_handler_cb handler, void* userdata)
{
    return DSL::Services::GetServices()->
        PipelineDisplayEventHandlerAdd(pipeline, handler, userdata);
}    

DslReturnType dsl_pipeline_display_event_handler_remove(const wchar_t* pipeline, 
    dsl_display_event_handler_cb handler)
{
    return DSL::Services::GetServices()->
        PipelineDisplayEventHandlerRemove(pipeline, handler);
}

#define RETURN_IF_PIPELINE_NAME_NOT_FOUND(_pipelines_, _name_) do \
{ \
    if (!_pipelines_[_name_]) \
    { \
        LOG_ERROR("Pipeline name '" << _name_ << "' was not found"); \
        return DSL_RESULT_PIPELINE_NAME_NOT_FOUND; \
    } \
}while(0); 
    
#define RETURN_IF_COMPONENT_NAME_NOT_FOUND(_components_, _name_) do \
{ \
    if (!_components_[_name_]) \
    { \
        LOG_ERROR("Component name '" << _name_ << "' was not found"); \
        return DSL_RESULT_COMPONENT_NAME_NOT_FOUND; \
    } \
}while(0); 

/**
 * @brief Static conversion memory for Component names.
 */
static std::wstring _cp_wstr_;
static std::string _cp_cstr_;
/**
 * @brief Converts a UNICODE character string to ASCII
 * @param wstr UNICODE character string to convert
 * @return a constant ASCII string
 */
inline const char* CP_WTCSTR(const wchar_t* wstr)
{
    _cp_wstr_.assign(wstr);
    _cp_cstr_.assign(_cp_wstr_.begin(), _cp_wstr_.end());
    return _cp_cstr_.c_str();
}

/**
 * @brief Converts an ASCII UNICODE character string to ASCII
 * @param wstr UNICODE character string to convert
 * @return a constant ASCII character string to UNICODE
 */
inline const wchar_t* CP_CTWSTR(const char* cstr)
{
    _cp_cstr_.assign(cstr); \
    _cp_wstr_.assign(_cp_cstr_.begin(), _cp_cstr_.end());
    return _cp_wstr_.c_str();
}

/**
 * @brief Static conversion memory for Pipeline names.
 */
static std::wstring _pl_wstr_;
static std::string _pl_cstr_;
/**
 * @brief Converts a UNICODE character string to ASCII
 * @param wstr UNICODE character string to convert
 * @return a constant ASCII string
 */
inline const char* PL_WTCSTR(const wchar_t* wstr)
{
    _pl_wstr_.assign(wstr);
    _pl_cstr_.assign(_pl_wstr_.begin(), _pl_wstr_.end());
    return _pl_cstr_.c_str();
}

/**
 * @brief Converts an ASCII UNICODE character string to ASCII
 * @param wstr UNICODE character string to convert
 * @return a constant ASCII character string to UNICODE
 */
inline const wchar_t* PL_CTWSTR(const char* cstr)
{
    _pl_cstr_.assign(cstr); \
    _pl_wstr_.assign(_pl_cstr_.begin(), _pl_cstr_.end());
    return _pl_wstr_.c_str();
}

/**
 * @brief Static conversion memory for File names.
 */
static std::wstring _fl1_wstr_;
static std::string _fl1_cstr_;
/**
 * @brief Converts a UNICODE character string to ASCII
 * @param wstr UNICODE character string to convert
 * @return a constant ASCII string
 */
inline const char* FL1_WTCSTR(const wchar_t* wstr)
{
    _fl1_wstr_.assign(wstr);
    _fl1_cstr_.assign(_fl1_wstr_.begin(), _fl1_wstr_.end());
    return _fl1_cstr_.c_str();
}

static std::wstring _fl2_wstr_;
static std::string _fl2_cstr_;

inline const char* FL2_WTCSTR(const wchar_t* wstr)
{
    _fl2_wstr_.assign(wstr);
    _fl2_cstr_.assign(_fl2_wstr_.begin(), _fl2_wstr_.end());
    return _fl2_cstr_.c_str();
}


//TODO - revisit need to remove const.
/**
 * @brief Converts a UNICODE character string to ASCII - non const return version
 * require for GStreamer DOT file API
 * @param wstr UNICODE character string to convert
 * @return a non-constant, non-type-safe ASCII string 
 */
inline char* WTCSTR_NC(const wchar_t* wstr)
{
    _fl1_wstr_.assign(wstr); \
    _fl1_cstr_.assign(_cp_wstr_.begin(), _cp_wstr_.end());
    return const_cast<char*>(_cp_cstr_.c_str());
}

#define INIT_MEMORY(m) memset(&m, 0, sizeof(m));
#define INIT_STRUCT(type, name) struct type name; INIT_MEMORY(name) 
/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void PrgItrSigIsr(int signum)
{
    INIT_STRUCT(sigaction, sa);

    sa.sa_handler = SIG_DFL;

    sigaction(SIGINT, &sa, NULL);

    g_main_loop_quit(DSL::Services::GetServices()->GetMainLoopHandle());
}

/**
 * Function to install custom handler for program interrupt signal.
 */
static void PrgItrSigIsrInstall(void)
{
    INIT_STRUCT(sigaction, sa);

    sa.sa_handler = PrgItrSigIsr;

    sigaction(SIGINT, &sa, NULL);
}    

void dsl_main_loop_run()
{
    PrgItrSigIsrInstall();
    g_main_loop_run(DSL::Services::GetServices()->GetMainLoopHandle());
}


namespace DSL
{
    // Initialize the Services's single instance pointer
    Services* Services::m_pInstatnce = NULL;

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
        , m_numSourceInUseMax(DSL_DEFAULT_SOURCE_IN_USE_MAX)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_servicesMutex);
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
        g_mutex_clear(&m_servicesMutex);
    }
    
    DslReturnType Services::SourceCsiNew(const wchar_t* name,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[CP_WTCSTR(name)])
        {   
            LOG_ERROR("Source name '" << name << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[CP_WTCSTR(name)] = DSL_CSI_SOURCE_NEW(CP_WTCSTR(name), width, height, fps_n, fps_d);
        }
        catch(...)
        {
            LOG_ERROR("New CSI Source '" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("new CSI Source '" << CP_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceUriNew(const wchar_t* name,
        const wchar_t* uri, uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[CP_WTCSTR(name)])
        {   
            LOG_ERROR("Source name '" << CP_WTCSTR(name) << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        std::ifstream streamUriFile(FL1_WTCSTR(uri));
        if (!streamUriFile.good())
        {
            LOG_ERROR("URI Source'" << FL1_WTCSTR(uri) << "' Not found");
            return DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND;
        }        
        try
        {
            m_components[CP_WTCSTR(name)] = DSL_URI_SOURCE_NEW(
                CP_WTCSTR(name), FL1_WTCSTR(uri), cudadecMemType, intraDecode, dropFrameInterval);
        }
        catch(...)
        {
            LOG_ERROR("New URI Source '" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("new URI Source '" << CP_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    boolean Services::SourceIsLive(const wchar_t* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, CP_WTCSTR(name));
        
        try
        {
            return std::dynamic_pointer_cast<SourceBintr>(m_components[CP_WTCSTR(name)])->IsLive();
        }
        catch(...)
        {
            LOG_ERROR("Component '" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    uint Services::GetNumSourceInUse()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        uint numInUse(0);
        
        for (auto const& imap: m_pipelines)
        {
            numInUse += imap.second->GetNumSourceInUse();
        }
        return numInUse;
    }
    
    uint Services::GetNumSourceInUseMax()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_numSourceInUseMax;
    }
    
    void Services::SetNumSourceInUseMax(uint max)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        m_numSourceInUseMax = max;
    }

    DslReturnType Services::OverlaySinkNew(const wchar_t* name, 
        uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[CP_WTCSTR(name)])
        {   
            LOG_ERROR("Sink name '" << CP_WTCSTR(name) << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[CP_WTCSTR(name)] = DSL_OVERLAY_SINK_NEW(CP_WTCSTR(name), offsetX, offsetY, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_SINK_NEW_EXCEPTION;
        }
        LOG_INFO("new Sink '" << CP_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdNew(const wchar_t* name, boolean isClockEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[CP_WTCSTR(name)])
        {   
            LOG_ERROR("OSD name '" << CP_WTCSTR(name) << "' is not unique");
            return DSL_RESULT_OSD_NAME_NOT_UNIQUE;
        }
        try
        {   
            m_components[CP_WTCSTR(name)] = std::shared_ptr<Bintr>(new OsdBintr(
                CP_WTCSTR(name), isClockEnabled));
        }
        catch(...)
        {
            LOG_ERROR("New OSD '" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_SINK_NEW_EXCEPTION;
        }
        LOG_INFO("new OSD '" << CP_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::DisplayNew(const wchar_t* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[CP_WTCSTR(name)])
        {   
            LOG_ERROR("Display name '" << CP_WTCSTR(name) << "' is not unique");
            return DSL_RESULT_DISPLAY_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[CP_WTCSTR(name)] = std::shared_ptr<Bintr>(new DisplayBintr(
                CP_WTCSTR(name), width, height));
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display New'" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_DISPLAY_NEW_EXCEPTION;
        }
        LOG_INFO("new Display '" << CP_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PrimaryGieNew(const wchar_t* name, const wchar_t* inferConfigFile,
        const wchar_t* modelEngineFile, uint interval, uint uniqueId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[CP_WTCSTR(name)])
        {   
            LOG_ERROR("GIE name '" << CP_WTCSTR(name) << "' is not unique");
            return DSL_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        
        LOG_INFO("Infer config file: " << FL1_WTCSTR(inferConfigFile));
        
        std::ifstream configFile(FL1_WTCSTR(inferConfigFile));
        if (!configFile.good())
        {
            LOG_ERROR("Infer Config File not found");
            return DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND;
        }
        
        LOG_INFO("Model engine file: " << FL2_WTCSTR(modelEngineFile));
        
        std::ifstream modelFile(FL2_WTCSTR(modelEngineFile));
        if (!modelFile.good())
        {
            LOG_ERROR("Model Engine File not found");
            return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
        }
        try
        {
            m_components[CP_WTCSTR(name)] = DSL_PRIMARY_GIE_NEW(CP_WTCSTR(name), 
                FL1_WTCSTR(inferConfigFile), FL2_WTCSTR(modelEngineFile), interval, uniqueId);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << CP_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_GIE_NEW_EXCEPTION;
        }
        LOG_INFO("new GIE '" << CP_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentDelete(const wchar_t* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, CP_WTCSTR(component));
        
        if (m_components[CP_WTCSTR(component)]->IsInUse())
        {
            LOG_INFO("Component '" << CP_WTCSTR(component) << "' is in use");
            return DSL_RESULT_COMPONENT_IN_USE;
        }
        m_components.erase(CP_WTCSTR(component));

        LOG_INFO("Component '" << CP_WTCSTR(component) << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentDeleteMany(const wchar_t** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // iterate through the list of provided components to verifiy the
        // existence of each... AND that each is not owned by a pipeline 
        // before making any updates to the list of components.
        for (const wchar_t** component = components; *component; component++)
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, CP_WTCSTR(*component));

            if (m_components[CP_WTCSTR(*component)]->IsInUse())
            {   
                LOG_ERROR("Component '" << CP_WTCSTR(*component) << "' is currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
        }
        LOG_DEBUG("All listed components found and un-owned");
        
        // iterate through the list a second time erasing each
        for (const wchar_t** component = components; *component; component++)
        {
            m_components.erase(CP_WTCSTR(*component));
        }

        LOG_INFO("All Components deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::ComponentDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        for (auto const& imap: m_components)
        {
            if (imap.second->IsInUse())
            {
                LOG_ERROR("Component '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
        }
        LOG_DEBUG("All components are un-owned and will be deleted");

        for (auto const& imap: m_components)
        {
            LOG_WARN("******* " << imap.second->GetName());
            m_components.erase(imap.second->GetName());
        }
        LOG_INFO("All Components deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::ComponentListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_components.size();
    }
    
    const wchar_t** Services::ComponentListAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        m_componentNames.clear();
        
        // reserve to avoid resizing - plus 1 for the NULL terminator
        m_componentNames.reserve(m_components.size() + 1);
        for(auto const& imap: m_components)
        {
            m_componentNames.push_back(CP_CTWSTR(imap.first.c_str()));
        }
        m_componentNames.push_back(NULL);
        
        return (const wchar_t**)&m_componentNames[0];
    }
    
    DslReturnType Services::PipelineNew(const wchar_t* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_pipelines[PL_WTCSTR(name)])
        {   
            LOG_ERROR("Pipeline name '" << PL_WTCSTR(name) << "' is not unique");
            return DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_pipelines[PL_WTCSTR(name)] = std::shared_ptr<PipelineBintr>(new PipelineBintr(CP_WTCSTR(name)));
        }
        catch(...)
        {
            LOG_ERROR("New Pipeline '" << PL_WTCSTR(name) << "' threw exception on create");
            return DSL_RESULT_PIPELINE_NEW_EXCEPTION;
        }
        LOG_INFO("new PIPELINE '" << PL_WTCSTR(name) << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineNewMany(const wchar_t** pipelines)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (const wchar_t** pipeline = pipelines; *pipeline; pipeline++)
        {
            if (m_pipelines[PL_WTCSTR(*pipeline)])
            {   
                LOG_ERROR("Pipeline name '" << *pipeline << "' is not unique");
                return DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE;
            }
            try
            {
                m_pipelines[PL_WTCSTR(*pipeline)] = std::shared_ptr<PipelineBintr>(new PipelineBintr(CP_WTCSTR(*pipeline)));
            }
            catch(...)
            {
                LOG_ERROR("New Pipeline '" << PL_WTCSTR(*pipeline) << "' threw exception on create");
                return DSL_RESULT_PIPELINE_NEW_EXCEPTION;
            }
            LOG_INFO("new PIPELINE '" << PL_WTCSTR(*pipeline) << "' created successfully");
        }
    }
    
    DslReturnType Services::PipelineDelete(const wchar_t* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        m_pipelines[PL_WTCSTR(pipeline)]->RemoveAllChildren();
        m_pipelines.erase(PL_WTCSTR(pipeline));

        LOG_INFO("Pipeline '" << PL_WTCSTR(pipeline) << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineDeleteMany(const wchar_t** pipelines)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // iterate through the list of provided pipelines to verifiy the
        // existence of each before making any updates to the list of pipelines.
        for (const wchar_t** pipeline = pipelines; *pipeline; pipeline++)
        {
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(*pipeline));
        }
        LOG_DEBUG("All listed pipelines found");
        
        // iterate through the list a second time erasing each
        for (const wchar_t** pipeline = pipelines; *pipeline; pipeline++)
        {
            m_pipelines[PL_WTCSTR(*pipeline)]->RemoveAllChildren();
            m_pipelines.erase(PL_WTCSTR(*pipeline));
        }

        LOG_INFO("All Pipelines deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (auto &imap: m_pipelines)
        {
            imap.second->RemoveAllChildren();
            imap.second = nullptr;
        }
        m_pipelines.clear();

        return DSL_RESULT_SUCCESS;
    }

    uint Services::PipelineListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_pipelines.size();
    }
    
    const wchar_t** Services::PipelineListAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        m_pipelineNames.clear();
        
        // reserve to avoid resizing - plus 1 for the NULL terminator
        m_pipelineNames.reserve(m_pipelines.size() + 1);
        for(auto const& imap: m_pipelines)
        {
            m_pipelineNames.push_back(PL_CTWSTR(imap.first.c_str()));
        }
        m_pipelineNames.push_back(NULL);
        
        return (const wchar_t**)&m_pipelineNames[0];
    }
    
    DslReturnType Services::PipelineComponentAdd(const wchar_t* pipeline, 
        const wchar_t* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, CP_WTCSTR(component));

        try
        {
            m_components[CP_WTCSTR(component)]->AddToParent(m_pipelines[PL_WTCSTR(pipeline)]);
            LOG_INFO("Component '" << CP_WTCSTR(component) 
                << "' was added to Pipeline '" << PL_WTCSTR(pipeline) << "' successfully");
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << PL_WTCSTR(pipeline)
                << "' threw exception adding component '" << CP_WTCSTR(component) << "'");
            return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
        }
    }    
        
    
    DslReturnType Services::PipelineComponentAddMany(const wchar_t* pipeline, 
        const wchar_t** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));
        
        // iterate through the list of provided components to verifiy the
        //  existence of each - before making any updates to the pipeline.
        for (const wchar_t** component = components; *component; component++)
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, CP_WTCSTR(*component));
        }
        LOG_INFO("All listed components found");
        
        // iterate through the list of provided components a second time
        // adding each to the named pipeline individually.
        for (const wchar_t** component = components; *component; component++)
        {
            try
            {
                m_components[CP_WTCSTR(*component)]->AddToParent(m_pipelines[PL_WTCSTR(pipeline)]);
                LOG_INFO("Component '" << CP_WTCSTR(*component) 
                    << "' was added to Pipeline '" << PL_WTCSTR(pipeline) << "' successfully");
            }
            catch(...)
            {
                LOG_ERROR("Pipeline '" << PL_WTCSTR(pipeline) 
                    << "' threw exception adding component '" << CP_WTCSTR(*component) << "'");
                return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
            }
        }
        
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineComponentRemove(const wchar_t* pipeline, 
        const wchar_t* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, CP_WTCSTR(component));

        if (!m_components[CP_WTCSTR(component)]->IsParent(m_pipelines[PL_WTCSTR(pipeline)]))
        {
            LOG_ERROR("Component '" << CP_WTCSTR(component) << 
                "' is not in use by Pipeline '" << PL_WTCSTR(pipeline) << "'");
            return DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE;
        }
        try
        {
            m_components[CP_WTCSTR(component)]->RemoveFromParent(m_pipelines[PL_WTCSTR(pipeline)]);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << PL_WTCSTR(pipeline) 
                << "' threw an exception removing component");
            return DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED;
        }
        return DSL_RESULT_SUCCESS;
}
    
    DslReturnType Services::PipelineComponentRemoveMany(const wchar_t* pipeline, 
        const wchar_t** components)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
    
    DslReturnType Services::PipelineStreamMuxSetBatchProperties(const wchar_t* pipeline,
        uint batchSize, uint batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        try
        {
            m_pipelines[PL_WTCSTR(pipeline)]->SetStreamMuxBatchProperties(batchSize, batchTimeout);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << PL_WTCSTR(pipeline) 
                << "' threw an exception setting the Stream Muxer batch_properties");
            return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStreamMuxSetOutputSize(const wchar_t* pipeline,
        uint width, uint height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        try
        {
            m_pipelines[PL_WTCSTR(pipeline)]->SetStreamMuxOutputSize(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << PL_WTCSTR(pipeline) 
                << "' threw an exception setting the Stream Muxer output size");
            return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelinePause(const wchar_t* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[PL_WTCSTR(pipeline)])->Pause())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_PAUSE;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelinePlay(const wchar_t* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[PL_WTCSTR(pipeline)])->Play())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_PLAY;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineStop(const wchar_t* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[PL_WTCSTR(pipeline)])->Stop())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_STOP;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineGetState(const wchar_t* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        return DSL_RESULT_API_NOT_IMPLEMENTED;
    }
        
    DslReturnType Services::PipelineDumpToDot(const wchar_t* pipeline, wchar_t* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[PL_WTCSTR(pipeline)]->DumpToDot(WTCSTR_NC(filename));
        
        return DSL_RESULT_SUCCESS;
    }   
    
    DslReturnType Services::PipelineDumpToDotWithTs(const wchar_t* pipeline, wchar_t* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[PL_WTCSTR(pipeline)]->DumpToDot(WTCSTR_NC(filename));

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStateChangeListenerAdd(const wchar_t* pipeline, 
        dsl_state_change_listener_cb listener, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));

        if (m_pipelines[PL_WTCSTR(pipeline)]->IsChildStateChangeListener(listener))
        {
            return DSL_RESULT_PIPELINE_LISTENER_NOT_UNIQUE;
        }
        return m_pipelines[PL_WTCSTR(pipeline)]->AddStateChangeListener(listener, userdata);
    }
        
    DslReturnType Services::PipelineStateChangeListenerRemove(const wchar_t* pipeline, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));
    
        if (!m_pipelines[PL_WTCSTR(pipeline)]->IsChildStateChangeListener(listener))
        {
            return DSL_RESULT_PIPELINE_LISTENER_NOT_FOUND;
        }
        return m_pipelines[PL_WTCSTR(pipeline)]->RemoveStateChangeListener(listener);
    }
    
    DslReturnType Services::PipelineDisplayEventHandlerAdd(const wchar_t* pipeline, 
        dsl_display_event_handler_cb handler, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));
        
        if (m_pipelines[PL_WTCSTR(pipeline)]->IsChildDisplayEventHandler(handler))
        {
            return DSL_RESULT_PIPELINE_HANDLER_NOT_UNIQUE;
        }
        return m_pipelines[PL_WTCSTR(pipeline)]->AddDisplayEventHandler(handler, userdata);
    }
        

    DslReturnType Services::PipelineDisplayEventHandlerRemove(const wchar_t* pipeline, 
        dsl_display_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, PL_WTCSTR(pipeline));
        
        if (!m_pipelines[PL_WTCSTR(pipeline)]->IsChildDisplayEventHandler(handler))
        {
            return DSL_RESULT_PIPELINE_HANDLER_NOT_FOUND;
        }
        return m_pipelines[PL_WTCSTR(pipeline)]->RemoveDisplayEventHandler(handler);
    }
    

} // namespace 