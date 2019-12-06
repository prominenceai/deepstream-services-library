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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceCsiNew(cstrName.c_str(), 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_uri_new(const wchar_t* name, 
    const wchar_t* uri, uint cudadec_mem_type, uint intra_decode, uint dropFrameInterval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceUriNew(cstrName.c_str(),
        cstrUri.c_str(), cudadec_mem_type, intra_decode, dropFrameInterval);
}

DslReturnType dsl_source_pause(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourcePause(cstrName.c_str());
}

DslReturnType dsl_source_resume(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceResume(cstrName.c_str());
}

boolean dsl_source_is_live(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceIsLive(cstrName.c_str());
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

DslReturnType dsl_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());
    
    return DSL::Services::GetServices()->PrimaryGieNew(cstrName.c_str(), cstrConfig.c_str(),
        cstrEngine.c_str(), interval);
}

DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie_name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());
    std::wstring wstrGie(infer_on_gie_name);
    std::string cstrGie(wstrGie.begin(), wstrGie.end());
    
    return DSL::Services::GetServices()->SecondaryGieNew(cstrName.c_str(), cstrConfig.c_str(),
        cstrEngine.c_str(), cstrGie.c_str());
}
    
DslReturnType dsl_osd_new(const wchar_t* name, boolean isClockEnabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdNew(cstrName.c_str(), isClockEnabled);
}

DslReturnType dsl_display_new(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayNew(cstrName.c_str(), width, height);
}

DslReturnType dsl_display_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_display_dimensions_set(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_display_tiles_get(const wchar_t* name, uint* cols, uint* rows)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTilesGet(cstrName.c_str(), cols, rows);
}

DslReturnType dsl_display_tiles_set(const wchar_t* name, uint cols, uint rows)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTilesSet(cstrName.c_str(), cols, rows);
}

DslReturnType dsl_sink_overlay_new(const wchar_t* name,
    uint offsetX, uint offsetY, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OverlaySinkNew(cstrName.c_str(), 
        offsetX, offsetY, width, height);
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
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineNew(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines)
{
    return DSL::Services::GetServices()->PipelineNewMany(pipelines);
}

DslReturnType dsl_pipeline_delete(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineDelete(cstrPipeline.c_str());
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
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentAdd(cstrPipeline.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    return DSL::Services::GetServices()->PipelineComponentAddMany(pipeline, components);
}

DslReturnType dsl_pipeline_component_remove(const wchar_t* pipeline, 
    const wchar_t* component)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentRemove(cstrPipeline.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_remove_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    return DSL::Services::GetServices()->PipelineComponentRemoveMany(pipeline, components);
}

DslReturnType dsl_pipeline_streammux_batch_properties_get(const wchar_t* pipeline, 
    uint* batchSize, uint* batchTimeout)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxBatchPropertiesGet(cstrPipeline.c_str(),
        batchSize, batchTimeout);
}

DslReturnType dsl_pipeline_streammux_batch_properties_set(const wchar_t* pipeline, 
    uint batchSize, uint batchTimeout)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxBatchPropertiesSet(cstrPipeline.c_str(),
        batchSize, batchTimeout);
}

DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxDimensionsGet(cstrPipeline.c_str(),
        width, height);
}

DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxDimensionsSet(cstrPipeline.c_str(),
        width, height);
}    

DslReturnType dsl_pipeline_streammux_padding_get(const wchar_t* pipeline, boolean* enabled)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxPaddingGet(cstrPipeline.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_padding_set(const wchar_t* pipeline, boolean enabled)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxPaddingSet(cstrPipeline.c_str(), enabled);
}
 
DslReturnType dsl_pipeline_pause(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelinePause(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_play(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelinePlay(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_stop(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStop(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_get_state(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineGetState(cstrPipeline.c_str());
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

    std::vector<const wchar_t*> Services::m_pipelineNames;
    std::vector<const wchar_t*> Services::m_componentNames;

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
    
    DslReturnType Services::SourceCsiNew(const char* name,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("Source name '" << name << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_CSI_SOURCE_NEW(name, width, height, fps_n, fps_d);
        }
        catch(...)
        {
            LOG_ERROR("New CSI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("new CSI Source '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceUriNew(const char* name,
        const char* uri, uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("Source name '" << name << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("URI Source'" << uri << "' Not found");
            return DSL_RESULT_SOURCE_STREAM_FILE_NOT_FOUND;
        }        
        try
        {
            m_components[name] = DSL_URI_SOURCE_NEW(
                name, uri, cudadecMemType, intraDecode, dropFrameInterval);
        }
        catch(...)
        {
            LOG_ERROR("New URI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("new URI Source '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SourcePause(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            DSL_SOURCE_PTR sourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
                
            if (!sourceBintr->IsInUse())
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in use");
                return DSL_RESULT_SOURCE_NOT_IN_USE;
            }
            if (sourceBintr->GetState() != GST_STATE_PLAYING)
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in play");
                return DSL_RESULT_SOURCE_NOT_IN_PLAY;
            }
            if (!sourceBintr->Pause())
            {
                LOG_ERROR("Source '" << name << "' failed to change state to paused");
                return DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on pause");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceResume(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            DSL_SOURCE_PTR sourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
                
            if (!sourceBintr->IsInUse())
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not in use");
                return DSL_RESULT_SOURCE_NOT_IN_USE;
            }
            if (sourceBintr->GetState() != GST_STATE_PAUSED)
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not in pause");
                return DSL_RESULT_SOURCE_NOT_IN_PAUSE;
            }

            if (!sourceBintr->Play())
            {
                LOG_ERROR("Source '" << name << "' failed to change state to play");
                return DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on pause");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
        
    boolean Services::SourceIsLive(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            return std::dynamic_pointer_cast<SourceBintr>(m_components[name])->IsLive();
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on create");
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

    DslReturnType Services::PrimaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("GIE name '" << name << "' is not unique");
            return DSL_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        
        LOG_INFO("Infer config file: " << inferConfigFile);
        
        std::ifstream configFile(inferConfigFile);
        if (!configFile.good())
        {
            LOG_ERROR("Infer Config File not found");
            return DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND;
        }
        
        LOG_INFO("Model engine file: " << modelEngineFile);
        
        std::ifstream modelFile(modelEngineFile);
        if (!modelFile.good())
        {
            LOG_ERROR("Model Engine File not found");
            return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
        }
        try
        {
            m_components[name] = DSL_PRIMARY_GIE_NEW(name, 
                inferConfigFile, modelEngineFile, interval);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        LOG_INFO("new Primary GIE '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SecondaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, const char* inferOnGieName)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("GIE name '" << name << "' is not unique");
            return DSL_RESULT_GIE_NAME_NOT_UNIQUE;
        }
        
        LOG_INFO("Infer config file: " << inferConfigFile);
        
        std::ifstream configFile(inferConfigFile);
        if (!configFile.good())
        {
            LOG_ERROR("Infer Config File not found");
            return DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND;
        }
        
        LOG_INFO("Model engine file: " << modelEngineFile);
        
        std::ifstream modelFile(modelEngineFile);
        if (!modelFile.good())
        {
            LOG_ERROR("Model Engine File not found");
            return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
        }
        try
        {
            m_components[name] = DSL_SECONDARY_GIE_NEW(name, 
                inferConfigFile, modelEngineFile, inferOnGieName);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        LOG_INFO("new Secondary GIE '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
   
    DslReturnType Services::DisplayNew(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("Display name '" << name << "' is not unique");
            return DSL_RESULT_DISPLAY_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new DisplayBintr(
                name, width, height));
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display New'" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_THREW_EXCEPTION;
        }
        LOG_INFO("new Display '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::DisplayDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);

        try
        {
            DSL_DISPLAY_PTR displayBintr = 
                std::dynamic_pointer_cast<DisplayBintr>(m_components[name]);

            // TODO verify args before calling
            displayBintr->GetDimensions(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_DISPLAY_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::DisplayDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        if (m_components[name]->IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for Tiled Display '" << name 
                << "' as it's currently in use");
            return DSL_RESULT_DISPLAY_IS_IN_USE;
        }
        try
        {
            DSL_DISPLAY_PTR displayBintr = 
                std::dynamic_pointer_cast<DisplayBintr>(m_components[name]);

            // TODO verify args before calling
            if (!displayBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Tiled Display '" << name << "' failed to settin dimensions");
                return DSL_RESULT_DISPLAY_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_DISPLAY_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::DisplayTilesGet(const char* name, uint* cols, uint* rows)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);

        try
        {
            DSL_DISPLAY_PTR displayBintr = 
                std::dynamic_pointer_cast<DisplayBintr>(m_components[name]);

            // TODO verify args before calling
            displayBintr->GetTiles(cols, rows);
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display '" << name << "' threw an exception getting Tiles");
            return DSL_RESULT_DISPLAY_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::DisplayTilesSet(const char* name, uint cols, uint rows)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);

        if (m_components[name]->IsInUse())
        {
            LOG_ERROR("Unable to set Tiles for Tiled Display '" << name 
                << "' as it's currently in use");
            return DSL_RESULT_DISPLAY_IS_IN_USE;
        }
        try
        {
            DSL_DISPLAY_PTR displayBintr = 
                std::dynamic_pointer_cast<DisplayBintr>(m_components[name]);

            // TODO verify args before calling
            if (!displayBintr->SetTiles(cols, rows))
            {
                LOG_ERROR("Tiled Display '" << name << "' failed to set Tiles");
                return DSL_RESULT_DISPLAY_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tiled Display '" << name << "' threw an exception setting Tiles");
            return DSL_RESULT_DISPLAY_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OverlaySinkNew(const char* name, 
        uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_OVERLAY_SINK_NEW(name, offsetX, offsetY, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("new Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdNew(const char* name, boolean isClockEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components[name])
        {   
            LOG_ERROR("OSD name '" << name << "' is not unique");
            return DSL_RESULT_OSD_NAME_NOT_UNIQUE;
        }
        try
        {   
            m_components[name] = std::shared_ptr<Bintr>(new OsdBintr(
                name, isClockEnabled));
        }
        catch(...)
        {
            LOG_ERROR("New OSD '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("new OSD '" << name << "' created successfully");

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
    
    DslReturnType Services::PipelineNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_pipelines[name])
        {   
            LOG_ERROR("Pipeline name '" << name << "' is not unique");
            return DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_pipelines[name] = std::shared_ptr<PipelineBintr>(new PipelineBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New Pipeline '" << name << "' threw exception on create");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        LOG_INFO("new PIPELINE '" << name << "' created successfully");

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
                return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
            }
            LOG_INFO("new PIPELINE '" << PL_WTCSTR(*pipeline) << "' created successfully");
        }
    }
    
    DslReturnType Services::PipelineDelete(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        m_pipelines[pipeline]->RemoveAllChildren();
        m_pipelines.erase(pipeline);

        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

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

    DslReturnType Services::PipelineComponentRemove(const char* pipeline, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

        if (!m_components[component]->IsParent(m_pipelines[pipeline]))
        {
            LOG_ERROR("Component '" << component << 
                "' is not in use by Pipeline '" << pipeline << "'");
            return DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE;
        }
        try
        {
            m_components[component]->RemoveFromParent(m_pipelines[pipeline]);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
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
    
    DslReturnType Services::PipelineStreamMuxBatchPropertiesGet(const char* pipeline,
        uint* batchSize, uint* batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->GetStreamMuxBatchProperties(batchSize, batchTimeout))
            {
                LOG_ERROR("Pipeline '" << pipeline
                    << "' failed to get the Stream Muxer Batch Properties");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting the Stream Muxer Batch Properties");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStreamMuxBatchPropertiesSet(const char* pipeline,
        uint batchSize, uint batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->SetStreamMuxBatchProperties(batchSize, batchTimeout))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Set the Stream Muxer Batch Properties");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer Batch Properties");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStreamMuxDimensionsGet(const char* pipeline,
        uint* width, uint* height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->GetStreamMuxDimensions(width, height))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer Output Dimensions");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer Output Dimensions");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineStreamMuxDimensionsSet(const char* pipeline,
        uint width, uint height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->SetStreamMuxDimensions(width, height))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Set the Stream Muxer Output Dimensions");
                return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer output size");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineStreamMuxPaddingGet(const char* pipeline,
        boolean* enabled)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->GetStreamMuxPadding((bool*)enabled))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer is Padding enabled setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline
                << "' threw an exception getting the Stream Muxer padding");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineStreamMuxPaddingSet(const char* pipeline,
        boolean enabled)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->SetStreamMuxPadding((bool)enabled))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer is Padding enabled setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer padding");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelinePause(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->Pause())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_PAUSE;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelinePlay(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->Play())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_PLAY;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineStop(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->Stop())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_STOP;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineGetState(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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