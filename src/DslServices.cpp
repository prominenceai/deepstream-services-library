/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslOdeTrigger.h"
#include "DslServices.h"
#include "DslServicesValidate.h"
#include "DslSegVisualBintr.h"
#include "DslPadProbeHandler.h"
#include "DslTilerBintr.h"
#include "DslOsdBintr.h"
#include "DslSinkBintr.h"

// TODO move these defines to DSL utility file
#define INIT_MEMORY(m) memset(&m, 0, sizeof(m));
#define INIT_STRUCT(type, name) struct type name; INIT_MEMORY(name) 

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void PrgItrSigIsr(int signum)
{
    dsl_main_loop_quit();
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

/**
 * Function to uninstall custom handler for program interrupt signal.
 */
static void PrgItrSigIsrUninstall(void)
{
    INIT_STRUCT(sigaction, sa);

    sa.sa_handler = SIG_DFL;
    sigaction(SIGINT, &sa, NULL);
}    

void dsl_main_loop_run()
{
    PrgItrSigIsrInstall();
    g_main_loop_run(DSL::Services::GetServices()->GetMainLoopHandle());
}

void dsl_main_loop_quit()
{
    PrgItrSigIsrUninstall();
    g_main_loop_quit(DSL::Services::GetServices()->GetMainLoopHandle());
}

const wchar_t* dsl_return_value_to_string(uint result)
{
    return DSL::Services::GetServices()->ReturnValueToString(result);
}

const wchar_t* dsl_state_value_to_string(uint state)
{
    return DSL::Services::GetServices()->StateValueToString(state);
}

const wchar_t* dsl_version_get()
{
    return DSL_VERSION;
}

void geosNoticeHandler(const char *fmt, ...)
{
    
}

void geosErrorHandler(const char *fmt, ...)
{
    
}

// Single GST debug catagory initialization
GST_DEBUG_CATEGORY(GST_CAT_DSL);

GQuark _dsmeta_quark;

namespace DSL
{
    // Initialize the Services's single instance pointer
    Services* Services::m_pInstatnce = NULL;

    Services* Services::GetServices()
    {
        // one time initialization of the single instance pointer
        if (!m_pInstatnce)
        {
            boolean doGstDeinit(false);

            // Initialize the single debug category used by the lib
            GST_DEBUG_CATEGORY_INIT(GST_CAT_DSL, "DSL", 0, "DeepStream Services");
        
            // If gst has not been initialized by the client software
            if (!gst_is_initialized())
            {
                int argc = 0;
                char** argv = NULL;
                
                // initialize the GStreamer library
                gst_init(&argc, &argv);
                doGstDeinit = true;
                
                // One-time init of Curl with no addition features
                CURLcode result = curl_global_init(CURL_GLOBAL_NOTHING);
                if (result != CURLE_OK)
                {
                    LOG_ERROR("curl_global_init failed: " << curl_easy_strerror(result));
                    throw;
                }
                curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
                
                LOG_INFO("Libcurl Initialized Successfully");
                LOG_INFO("Version: " << info->version);
                LOG_INFO("Host: " << info->host);
                LOG_INFO("Features: " << info->features);
                LOG_INFO("SSL Version: " << info->ssl_version);
                LOG_INFO("Libz Version: " << info->libz_version);
                LOG_INFO("Protocols: " << info->protocols);
            }
            
            // Safe to start logging
            LOG_INFO("Services Initialization");
            
            _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
            
            // Single instantiation for the lib's lifetime
            m_pInstatnce = new Services(doGstDeinit);
            
            // initialization of GEOS
            initGEOS(geosNoticeHandler, geosErrorHandler);
            
            // Initialize private containers
            m_pInstatnce->InitToStringMaps();
        }
        return m_pInstatnce;
    }
        
    Services::Services(bool doGstDeinit)
        : m_doGstDeinit(doGstDeinit)
        , m_pMainLoop(g_main_loop_new(NULL, FALSE))
        , m_sourceNumInUseMax(DSL_DEFAULT_SOURCE_IN_USE_MAX)
        , m_sinkNumInUseMax(DSL_DEFAULT_SINK_IN_USE_MAX)
    {
        LOG_FUNC();

        g_mutex_init(&m_servicesMutex);
    }

    Services::~Services()
    {
        LOG_FUNC();
        
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

            // Cleanup GEOS
            finishGEOS();
            
            // Cleanup Lib cURL
            curl_global_cleanup();
            
            // If this Services object called gst_init(), and not the client.
            if (m_doGstDeinit)
            {
                gst_deinit();
            }
            
            if (m_pMainLoop)
            {
                g_main_loop_unref(m_pMainLoop);
            }
        }
        g_mutex_clear(&m_servicesMutex);
    }
    
    DslReturnType Services::OfvNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("OFV name '" << name << "' is not unique");
                return DSL_RESULT_OFV_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new OfvBintr(name));

            LOG_INFO("New OFV '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New OFV '" << name << "' threw exception on create");
            return DSL_RESULT_OFV_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentDelete(const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
        
        if (m_components[component]->IsInUse())
        {
            LOG_INFO("Component '" << component << "' is in use");
            return DSL_RESULT_COMPONENT_IN_USE;
        }
        m_components.erase(component);

        LOG_INFO("Component '" << component << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_components.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            // Only if there are Pipelines do we check if the component is in use.
            if (m_pipelines.size())
            {
                for (auto const& imap: m_components)
                {
                    // In the case of Delete all
                    if (imap.second->IsInUse())
                    {
                        LOG_ERROR("Component '" << imap.second->GetName() << "' is currently in use");
                        return DSL_RESULT_COMPONENT_IN_USE;
                    }
                }
            }

            m_components.clear();
            LOG_INFO("All Components deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw exception on Delete All Components");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }

    uint Services::ComponentListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_components.size();
    }

    DslReturnType Services::ComponentGpuIdGet(const char* component, uint* gpuid)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
            
            if (m_components[component]->IsInUse())
            {
                LOG_INFO("Component '" << component << "' is in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
            *gpuid = m_components[component]->GetGpuId();

            LOG_INFO("Current GPU ID = " << *gpuid << " for component '" << component << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Component '" << component << "' threw exception on Get GPU ID");
            return DSL_RESULT_COMPONENT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::ComponentGpuIdSet(const char* component, uint gpuid)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
        
        if (m_components[component]->IsInUse())
        {
            LOG_INFO("Component '" << component << "' is in use");
            return DSL_RESULT_COMPONENT_IN_USE;
        }
        if (!m_components[component]->SetGpuId(gpuid))
        {
            LOG_INFO("Component '" << component << "' faild to set GPU ID = " << gpuid);
            return DSL_RESULT_COMPONENT_SET_GPUID_FAILED;
        }

        LOG_INFO("New GPU ID = " << gpuid << " for component '" << component << "'");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::BranchNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_components[name])
        {   
            LOG_ERROR("Branch name '" << name << "' is not unique");
            return DSL_RESULT_BRANCH_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new BranchBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New Branch '" << name << "' threw exception on create");
            return DSL_RESULT_BRANCH_THREW_EXCEPTION;
        }
        LOG_INFO("New BRANCH '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::BranchComponentAdd(const char* branch, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, branch);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

        try
        {
            // Can't add components if they're In use by another Pipeline
            if (m_components[component]->IsInUse())
            {
                LOG_ERROR("Unable to add component '" << component 
                    << "' as it's currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }

            // Check for MAX Sources in Use - Do not exceed!
            if (IsSourceComponent(component) )
            {
                LOG_ERROR("Can't add source '" << component << "' to branch '" << branch << 
                    "' sources can only be added to Pipelines");
                return DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED;
            }

            if (IsSinkComponent(component) and (GetNumSinksInUse() == m_sinkNumInUseMax))
            {
                LOG_ERROR("Adding Sink '" << component << "' to Branch '" << branch << 
                    "' would exceed the maximum num-in-use limit");
                return DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACHED;
            }
            if (!m_components[component]->AddToParent(m_components[branch]))
            {
                LOG_ERROR("Branch '" << branch
                    << "' failed to add component '" << component << "'");
                return DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Branch '" << branch
                << "' threw exception adding component '" << component << "'");
            return DSL_RESULT_BRANCH_THREW_EXCEPTION;
        }
        LOG_INFO("Component '" << component 
            << "' was added to Branch '" << branch << "' successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::BranchComponentRemove(const char* branch, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, branch);
        DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

        if (!m_components[component]->IsParent(m_components[branch]))
        {
            LOG_ERROR("Component '" << component << 
                "' is not in use by Branch '" << branch << "'");
            return DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH;
        }
        try
        {
            if (!m_components[component]->RemoveFromParent(m_components[branch]))
            {
                LOG_ERROR("Branch '" << branch
                    << "' failed to remove component '" << component << "'");
                return DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Branch '" << branch 
                << "' threw an exception removing component");
            return DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED;
        }
        LOG_INFO("Component '" << component 
            << "' was removed from Branch '" << branch << "' successfully");

        return DSL_RESULT_SUCCESS;
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
        LOG_INFO("New PIPELINE '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineDelete(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            m_pipelines[pipeline]->RemoveAllChildren();
            m_pipelines.erase(pipeline);

            LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline << "' threw an exception on Delete");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
            
    }

    DslReturnType Services::PipelineDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            for (auto &imap: m_pipelines)
            {
                imap.second->RemoveAllChildren();
                imap.second = nullptr;
            }
            m_pipelines.clear();

            LOG_INFO("All Pipelines deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on PipelineDeleteAll");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    uint Services::PipelineListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_pipelines.size();
    }
    
    DslReturnType Services::PipelineComponentAdd(const char* pipeline, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
            
            // Can't add components if they're In use by another Pipeline
            if (m_components[component]->IsInUse())
            {
                LOG_ERROR("Unable to add component '" << component 
                    << "' as it's currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }

            // Check for MAX Sources in Use - Do not exceed!
            if (IsSourceComponent(component) and (GetNumSourcesInUse() == m_sourceNumInUseMax))
            {
                LOG_ERROR("Adding Source '" << component << "' to Pipeline '" << pipeline << 
                    "' would exceed the maximum num-in-use limit");
                return DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED;
            }

            if (IsSinkComponent(component) and (GetNumSinksInUse() == m_sinkNumInUseMax))
            {
                LOG_ERROR("Adding Sink '" << component << "' to Pipeline '" << pipeline << 
                    "' would exceed the maximum num-in-use limit");
                return DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACHED;
            }

            if (!m_components[component]->AddToParent(m_pipelines[pipeline]))
            {
                LOG_ERROR("Pipeline '" << pipeline
                    << "' failed component '" << component << "'");
                return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
            }
            LOG_INFO("Component '" << component 
                << "' was added to Pipeline '" << pipeline << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline
                << "' threw exception adding component '" << component << "'");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }    
    
    DslReturnType Services::PipelineComponentRemove(const char* pipeline, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

            if (!m_components[component]->IsParent(m_pipelines[pipeline]))
            {
                LOG_ERROR("Component '" << component << 
                    "' is not in use by Pipeline '" << pipeline << "'");
                return DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE;
            }
            m_components[component]->RemoveFromParent(m_pipelines[pipeline]);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing component");
            return DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED;
        }
}
    
    DslReturnType Services::PipelineStreamMuxBatchPropertiesGet(const char* pipeline,
        uint* batchSize, uint* batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            m_pipelines[pipeline]->GetStreamMuxBatchProperties(batchSize, batchTimeout);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting the Stream Muxer Batch Properties");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineStreamMuxBatchPropertiesSet(const char* pipeline,
        uint batchSize, uint batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->SetStreamMuxBatchProperties(batchSize, batchTimeout))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Set the Stream Muxer Batch Properties");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer Batch Properties");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineStreamMuxDimensionsGet(const char* pipeline,
        uint* width, uint* height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->GetStreamMuxDimensions(width, height))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer Output Dimensions");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer Output Dimensions");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreamMuxDimensionsSet(const char* pipeline,
        uint width, uint height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->SetStreamMuxDimensions(width, height))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Set the Stream Muxer Output Dimensions");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer output size");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreamMuxPaddingGet(const char* pipeline,
        boolean* enabled)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->GetStreamMuxPadding((bool*)enabled))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer is Padding enabled setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline
                << "' threw an exception getting the Stream Muxer padding");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreamMuxPaddingSet(const char* pipeline,
        boolean enabled)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->SetStreamMuxPadding((bool)enabled))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer is Padding enabled setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer padding");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreamMuxNumSurfacesPerFrameGet(const char* pipeline,
        uint* num)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->GetStreamMuxNumSurfacesPerFrame(num))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer num-surfaces-per-frame setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline
                << "' threw an exception getting the Stream Muxer num-surfaces-per-frame");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStreamMuxNumSurfacesPerFrameSet(const char* pipeline,
        uint num)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (num > 4)
            {
                LOG_ERROR("The value of '" << num 
                    << "' is invalid for Stream Muxer num-surfaces-per-frame setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            
            if (!m_pipelines[pipeline]->SetStreamMuxNumSurfacesPerFrame(num))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Get the Stream Muxer is Padding enabled setting");
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the Stream Muxer padding");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineXWindowHandleGet(const char* pipeline, uint64_t* xwindow) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            *xwindow = m_pipelines[pipeline]->GetXWindow();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline << "' threw an exception getting XWindow handle");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineXWindowHandleSet(const char* pipeline, uint64_t xwindow)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->SetXWindow(xwindow))
            {
                LOG_ERROR("Failure setting XWindow handle for Pipeline '" << pipeline << "'");
                return DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline << "' threw an exception setting XWindow handle");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineXWindowClear(const char* pipeline)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->ClearXWindow())
            {
                LOG_ERROR("Pipeline '" << pipeline << "' failed to clear its XWindow");
                return DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED;
            }
            LOG_ERROR("Pipeline '" << pipeline << "' successfully cleared its XWindow");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline << "' threw an exception clearing its XWindow");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineXWindowDestroy(const char* pipeline)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->DestroyXWindow())
            {
                LOG_ERROR("Pipeline '" << pipeline << "' failed to destroy its XWindow");
                return DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED;
            }
            LOG_ERROR("Pipeline '" << pipeline << "' successfully destroyed its XWindow");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline << "' threw an exception destroying its XWindow");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineXWindowOffsetsGet(const char* pipeline,
        uint* xOffset, uint* yOffset)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            m_pipelines[pipeline]->GetXWindowOffsets(xOffset, yOffset);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting the XWindow Offsets");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineXWindowDimensionsGet(const char* pipeline,
        uint* width, uint* height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            m_pipelines[pipeline]->GetXWindowDimensions(width, height);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting the XWindow Dimensions");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowFullScreenEnabledGet(const char* pipeline, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            *enabled = (boolean)m_pipelines[pipeline]->GetXWindowFullScreenEnabled();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting the XWindow full-screen-enabled setting");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowFullScreenEnabledSet(const char* pipeline, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->SetXWindowFullScreenEnabled(enabled))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to set the XWindow full-screen-enabled setting");
                return DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the XWindow full-screen-enabled setting");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

            
    DslReturnType Services::PipelinePause(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        if (!std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->Stop())
        {
            return DSL_RESULT_PIPELINE_FAILED_TO_STOP;
        }

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineStateGet(const char* pipeline, uint* state)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            GstState gstState;
            std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->GetState(gstState, 0);
            *state = (uint)gstState;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting state");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineIsLive(const char* pipeline, boolean* isLive)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            *isLive = std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->IsLive();
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting 'is-live'");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineDumpToDot(const char* pipeline, char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[pipeline]->DumpToDot(filename);
        
        return DSL_RESULT_SUCCESS;
    }   
    
    DslReturnType Services::PipelineDumpToDotWithTs(const char* pipeline, char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[pipeline]->DumpToDot(filename);

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineStateChangeListenerAdd(const char* pipeline, 
        dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddStateChangeListener(listener, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineStateChangeListenerRemove(const char* pipeline, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->RemoveStateChangeListener(listener))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineEosListenerAdd(const char* pipeline, 
        dsl_eos_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddEosListener(listener, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add an EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding an EOS Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineEosListenerRemove(const char* pipeline, 
        dsl_eos_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->RemoveEosListener(listener))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove an EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing an EOS Listener");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineErrorMessageHandlerAdd(const char* pipeline, 
        dsl_error_message_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddErrorMessageHandler(handler, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add an Error Message Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding an Error Message Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PipelineErrorMessageHandlerRemove(const char* pipeline, 
        dsl_error_message_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->RemoveErrorMessageHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove an Error Message Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing an Error Message Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineErrorMessageLastGet(const char* pipeline,
        std::wstring& source, std::wstring& message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
    
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            m_pipelines[pipeline]->GetLastErrorMessage(source, message);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing an Error Message Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PipelineXWindowKeyEventHandlerAdd(const char* pipeline, 
        dsl_xwindow_key_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddXWindowKeyEventHandler(handler, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add XWindow Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding XWindow Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowKeyEventHandlerRemove(const char* pipeline, 
        dsl_xwindow_key_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->RemoveXWindowKeyEventHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove XWindow Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing XWindow Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowButtonEventHandlerAdd(const char* pipeline, 
        dsl_xwindow_button_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->AddXWindowButtonEventHandler(handler, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add XWindow Button Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding XWindow Button Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowButtonEventHandlerRemove(const char* pipeline, 
        dsl_xwindow_button_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->RemoveXWindowButtonEventHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove XWindow Button Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing XWindow Button Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowDeleteEventHandlerAdd(const char* pipeline, 
        dsl_xwindow_delete_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->AddXWindowDeleteEventHandler(handler, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add XWindow Delete Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding XWindow Delete Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PipelineXWindowDeleteEventHandlerRemove(const char* pipeline, 
        dsl_xwindow_delete_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

            if (!m_pipelines[pipeline]->RemoveXWindowDeleteEventHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove XWindow Delete Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing XWindow Delete Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerNew(const char* name, 
        const char* source, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, source);
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, sink);
            DSL_RETURN_IF_COMPONENT_IS_NOT_SINK(m_components, sink);
        
            if (m_players.find(name) != m_players.end())
            {   
                LOG_ERROR("Player name '" << name << "' is not unique");
                return DSL_RESULT_PLAYER_NAME_NOT_UNIQUE;
            }
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[source]);

            DSL_SINK_PTR pSinkBintr = 
                std::dynamic_pointer_cast<SinkBintr>(m_components[sink]);
            
            m_players[name] = std::shared_ptr<PlayerBintr>(new 
                PlayerBintr(name, pSourceBintr, pSinkBintr));
        }
        catch(...)
        {
            LOG_ERROR("New Player '" << name << "' threw exception on create");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
        LOG_INFO("New Player '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PlayerRenderVideoNew(const char* name, const char* filePath,
            uint renderType, uint offsetX, uint offsetY, uint zoom, boolean repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
        
            if (m_players.find(name) != m_players.end())
            {   
                LOG_ERROR("Player name '" << name << "' is not unique");
                return DSL_RESULT_PLAYER_NAME_NOT_UNIQUE;
            }
            std::string pathString(filePath);
            if (pathString.size())
            {
                std::ifstream streamUriFile(filePath);
                if (!streamUriFile.good())
                {
                    LOG_ERROR("File Source'" << filePath << "' Not found");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
            }
            m_players[name] = std::shared_ptr<VideoRenderPlayerBintr>(new 
                VideoRenderPlayerBintr(name, filePath, renderType,
                    offsetX, offsetY, zoom, repeatEnabled));
                    
            LOG_INFO("New Render File Player '" << name << "' created successfully");
        }
        catch(...)
        {
            LOG_ERROR("New Render File Player '" << name << "' threw exception on create");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PlayerRenderImageNew(const char* name, const char* filePath,
            uint renderType, uint offsetX, uint offsetY, uint zoom, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_players.find(name) != m_players.end())
            {   
                LOG_ERROR("Player name '" << name << "' is not unique");
                return DSL_RESULT_PLAYER_NAME_NOT_UNIQUE;
            }
            std::string pathString(filePath);
            if (pathString.size())
            {
                std::ifstream streamUriFile(filePath);
                if (!streamUriFile.good())
                {
                    LOG_ERROR("File Source'" << filePath << "' Not found");
                    return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
                }
            }
            m_players[name] = std::shared_ptr<ImageRenderPlayerBintr>(new 
                ImageRenderPlayerBintr(name, filePath, renderType,
                    offsetX, offsetY, zoom, timeout));
                    
            LOG_INFO("New Render Image Player '" << name << "' created successfully");
        }
        catch(...)
        {
            LOG_ERROR("New Render Image Player '" << name 
                << "' threw exception on create");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PlayerRenderFilePathGet(const char* name, 
        const char** filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            *filePath = pRenderPlayer->GetFilePath();
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception getting File Path");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::PlayerRenderFilePathSet(const char* name, const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->SetFilePath(filePath))
            {
                LOG_ERROR("Failed to Set File Path '" << filePath 
                    << "' for Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception setting File Path");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderFilePathQueue(const char* name, 
        const char* filePath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->QueueFilePath(filePath))
            {
                LOG_ERROR("Failed to Queue File Path '" << filePath 
                    << "' for Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception queuing File Path");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderOffsetsGet(const char* name, uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            pRenderPlayer->GetOffsets(offsetX, offsetY);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name << "' threw an exception getting offsets");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderOffsetsSet(const char* name, uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->SetOffsets(offsetX, offsetY))
            {
                LOG_ERROR("Render Player '" << name << "' failed to set offsets");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderZoomGet(const char* name, uint* zoom)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            *zoom = pRenderPlayer->GetZoom();
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception getting Zoom");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::PlayerRenderZoomSet(const char* name, uint zoom)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->SetZoom(zoom))
            {
                LOG_ERROR("Failed to Set Zooom '" << zoom 
                    << "' for Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception setting Zoom");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderReset(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->Reset())
            {
                LOG_ERROR("Failed to Reset Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Render Player '" << name 
                << "' threw exception on Reset");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderImageTimeoutGet(const char* name, 
        uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, name, ImageRenderPlayerBintr);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pImageRenderPlayer = 
                std::dynamic_pointer_cast<ImageRenderPlayerBintr>(m_players[name]);

            *timeout = pImageRenderPlayer->GetTimeout();
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Render Player '" << name 
                << "' threw exception getting Timeout");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
            

    DslReturnType Services::PlayerRenderImageTimeoutSet(const char* name, 
        uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, name, ImageRenderPlayerBintr);

            DSL_PLAYER_RENDER_IMAGE_BINTR_PTR pImageRenderPlayer = 
                std::dynamic_pointer_cast<ImageRenderPlayerBintr>(m_players[name]);

            if (!pImageRenderPlayer->SetTimeout(timeout))
            {
                LOG_ERROR("Failed to Set Timeout to '" << timeout 
                    << "s' for Image Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Render Player '" << name 
                << "' threw exception setting Timeout");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderVideoRepeatEnabledGet(const char* name, 
        boolean* repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, name, VideoRenderPlayerBintr);

            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pVideoRenderPlayer = 
                std::dynamic_pointer_cast<VideoRenderPlayerBintr>(m_players[name]);

            *repeatEnabled = pVideoRenderPlayer->GetRepeatEnabled();
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Image Render Player '" << name 
                << "' threw exception getting Timeout");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerRenderVideoRepeatEnabledSet(const char* name, 
        boolean repeatEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_players, name, VideoRenderPlayerBintr);

            DSL_PLAYER_RENDER_VIDEO_BINTR_PTR pVideoRenderPlayer = 
                std::dynamic_pointer_cast<VideoRenderPlayerBintr>(m_players[name]);

            if (!pVideoRenderPlayer->SetRepeatEnabled(repeatEnabled))
            {
                LOG_ERROR("Failed to Set Repeat Enabled to '" << repeatEnabled 
                    << "' for Video Render Player '" << name << "'");
                return DSL_RESULT_PLAYER_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Video Render Player '" << name 
                << "' threw exception setting Repeat Enabled");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerTerminationEventListenerAdd(const char* name,
        dsl_player_termination_event_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

        try
        {
            if (!m_players[name]->AddTerminationEventListener(listener, clientData))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to add Termination Event Listener");
                return DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception adding Termination Event Listner");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PlayerTerminationEventListenerRemove(const char* name,
        dsl_player_termination_event_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            
            if (!m_players[name]->RemoveTerminationEventListener(listener))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to remove Termination Event Listener");
                return DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception adding Termination Event Listner");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerXWindowHandleGet(const char* name, uint64_t* xwindow) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            
            *xwindow = m_players[name]->GetXWindow();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name << "' threw an exception getting XWindow handle");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::PlayerXWindowHandleSet(const char* name, uint64_t xwindow)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            
            if (!m_players[name]->SetXWindow(xwindow))
            {
                LOG_ERROR("Failure setting XWindow handle for Player '" << name << "'");
                return DSL_RESULT_PLAYER_XWINDOW_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name << "' threw an exception setting XWindow handle");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerXWindowKeyEventHandlerAdd(const char* name, 
        dsl_xwindow_key_event_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->AddXWindowKeyEventHandler(handler, clientData))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to add XWindow Key Event Handler");
                return DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception adding XWindow Key Event Handler");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerXWindowKeyEventHandlerRemove(const char* name, 
        dsl_xwindow_key_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->RemoveXWindowKeyEventHandler(handler))
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to remove XWindow Key Event Handler");
                return DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception removing XWindow Key Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerPlay(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->Play())
            {
                return DSL_RESULT_PLAYER_FAILED_TO_PLAY;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Play");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PlayerPause(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            if (!m_players[name]->Pause())
            {
                return DSL_RESULT_PLAYER_FAILED_TO_PAUSE;
            }

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Pause");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerStop(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

        if (!m_players[name]->Stop())
        {
            return DSL_RESULT_PLAYER_FAILED_TO_STOP;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PlayerRenderNext(const char* name)
    {
        LOG_FUNC();

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);

            DSL_PLAYER_RENDER_BINTR_PTR pRenderPlayer = 
                std::dynamic_pointer_cast<RenderPlayerBintr>(m_players[name]);

            if (!pRenderPlayer->Next())
            {
                LOG_ERROR("Player '" << name 
                    << "' failed to Play Next");
                return DSL_RESULT_PLAYER_RENDER_FAILED_TO_PLAY_NEXT;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Play Next");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PlayerStateGet(const char* name, uint* state)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);
            DSL_RETURN_IF_PLAYER_IS_NOT_RENDER_PLAYER(m_players, name);
            GstState gstState;
            m_players[name]->GetState(gstState, 0);
            *state = (uint)gstState;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception getting state");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    boolean Services::PlayerExists(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            return (boolean)(m_players.find(name) != m_players.end());
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on check for Exists");
            return false;
        }
    }

    DslReturnType Services::PlayerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, name);

            m_players.erase(name);

            LOG_INFO("Player '" << name << "' deleted successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Player '" << name 
                << "' threw an exception on Delete");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }

    }

    DslReturnType Services::PlayerDeleteAll(bool checkInUse)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_players.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto &imap: m_players)
            {
                // In the case of DSL Delete all - we don't check for in-use
                // as their can be a circular type ownership/relation that will
                // cause it to fail... i.e. players can own record sinks which 
                // can own players, and so on...
                if (checkInUse and imap.second.use_count() > 1)
                {
                    LOG_ERROR("Can't delete Player '" << imap.second->GetName() 
                        << "' as it is currently in use");
                    return DSL_RESULT_PLAYER_IN_USE;
                }

                imap.second->RemoveAllChildren();
                imap.second = nullptr;
            }

            m_players.clear();

            LOG_INFO("All Players deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on PlayerDeleteAll");
            return DSL_RESULT_PLAYER_THREW_EXCEPTION;
        }
    }

    uint Services::PlayerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_players.size();
    }

    DslReturnType Services::MailerNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_mailers.find(name) != m_mailers.end())
            {   
                LOG_ERROR("Mailer name '" << name << "' is not unique");
                return DSL_RESULT_MAILER_NAME_NOT_UNIQUE;
            }
            m_mailers[name] = std::shared_ptr<Mailer>(new Mailer(name));
            LOG_INFO("New Mailer '" << name << "' created successfully");
        }
        catch(...)
        {
            LOG_ERROR("New Mailer '" << name << "' threw exception on create");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::MailerEnabledGet(const char* name, 
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);
            
            *enabled = m_mailers[name]->GetEnabled();
            LOG_INFO("Returning Mailer Enabled = " << *enabled);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception on get enabled setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerEnabledSet(const char* name, 
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetEnabled(enabled);
            LOG_INFO("Setting SMTP Mail Enabled = " << enabled);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception on set enabled setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
     
    DslReturnType Services::MailerCredentialsSet(const char* name,
        const char* username, const char* password)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);
            
            m_mailers[name]->SetCredentials(username, password);

            LOG_INFO("New SMTP Username and Password set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception setting the enabled setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerServerUrlGet(const char* name,
        const char** serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->GetServerUrl(serverUrl);

            LOG_INFO("Returning SMTP Server URL = '" << *serverUrl << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the Server URL");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerServerUrlSet(const char* name,
        const char* serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetServerUrl(serverUrl);

            LOG_INFO("New SMTP Server URL = '" << serverUrl << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception setting the Server URL");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerFromAddressGet(const char* name,
        const char** displayName, const char** address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->GetFromAddress(displayName, address);

            LOG_INFO("Returning SMTP From Address with Name = '" << *name 
                << "', and Address = '" << *address << "'" );
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the From Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerFromAddressSet(const char* name,
        const char* displayName, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetFromAddress(displayName, address);

            LOG_INFO("New SMTP From Address with Name = '" << name 
                << "', and Address = '" << address << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception setting the From Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerSslEnabledGet(const char* name,
        boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            *enabled = m_mailers[name]->GetSslEnabled();
            
            LOG_INFO("Returning SSL Enabled = '" << *enabled  << "'" );
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the SSL Enabled Setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerSslEnabledSet(const char* name,
        boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->SetSslEnabled(enabled);
            LOG_INFO("Set SSL Enabled = '" << enabled  << "'" );
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception getting the SSL Enabled Setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerToAddressAdd(const char* name,
        const char* displayName, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->AddToAddress(displayName, address);

            LOG_INFO("New To Address with Name = '" << name 
                << "', and Address = '" << address << "' added");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception adding a To Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerToAddressesRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->RemoveAllToAddresses();

            LOG_INFO("All To Addresses removed");
        
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception removing SSL Enabled Setting");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerCcAddressAdd(const char* name,
        const char* displayName, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->AddCcAddress(displayName, address);

            LOG_INFO("New Cc Address with Name = '" << name 
                << "', and Address = '" << address << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception adding a Cc Address");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::MailerCcAddressesRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            m_mailers[name]->RemoveAllCcAddresses();

            LOG_INFO("All Cc Addresses removed");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception removing all Cc Addresses");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::MailerSendTestMessage(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);

            std::string subject("Test message");
            std::string bline1("Test message.\r\n");
            
            std::vector<std::string> body{bline1};

            if (!m_mailers[name]->QueueMessage(subject, body))
            {
                LOG_ERROR("Failed to queue SMTP Test Message");
                return DSL_RESULT_FAILURE;
            }
            LOG_INFO("Test message Queued successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw exception queuing a Test Message");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }
 
    boolean Services::MailerExists(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            return (boolean)(m_mailers.find(name) != m_mailers.end());
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw an exception on check for Exists");
            return false;
        }
    }

    DslReturnType Services::MailerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, name);
            
            if (m_mailers[name]->IsInUse())
            {
                LOG_ERROR("Cannot delete Mailer '" << name 
                    << "' as it is currently in use");
                return DSL_RESULT_MAILER_IN_USE;
            }

            m_mailers.erase(name);

            LOG_INFO("Mailer '" << name << "' deleted successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Mailer '" << name 
                << "' threw an exception on Delete");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }

    }

    DslReturnType Services::MailerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_mailers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto &imap: m_mailers)
            {
                // In the case of Delete all
                if (imap.second.use_count() > 1)
                {
                    LOG_ERROR("Can't delete Player '" << imap.second->GetName() 
                        << "' as it is currently in use");
                    return DSL_RESULT_MAILER_IN_USE;
                }
            }
            m_mailers.clear();

            LOG_INFO("All Mailers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception on MailerDeleteAll");
            return DSL_RESULT_MAILER_THREW_EXCEPTION;
        }
    }

    uint Services::MailerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_mailers.size();
    }

    void Services::DeleteAll()
    {
        LOG_FUNC();
        // DO NOT lock mutex - will be done by each service
        
        PipelineDeleteAll();
        PlayerDeleteAll(false);
        ComponentDeleteAll();
        PphDeleteAll();
        OdeTriggerDeleteAll();
        OdeAreaDeleteAll();
        OdeActionDeleteAll();
        DisplayTypeDeleteAll();
        MailerDeleteAll();
    }
    
    DslReturnType Services::StdOutRedirect(const char* filepath)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (m_stdOutRedirectFile.is_open())
            {
                LOG_ERROR("stdout is currently/already in a redirected state");
                return DSL_RESULT_FAILURE;
            }
            
            // backup the default 
            m_stdOutRdBufBackup = std::cout.rdbuf();
            
            // open the redirect file and the rdbuf
            m_stdOutRedirectFile.open(filepath, std::ios::out);
            std::streambuf* redirectFileRdBuf = m_stdOutRedirectFile.rdbuf();
            
            // assign the file's rdbuf to the stdout's
            std::cout.rdbuf(redirectFileRdBuf);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception redirecting stdout");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    void Services::StdOutRestore()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            if (!m_stdOutRedirectFile.is_open())
            {
                LOG_ERROR("stdout is not currently in a redirected state");
                return;
            }

            // restore the stdout to the initial backupt
            std::cout.rdbuf(m_stdOutRdBufBackup);

            // close the redirct file
            m_stdOutRedirectFile.close();
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception close stdout redirect file");
        }
    }
   
    // ------------------------------------------------------------------------------
    
    bool Services::IsSourceComponent(const char* component)
    {
        LOG_FUNC();
     
        return (m_components[component]->IsType(typeid(CsiSourceBintr)) or 
            m_components[component]->IsType(typeid(UriSourceBintr)) or
            m_components[component]->IsType(typeid(RtspSourceBintr)));
    }
 
    uint Services::GetNumSourcesInUse()
    {
        LOG_FUNC();
        
        uint numInUse(0);
        
        for (auto const& imap: m_pipelines)
        {
            numInUse += imap.second->GetNumSourcesInUse();
        }
        return numInUse;
    }
    
    bool Services::IsSinkComponent(const char* component)
    {
        LOG_FUNC();
     
        return (m_components[component]->IsType(typeid(FakeSinkBintr)) or 
            m_components[component]->IsType(typeid(OverlaySinkBintr)) or
            m_components[component]->IsType(typeid(WindowSinkBintr)) or
            m_components[component]->IsType(typeid(FileSinkBintr)) or
            m_components[component]->IsType(typeid(RtspSinkBintr)));
    }
 
    uint Services::GetNumSinksInUse()
    {
        LOG_FUNC();
        
        uint numInUse(0);
        
        for (auto const& imap: m_pipelines)
        {
            numInUse += imap.second->GetNumSinksInUse();
        }
        return numInUse;
    }

    const wchar_t* Services::ReturnValueToString(uint result)
    {
        LOG_FUNC();
        
        if (m_returnValueToString.find(result) == m_returnValueToString.end())
        {
            LOG_ERROR("Invalid result = " << result << " unable to convert to string");
            return m_returnValueToString[DSL_RESULT_INVALID_RESULT_CODE].c_str();
        }

        std::string cstrResult(m_returnValueToString[result].begin(), m_returnValueToString[result].end());
        LOG_INFO("Result = " << result << " = " << cstrResult);
        return m_returnValueToString[result].c_str();
    }
    
    const wchar_t* Services::StateValueToString(uint state)
    {
        LOG_FUNC();
        
        if (m_stateValueToString.find(state) == m_stateValueToString.end())
        {
            state = DSL_STATE_UNKNOWN;
        }

        std::string cstrState(m_stateValueToString[state].begin(), m_stateValueToString[state].end());
        LOG_INFO("State = " << state << " = " << cstrState);
        return m_stateValueToString[state].c_str();
    }
    
    void Services::InitToStringMaps()
    {
        LOG_FUNC();
        
        m_mapParserTypes[DSL_SOURCE_CODEC_PARSER_H264] = "h264parse";
        m_mapParserTypes[DSL_SOURCE_CODEC_PARSER_H265] = "h265parse";
        
        m_stateValueToString[DSL_STATE_NULL] = L"DSL_STATE_NULL";
        m_stateValueToString[DSL_STATE_READY] = L"DSL_STATE_READY";
        m_stateValueToString[DSL_STATE_PAUSED] = L"DSL_STATE_PAUSED";
        m_stateValueToString[DSL_STATE_PLAYING] = L"DSL_STATE_PLAYING";
        m_stateValueToString[DSL_STATE_CHANGE_ASYNC] = L"DSL_STATE_CHANGE_ASYNC";
        m_stateValueToString[DSL_STATE_UNKNOWN] = L"DSL_STATE_UNKNOWN";

        m_returnValueToString[DSL_RESULT_SUCCESS] = L"DSL_RESULT_SUCCESS";
        m_returnValueToString[DSL_RESULT_FAILURE] = L"DSL_RESULT_FAILURE";
        m_returnValueToString[DSL_RESULT_INVALID_INPUT_PARAM] = L"DSL_RESULT_INVALID_INPUT_PARAM";
        m_returnValueToString[DSL_RESULT_THREW_EXCEPTION] = L"DSL_RESULT_THREW_EXCEPTION";
        
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE] = L"DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_NOT_FOUND] = L"DSL_RESULT_COMPONENT_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_BAD_FORMAT] = L"DSL_RESULT_COMPONENT_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_COMPONENT_THREW_EXCEPTION] = L"DSL_RESULT_COMPONENT_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_COMPONENT_IN_USE] = L"DSL_RESULT_COMPONENT_IN_USE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE] = L"DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH] = L"DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH";
        m_returnValueToString[DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE";
        m_returnValueToString[DSL_RESULT_COMPONENT_SET_GPUID_FAILED] = L"DSL_RESULT_COMPONENT_SET_GPUID_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_NAME_NOT_UNIQUE] = L"DSL_RESULT_SOURCE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_SOURCE_NAME_NOT_FOUND] = L"DSL_RESULT_SOURCE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_SOURCE_NAME_BAD_FORMAT] = L"DSL_RESULT_SOURCE_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_SOURCE_THREW_EXCEPTION] = L"DSL_RESULT_SOURCE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_SOURCE_FILE_NOT_FOUND] = L"DSL_RESULT_SOURCE_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_SOURCE_NOT_IN_USE] = L"DSL_RESULT_SOURCE_NOT_IN_USE";
        m_returnValueToString[DSL_RESULT_SOURCE_NOT_IN_PLAY] = L"DSL_RESULT_SOURCE_NOT_IN_PLAY";
        m_returnValueToString[DSL_RESULT_SOURCE_NOT_IN_PAUSE] = L"DSL_RESULT_SOURCE_NOT_IN_PAUSE";
        m_returnValueToString[DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE] = L"DSL_RESULT_SOURCE_FAILED_TO_CHANGE_STATE";
        m_returnValueToString[DSL_RESULT_SOURCE_CODEC_PARSER_INVALID] = L"DSL_RESULT_SOURCE_CODEC_PARSER_INVALID";
        m_returnValueToString[DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED] = L"DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED] = L"DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_TAP_ADD_FAILED] = L"DSL_RESULT_SOURCE_TAP_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_TAP_REMOVE_FAILED] = L"DSL_RESULT_SOURCE_TAP_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE] = L"DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE";
        m_returnValueToString[DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED] = L"DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED] = L"DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_SET_FAILED] = L"DSL_RESULT_SOURCE_SET_FAILED";
        m_returnValueToString[DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE] = L"DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DEWARPER_NAME_NOT_FOUND] = L"DSL_RESULT_DEWARPER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_DEWARPER_NAME_BAD_FORMAT] = L"DSL_RESULT_DEWARPER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_DEWARPER_THREW_EXCEPTION] = L"DSL_RESULT_DEWARPER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TRACKER_NAME_NOT_UNIQUE] = L"DSL_RESULT_TRACKER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_TRACKER_NAME_NOT_FOUND] = L"DSL_RESULT_TRACKER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TRACKER_NAME_BAD_FORMAT] = L"DSL_RESULT_TRACKER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_TRACKER_THREW_EXCEPTION] = L"DSL_RESULT_TRACKER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_TRACKER_CONFIG_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TRACKER_IS_IN_USE] = L"DSL_RESULT_TRACKER_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_TRACKER_SET_FAILED] = L"DSL_RESULT_TRACKER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_TRACKER_HANDLER_ADD_FAILED] = L"DSL_RESULT_TRACKER_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TRACKER_PAD_TYPE_INVALID] = L"DSL_RESULT_TRACKER_PAD_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER] = L"DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER";
        m_returnValueToString[DSL_RESULT_PPH_NAME_NOT_UNIQUE] = L"DSL_RESULT_PPH_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_PPH_NAME_NOT_FOUND] = L"DSL_RESULT_PPH_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_PPH_NAME_BAD_FORMAT] = L"DSL_RESULT_PPH_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_PPH_THREW_EXCEPTION] = L"DSL_RESULT_PPH_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_PPH_IS_IN_USE] = L"DSL_RESULT_PPH_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_PPH_SET_FAILED] = L"DSL_RESULT_PPH_SET_FAILED";
        m_returnValueToString[DSL_RESULT_PPH_ODE_TRIGGER_ADD_FAILED] = L"DSL_RESULT_PPH_ODE_TRIGGER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_PPH_ODE_TRIGGER_REMOVE_FAILED] = L"DSL_RESULT_PPH_ODE_TRIGGER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE] = L"DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE";
        m_returnValueToString[DSL_RESULT_PPH_METER_INVALID_INTERVAL] = L"DSL_RESULT_PPH_METER_INVALID_INTERVAL";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION] = L"DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_IN_USE] = L"DSL_RESULT_ODE_TRIGGER_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_SET_FAILED] = L"DSL_RESULT_ODE_TRIGGER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_IS_NOT_ODE_TRIGGER] = L"DSL_RESULT_ODE_TRIGGER_IS_NOT_ODE_TRIGGER";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_ACTION_ADD_FAILED] = L"DSL_RESULT_ODE_TRIGGER_ACTION_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_ACTION_REMOVE_FAILED] = L"DSL_RESULT_ODE_TRIGGER_ACTION_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE] = L"DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_AREA_ADD_FAILED] = L"DSL_RESULT_ODE_TRIGGER_AREA_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_AREA_REMOVE_FAILED] = L"DSL_RESULT_ODE_TRIGGER_AREA_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE] = L"DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_CLIENT_CALLBACK_INVALID] = L"DSL_RESULT_ODE_TRIGGER_CLIENT_CALLBACK_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID] = L"DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE] = L"DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_THREW_EXCEPTION] = L"DSL_RESULT_ODE_ACTION_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_IN_USE] = L"DSL_RESULT_ODE_ACTION_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_SET_FAILED] = L"DSL_RESULT_ODE_ACTION_SET_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_IS_NOT_ACTION] = L"DSL_RESULT_ODE_ACTION_IS_NOT_ACTION";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND] = L"DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID] = L"DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_PLAYER_ADD_FAILED] = L"DSL_RESULT_ODE_ACTION_PLAYER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_PLAYER_REMOVE_FAILED] = L"DSL_RESULT_ODE_ACTION_PLAYER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_MAILER_ADD_FAILED] = L"DSL_RESULT_ODE_ACTION_MAILER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_MAILER_REMOVE_FAILED] = L"DSL_RESULT_ODE_ACTION_MAILER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED] = L"DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_PARAMETER_INVALID] = L"DSL_RESULT_ODE_ACTION_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED] = L"DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_AREA_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_AREA_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_AREA_THREW_EXCEPTION] = L"DSL_RESULT_ODE_AREA_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_AREA_PARAMETER_INVALID] = L"DSL_RESULT_ODE_AREA_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_AREA_SET_FAILED] = L"DSL_RESULT_ODE_AREA_SET_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_NAME_NOT_UNIQUE] = L"DSL_RESULT_SINK_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_SINK_NAME_NOT_FOUND] = L"DSL_RESULT_SINK_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_SINK_NAME_BAD_FORMAT] = L"DSL_RESULT_SINK_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_SINK_THREW_EXCEPTION] = L"DSL_RESULT_SINK_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_SINK_FILE_PATH_NOT_FOUND] = L"DSL_RESULT_SINK_FILE_PATH_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_SINK_IS_IN_USE] = L"DSL_RESULT_SINK_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_SINK_SET_FAILED] = L"DSL_RESULT_SINK_SET_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_CODEC_VALUE_INVALID] = L"DSL_RESULT_SINK_CODEC_VALUE_INVALID";
        m_returnValueToString[DSL_RESULT_SINK_CONTAINER_VALUE_INVALID] = L"DSL_RESULT_SINK_CONTAINER_VALUE_INVALID";
        m_returnValueToString[DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK] = L"DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK";
        m_returnValueToString[DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK] = L"DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK";
        m_returnValueToString[DSL_RESULT_SINK_COMPONENT_IS_NOT_RENDER_SINK] = L"DSL_RESULT_SINK_COMPONENT_IS_NOT_RENDER_SINK";
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_HANDLER_ADD_FAILED] = L"DSL_RESULT_SINK_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_SINK_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_PLAYER_ADD_FAILED] = L"DSL_RESULT_SINK_PLAYER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_PLAYER_REMOVE_FAILED] = L"DSL_RESULT_SINK_PLAYER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_MAILER_ADD_FAILED] = L"DSL_RESULT_SINK_MAILER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_MAILER_REMOVE_FAILED] = L"DSL_RESULT_SINK_MAILER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_OSD_NAME_NOT_UNIQUE] = L"DSL_RESULT_OSD_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_OSD_NAME_NOT_FOUND] = L"DSL_RESULT_OSD_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_OSD_NAME_BAD_FORMAT] = L"DSL_RESULT_OSD_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_OSD_THREW_EXCEPTION] = L"DSL_RESULT_OSD_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_OSD_MAX_DIMENSIONS_INVALID] = L"DSL_RESULT_OSD_MAX_DIMENSIONS_INVALID";
        m_returnValueToString[DSL_RESULT_OSD_IS_IN_USE] = L"DSL_RESULT_OSD_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_OSD_SET_FAILED] = L"DSL_RESULT_OSD_SET_FAILED";
        m_returnValueToString[DSL_RESULT_OSD_HANDLER_ADD_FAILED] = L"DSL_RESULT_OSD_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_OSD_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_OSD_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_OSD_PAD_TYPE_INVALID] = L"DSL_RESULT_OSD_PAD_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_OSD_COMPONENT_IS_NOT_OSD] = L"DSL_RESULT_OSD_COMPONENT_IS_NOT_OSD";
        m_returnValueToString[DSL_RESULT_OSD_COLOR_PARAM_INVALID] = L"DSL_RESULT_OSD_COLOR_PARAM_INVALID";
        m_returnValueToString[DSL_RESULT_INFER_NAME_NOT_UNIQUE] = L"DSL_RESULT_INFER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_INFER_NAME_NOT_FOUND] = L"DSL_RESULT_INFER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_INFER_NAME_BAD_FORMAT] = L"DSL_RESULT_INFER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_INFER_CONFIG_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_INFER_MODEL_FILE_NOT_FOUND] = L"DSL_RESULT_INFER_MODEL_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_INFER_THREW_EXCEPTION] = L"DSL_RESULT_INFER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_INFER_IS_IN_USE] = L"DSL_RESULT_INFER_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_INFER_SET_FAILED] = L"DSL_RESULT_INFER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_INFER_HANDLER_ADD_FAILED] = L"DSL_RESULT_INFER_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_INFER_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_INFER_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_INFER_PAD_TYPE_INVALID] = L"DSL_RESULT_INFER_PAD_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER] = L"DSL_RESULT_INFER_COMPONENT_IS_NOT_INFER";
        m_returnValueToString[DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST] = L"DSL_RESULT_INFER_OUTPUT_DIR_DOES_NOT_EXIST";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_NAME_NOT_UNIQUE] = L"DSL_RESULT_SEGVISUAL_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_NAME_NOT_FOUND] = L"DSL_RESULT_SEGVISUAL_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_THREW_EXCEPTION] = L"DSL_RESULT_SEGVISUAL_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_IN_USE] = L"DSL_RESULT_SEGVISUAL_IN_USE";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_SET_FAILED] = L"DSL_RESULT_SEGVISUAL_SET_FAILED";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_PARAMETER_INVALID] = L"DSL_RESULT_SEGVISUAL_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED] = L"DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TEE_NAME_NOT_UNIQUE] = L"DSL_RESULT_TEE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_TEE_NAME_NOT_FOUND] = L"DSL_RESULT_TEE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TEE_NAME_BAD_FORMAT] = L"DSL_RESULT_TEE_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_TEE_THREW_EXCEPTION] = L"DSL_RESULT_TEE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD] = L"DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD";
        m_returnValueToString[DSL_RESULT_TEE_BRANCH_IS_NOT_BRANCH] = L"DSL_RESULT_TEE_BRANCH_IS_NOT_BRANCH";
        m_returnValueToString[DSL_RESULT_TEE_BRANCH_ADD_FAILED] = L"DSL_RESULT_TEE_BRANCH_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TEE_BRANCH_REMOVE_FAILED] = L"DSL_RESULT_TEE_BRANCH_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TEE_HANDLER_ADD_FAILED] = L"DSL_RESULT_TEE_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TEE_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_TEE_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TEE_COMPONENT_IS_NOT_TEE] = L"DSL_RESULT_TEE_COMPONENT_IS_NOT_TEE";
        m_returnValueToString[DSL_RESULT_TILER_NAME_NOT_UNIQUE] = L"DSL_RESULT_TILER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_TILER_NAME_NOT_FOUND] = L"DSL_RESULT_TILER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TILER_NAME_BAD_FORMAT] = L"DSL_RESULT_TILER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_TILER_THREW_EXCEPTION] = L"DSL_RESULT_TILER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_TILER_IS_IN_USE] = L"DSL_RESULT_TILER_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_TILER_SET_FAILED] = L"DSL_RESULT_TILER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_TILER_HANDLER_ADD_FAILED] = L"DSL_RESULT_TILER_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TILER_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_TILER_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TILER_PAD_TYPE_INVALID] = L"DSL_RESULT_TILER_PAD_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_TILER_COMPONENT_IS_NOT_TILER] = L"DSL_RESULT_TILER_COMPONENT_IS_NOT_TILER";
        m_returnValueToString[DSL_RESULT_BRANCH_RESULT] = L"DSL_RESULT_BRANCH_RESULT";
        m_returnValueToString[DSL_RESULT_BRANCH_NAME_NOT_UNIQUE] = L"DSL_RESULT_BRANCH_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_BRANCH_NAME_NOT_FOUND] = L"DSL_RESULT_BRANCH_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_BRANCH_NAME_BAD_FORMAT] = L"DSL_RESULT_BRANCH_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_BRANCH_THREW_EXCEPTION] = L"DSL_RESULT_BRANCH_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED] = L"DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED] = L"DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_BRANCH_SOURCE_NOT_ALLOWED] = L"DSL_RESULT_BRANCH_SOURCE_NOT_ALLOWED";
        m_returnValueToString[DSL_RESULT_BRANCH_SINK_MAX_IN_USE_REACHED] = L"DSL_RESULT_BRANCH_SINK_MAX_IN_USE_REACHED";
        m_returnValueToString[DSL_RESULT_PIPELINE_RESULT] = L"DSL_RESULT_PIPELINE_RESULT";
        m_returnValueToString[DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE] = L"DSL_RESULT_PIPELINE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_PIPELINE_NAME_NOT_FOUND] = L"DSL_RESULT_PIPELINE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_PIPELINE_NAME_BAD_FORMAT] = L"DSL_RESULT_PIPELINE_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_PIPELINE_STATE_PAUSED] = L"DSL_RESULT_PIPELINE_STATE_PAUSED";
        m_returnValueToString[DSL_RESULT_PIPELINE_STATE_RUNNING] = L"DSL_RESULT_PIPELINE_STATE_RUNNING";
        m_returnValueToString[DSL_RESULT_PIPELINE_THREW_EXCEPTION] = L"DSL_RESULT_PIPELINE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED] = L"DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED] = L"DSL_RESULT_PIPELINE_COMPONENT_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED] = L"DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED] = L"DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_XWINDOW_GET_FAILED] = L"DSL_RESULT_PIPELINE_XWINDOW_GET_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED] = L"DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED] = L"DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED] = L"DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_PIPELINE_FAILED_TO_PLAY] = L"DSL_RESULT_PIPELINE_FAILED_TO_PLAY";
        m_returnValueToString[DSL_RESULT_PIPELINE_FAILED_TO_PAUSE] = L"DSL_RESULT_PIPELINE_FAILED_TO_PAUSE";
        m_returnValueToString[DSL_RESULT_PIPELINE_FAILED_TO_STOP] = L"DSL_RESULT_PIPELINE_FAILED_TO_STOP";
        m_returnValueToString[DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED] = L"DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACHED";
        m_returnValueToString[DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACHED] = L"DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACHED";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION] = L"DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_IN_USE] = L"DSL_RESULT_DISPLAY_TYPE_IN_USE";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND] = L"DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE] = L"DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_COLOR_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_COLOR_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_FONT_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_FONT_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_TEXT_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_TEXT_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_LINE_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_LINE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_ARROW_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_ARROW_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_ARROW_HEAD_INVALID] = L"DSL_RESULT_DISPLAY_RGBA_ARROW_HEAD_INVALID";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_RECTANGLE_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_RECTANGLE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_POLYGON_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_POLYGON_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_SOURCE_NAME_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_SOURCE_NAME_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_SOURCE_DIMENSIONS_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_SOURCE_DIMENSIONS_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_SOURCE_FRAMERATE_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_PARAMETER_INVALID] = L"DSL_RESULT_DISPLAY_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_TAP_NAME_NOT_UNIQUE] = L"DSL_RESULT_TAP_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_TAP_NAME_NOT_FOUND] = L"DSL_RESULT_TAP_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TAP_THREW_EXCEPTION] = L"DSL_RESULT_TAP_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_TAP_IN_USE] = L"DSL_RESULT_TAP_IN_USE";
        m_returnValueToString[DSL_RESULT_TAP_SET_FAILED] = L"DSL_RESULT_TAP_SET_FAILED";
        m_returnValueToString[DSL_RESULT_TAP_FILE_PATH_NOT_FOUND] = L"DSL_RESULT_TAP_FILE_PATH_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TAP_CONTAINER_VALUE_INVALID] = L"DSL_RESULT_TAP_CONTAINER_VALUE_INVALID";
        m_returnValueToString[DSL_RESULT_TAP_PLAYER_ADD_FAILED] = L"DSL_RESULT_TAP_PLAYER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TAP_PLAYER_REMOVE_FAILED] = L"DSL_RESULT_TAP_PLAYER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TAP_MAILER_ADD_FAILED] = L"DSL_RESULT_TAP_MAILER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TAP_MAILER_REMOVE_FAILED] = L"DSL_RESULT_TAP_MAILER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_PLAYER_RESULT] = L"DSL_RESULT_PLAYER_RESULT";
        m_returnValueToString[DSL_RESULT_PLAYER_NAME_NOT_UNIQUE] = L"DSL_RESULT_PLAYER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_PLAYER_NAME_NOT_FOUND] = L"DSL_RESULT_PLAYER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_PLAYER_NAME_BAD_FORMAT] = L"DSL_RESULT_PLAYER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_PLAYER_IS_NOT_RENDER_PLAYER] = L"DSL_RESULT_PLAYER_IS_NOT_RENDER_PLAYER";
        m_returnValueToString[DSL_RESULT_PLAYER_IS_NOT_IMAGE_PLAYER] = L"DSL_RESULT_PLAYER_IS_NOT_IMAGE_PLAYER";
        m_returnValueToString[DSL_RESULT_PLAYER_IS_NOT_VIDEO_PLAYER] = L"DSL_RESULT_PLAYER_IS_NOT_VIDEO_PLAYER";
        m_returnValueToString[DSL_RESULT_PLAYER_THREW_EXCEPTION] = L"DSL_RESULT_PLAYER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_PLAYER_XWINDOW_GET_FAILED] = L"DSL_RESULT_PLAYER_XWINDOW_GET_FAILED";
        m_returnValueToString[DSL_RESULT_PLAYER_XWINDOW_SET_FAILED] = L"DSL_RESULT_PLAYER_XWINDOW_SET_FAILED";
        m_returnValueToString[DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED] = L"DSL_RESULT_PLAYER_CALLBACK_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED] = L"DSL_RESULT_PLAYER_CALLBACK_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_PLAYER_FAILED_TO_PLAY] = L"DSL_RESULT_PLAYER_FAILED_TO_PLAY";
        m_returnValueToString[DSL_RESULT_PLAYER_FAILED_TO_PAUSE] = L"DSL_RESULT_PLAYER_FAILED_TO_PAUSE";
        m_returnValueToString[DSL_RESULT_PLAYER_FAILED_TO_STOP] = L"DSL_RESULT_PLAYER_FAILED_TO_STOP";
        m_returnValueToString[DSL_RESULT_PLAYER_RENDER_FAILED_TO_PLAY_NEXT] = L"DSL_RESULT_PLAYER_RENDER_FAILED_TO_PLAY_NEXT";
        m_returnValueToString[DSL_RESULT_PLAYER_SET_FAILED] = L"DSL_RESULT_PLAYER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_MAILER_NAME_NOT_UNIQUE] = L"DSL_RESULT_MAILER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_MAILER_NAME_NOT_FOUND] = L"DSL_RESULT_MAILER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_MAILER_THREW_EXCEPTION] = L"DSL_RESULT_MAILER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_MAILER_IN_USE] = L"DSL_RESULT_MAILER_IN_USE";
        m_returnValueToString[DSL_RESULT_MAILER_SET_FAILED] = L"DSL_RESULT_MAILER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_MAILER_PARAMETER_INVALID] = L"DSL_RESULT_MAILER_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_INVALID_RESULT_CODE] = L"Invalid DSL Result CODE";
   }

} // namespace
 