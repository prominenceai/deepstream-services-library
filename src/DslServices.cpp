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
#include "DslOdeTrigger.h"
#include "DslServices.h"
#include "DslSourceBintr.h"
#include "DslGieBintr.h"
#include "DslTrackerBintr.h"
#include "DslOdeHandlerBintr.h"
#include "DslTilerBintr.h"
#include "DslOsdBintr.h"
#include "DslSinkBintr.h"


#define RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(actions, name) do \
{ \
    if (actions.find(name) == actions.end()) \
    { \
        LOG_ERROR("ODE Action name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND; \
    } \
}while(0); 

#define RETURN_IF_ODE_AREA_NAME_NOT_FOUND(areas, name) do \
{ \
    if (areas.find(name) == areas.end()) \
    { \
        LOG_ERROR("ODE Area name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_AREA_NAME_NOT_FOUND; \
    } \
}while(0); 

#define RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(actions, name, action) do \
{ \
    if (!actions[name]->IsType(typeid(action)))\
    { \
        LOG_ERROR("ODE Action '" << name << "' is not the correct type"); \
        return DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(events, name) do \
{ \
    if (events.find(name) == events.end()) \
    { \
        LOG_ERROR("ODE Trigger name '" << name << "' was not found"); \
        return DSL_RESULT_ODE_TRIGGER_NAME_NOT_FOUND; \
    } \
}while(0); 


#define RETURN_IF_BRANCH_NAME_NOT_FOUND(branches, name) do \
{ \
    if (branches.find(name) == branches.end()) \
    { \
        LOG_ERROR("Branch name '" << name << "' was not found"); \
        return DSL_RESULT_BRANCH_NAME_NOT_FOUND; \
    } \
}while(0); 
    
#define RETURN_IF_PIPELINE_NAME_NOT_FOUND(pipelines, name) do \
{ \
    if (pipelines.find(name) == pipelines.end()) \
    { \
        LOG_ERROR("Pipeline name '" << name << "' was not found"); \
        return DSL_RESULT_PIPELINE_NAME_NOT_FOUND; \
    } \
}while(0); 
    
#define RETURN_IF_COMPONENT_NAME_NOT_FOUND(components, name) do \
{ \
    if (components.find(name) == components.end()) \
    { \
        LOG_ERROR("Component name '" << name << "' was not found"); \
        return DSL_RESULT_COMPONENT_NAME_NOT_FOUND; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(components, name, bintr) do \
{ \
    if (!components[name]->IsType(typeid(bintr)))\
    { \
        LOG_ERROR("Component '" << name << "' is not the correct type"); \
        return DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_SOURCE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(CsiSourceBintr)) and  \
        !components[name]->IsType(typeid(UsbSourceBintr)) and  \
        !components[name]->IsType(typeid(UriSourceBintr)) and  \
        !components[name]->IsType(typeid(RtspSourceBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Source"); \
        return DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(UriSourceBintr)) and  \
        !components[name]->IsType(typeid(RtspSourceBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Decode Source"); \
        return DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_GIE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(PrimaryGieBintr)) and  \
        !components[name]->IsType(typeid(SecondaryGieBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Primary or Secondary GIE"); \
        return DSL_RESULT_GIE_COMPONENT_IS_NOT_GIE; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_TRACKER(components, name) do \
{ \
    if (!components[name]->IsType(typeid(KtlTrackerBintr)) and  \
        !components[name]->IsType(typeid(IouTrackerBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Tracker"); \
        return DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_TEE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(DemuxerBintr)) and  \
        !components[name]->IsType(typeid(SplitterBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Tee"); \
        return DSL_RESULT_TEE_COMPONENT_IS_NOT_TEE; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_BRANCH(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(OverlaySinkBintr)) and  \
        !components[name]->IsType(typeid(WindowSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RtspSinkBintr)) and \
        !components[name]->IsType(typeid(BranchBintr)) and \
        !components[name]->IsType(typeid(DemuxerBintr)) and \
        !components[name]->IsType(typeid(BranchBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Branch type"); \
        return DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(OverlaySinkBintr)) and  \
        !components[name]->IsType(typeid(WindowSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RtspSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK; \
    } \
}while(0); 

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
        
            // If gst has not been initialized by the client software
            if (!gst_is_initialized())
            {
                int argc = 0;
                char** argv = NULL;
                
                // initialize the GStreamer library
                gst_init(&argc, &argv);
                doGstDeinit = true;
            }
            // Initialize the single debug category used by the lib
            GST_DEBUG_CATEGORY_INIT(GST_CAT_DSL, "DSL", 0, "DeepStream Services");
            
            // Safe to start logging
            LOG_INFO("Services Initialization");
            
            // Single instantiation for the lib's lifetime
            m_pInstatnce = new Services(doGstDeinit);
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
    
    DslReturnType Services::OdeActionCallbackNew(const char* name,
        dsl_ode_handle_occurrence_cb clientHandler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_CALLBACK_NEW(name, clientHandler, clientData);

            LOG_INFO("New ODE Callback Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Callback Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionCaptureFrameNew(const char* name,
        const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            // ensure outdir exists
            struct stat info;
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir << "' for Capture Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND;
            }
            m_odeActions[name] = DSL_ODE_ACTION_CAPTURE_FRAME_NEW(name, outdir);

            LOG_INFO("New Capture Frame ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Capture Frame ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionCaptureObjectNew(const char* name,
        const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            
            // ensure outdir exists
            struct stat info;
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir << "' for Capture Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND;
            }
            m_odeActions[name] = DSL_ODE_ACTION_CAPTURE_OBJECT_NEW(name, outdir);

            LOG_INFO("New Capture Object ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Capture Object ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionDisplayNew(const char* name, uint offsetX, uint offsetY, bool offsetYWithClassId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_DISPLAY_NEW(name, 
                offsetX, offsetY, offsetYWithClassId);
            LOG_INFO("New Display ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Display ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillAreaNew(const char* name,
        const char* area, double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_FILL_AREA_NEW(name, area,
                red, green, blue, alpha);

            LOG_INFO("New ODE Fill Area Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Area Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillFrameNew(const char* name,
        double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_FILL_FRAME_NEW(name,
                red, green, blue, alpha);

            LOG_INFO("New ODE Fill Frame Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Frame Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillObjectNew(const char* name,
        double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_FILL_OBJECT_NEW(name,
                red, green, blue, alpha);

            LOG_INFO("New ODE Fill Object Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Object Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionHandlerDisableNew(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_DISABLE_HANDLER_NEW(name, handler);

            LOG_INFO("New ODE Disable Handler Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Disable Handler Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionHideNew(const char* name, boolean text, boolean border)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_HIDE_NEW(name, text, border);

            LOG_INFO("New ODE Hide Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Hide Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionLogNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_LOG_NEW(name);

            LOG_INFO("New ODE Log Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Log Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionPauseNew(const char* name, const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_PAUSE_NEW(name, pipeline);

            LOG_INFO("New ODE Pause Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Pause Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionPrintNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_PRINT_NEW(name);

            LOG_INFO("New ODE Print Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Print Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionRecordStartNew(const char* name,
        const char* recordSink, uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure event name uniqueness 
        if (m_odeActions.find(name) != m_odeActions.end())
        {   
            LOG_ERROR("ODE Action name '" << name << "' is not unique");
            return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
        }
        try
        {
            m_odeActions[name] = DSL_ODE_ACTION_RECORD_START_NEW(name,
                recordSink, start, duration, clientData);

            LOG_INFO("New ODE Record Start Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Start Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionRedactNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure event name uniqueness 
        if (m_odeActions.find(name) != m_odeActions.end())
        {   
            LOG_ERROR("ODE Action name '" << name << "' is not unique");
            return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
        }
        try
        {
            m_odeActions[name] = DSL_ODE_ACTION_REDACT_NEW(name);

            LOG_INFO("New ODE Redact Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Redact Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkAddNew(const char* name, 
        const char* pipeline, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SINK_ADD_NEW(name, pipeline, sink);

            LOG_INFO("New Sink Add ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkRemoveNew(const char* name, 
        const char* pipeline, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SINK_REMOVE_NEW(name, pipeline, sink);

            LOG_INFO("New Sink Remove ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink Remove ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSourceAddNew(const char* name, 
        const char* pipeline, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SOURCE_ADD_NEW(name, pipeline, source);

            LOG_INFO("New Source Add ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Add ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSourceRemoveNew(const char* name, 
        const char* pipeline, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_SOURCE_REMOVE_NEW(name, pipeline, source);

            LOG_INFO("New Source Remove ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Remove ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionActionDisableNew(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_ACTION_DISABLE_NEW(name, action);

            LOG_INFO("New Action Disable ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Action Disable ODE Action'" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionActionEnableNew(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_ACTION_ENABLE_NEW(name, action);

            LOG_INFO("New Action Enable ODE Action'" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Action Enable ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTriggerResetNew(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TRIGGER_RESET_NEW(name, trigger);

            LOG_INFO("New Trigger Reset ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Trigger Reset ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTriggerDisableNew(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TRIGGER_DISABLE_NEW(name, trigger);

            LOG_INFO("New Trigger Disable ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Trigger Disable ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTriggerEnableNew(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_TRIGGER_ENABLE_NEW(name, trigger);

            LOG_INFO("New Trigger Enable ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Trigger Enable ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionAreaAddNew(const char* name, 
        const char* trigger, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_AREA_ADD_NEW(name, trigger, area);

            LOG_INFO("New Area Add ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Area Add ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionAreaRemoveNew(const char* name, 
        const char* trigger, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeActions.find(name) != m_odeActions.end())
            {   
                LOG_ERROR("ODE Action name '" << name << "' is not unique");
                return DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE;
            }
            m_odeActions[name] = DSL_ODE_ACTION_AREA_REMOVE_NEW(name, trigger, area);

            LOG_INFO("New Area Remove ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Area Remove ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(m_odeActions[name]);
         
            *enabled = pOdeAction->GetEnabled();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeActionEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
            DSL_ODE_ACTION_PTR pOdeAction = 
                std::dynamic_pointer_cast<OdeAction>(m_odeActions[name]);
         
            pOdeAction->SetEnabled(enabled);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name << "' threw exception setting Enabled");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeActionDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
        
        if (m_odeActions[name].use_count() > 1)
        {
            LOG_INFO("ODE Action'" << name << "' is in use");
            return DSL_RESULT_ODE_ACTION_IN_USE;
        }
        m_odeActions.erase(name);

        LOG_INFO("ODE Action '" << name << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeActionDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (auto const& imap: m_odeActions)
        {
            // In the case of Delete all
            if (imap.second.use_count() > 1)
            {
                LOG_ERROR("ODE Action '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_ODE_ACTION_IN_USE;
            }
        }
        m_odeActions.clear();

        LOG_INFO("All ODE Actions deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::OdeActionListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeActions.size();
    }
    
    DslReturnType Services::OdeAreaNew(const char* name, 
        uint left, uint top, uint width, uint height, boolean display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure ODE Area name uniqueness 
            if (m_odeAreas.find(name) != m_odeAreas.end())
            {   
                LOG_ERROR("ODE Area name '" << name << "' is not unique");
                return DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE;
            }
            
            m_odeAreas[name] = DSL_ODE_AREA_NEW(name, left, top, width, height, display);
         
            LOG_INFO("New ODE Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE ODE Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                
    DslReturnType Services::OdeAreaGet(const char* name, 
        uint* left, uint* top, uint* width, uint* height, boolean* display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, name);
            
            DSL_ODE_AREA_PTR pOdeArea = 
                std::dynamic_pointer_cast<OdeArea>(m_odeAreas[name]);
         
            pOdeArea->GetArea(left, top, width, height, (bool*)display);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Area criteria");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
            
    DslReturnType Services::OdeAreaSet(const char* name, 
        uint left, uint top, uint width, uint height, boolean display)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, name);
            
            DSL_ODE_AREA_PTR pOdeArea = 
                std::dynamic_pointer_cast<OdeArea>(m_odeAreas[name]);
         
            // TODO: validate the values for in-range
            pOdeArea->SetArea(left, top, width, height, display);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Area '" << name << "' threw exception setting Area criteria");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
            
    DslReturnType Services::OdeAreaColorGet(const char* name, 
        double* red, double* green, double* blue, double* alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, name);
            
            DSL_ODE_AREA_PTR pOdeArea = 
                std::dynamic_pointer_cast<OdeArea>(m_odeAreas[name]);
         
            pOdeArea->GetColor(red, green, blue, alpha);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Area Color");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
            
    DslReturnType Services::OdeAreaColorSet(const char* name, 
        double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, name);
            
            DSL_ODE_AREA_PTR pOdeArea = 
                std::dynamic_pointer_cast<OdeArea>(m_odeAreas[name]);
                
            LOG_INFO("Setting Area '" << name << "to: red = " << red << " green = " 
                << green << " blue = " << blue << " alpha = " << alpha);
                
            if ((red > 1.0) or (green > 1.0) or (blue > 1.0) or (alpha > 1.0))
            {
                LOG_ERROR("Invalid color value for ODE Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_SET_FAILED;
            }
            
            DSL_ODE_AREA_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeArea>(m_odeAreas[name]);
         
            pOdeArea->SetColor(red, green, blue, alpha);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Area '" << name << "' threw exception setting Color");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeAreaDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeAreas, name);
        
        if (m_odeAreas[name].use_count() > 1)
        {
            LOG_INFO("ODE Area'" << name << "' is in use");
            return DSL_RESULT_ODE_ACTION_IN_USE;
        }
        m_odeAreas.erase(name);

        LOG_INFO("ODE Area '" << name << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeAreaDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (auto const& imap: m_odeAreas)
        {
            // In the case of Delete all
            if (imap.second.use_count() > 1)
            {
                LOG_ERROR("ODE Area '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_ODE_ACTION_IN_USE;
            }
        }
        m_odeAreas.clear();

        LOG_INFO("All ODE Areas deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::OdeAreaListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeAreas.size();
    }
        
    DslReturnType Services::OdeTriggerOccurrenceNew(const char* name, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, classId, limit);
            
            LOG_INFO("New Occurrence ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Occurrence ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAbsenceNew(const char* name, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ABSENCE_NEW(name, classId, limit);
            
            LOG_INFO("New Absence ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Absence ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerIntersectionNew(const char* name, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_INTERSECTION_NEW(name, classId, limit);
            
            LOG_INFO("New Intersection ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Intersection ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerSummationNew(const char* name, uint classId, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_SUMMATION_NEW(name, classId, limit);
            
            LOG_INFO("New Summation ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Summation ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerCustomNew(const char* name, 
        uint classId, uint limit,  dsl_ode_check_for_occurrence_cb client_checker, void* client_data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_CLIENT_CALLBACK_INVALID;
            }
            
            if (!client_checker)
            {
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_CLIENT_CALLBACK_INVALID;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_CUSTOM_NEW(name, classId, limit, client_checker, client_data);
            
            LOG_INFO("New Custom ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Custon ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::OdeTriggerMinimumNew(const char* name, 
        uint classId, uint limit, uint minimum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_MINIMUM_NEW(name, classId, limit, minimum);
            
            LOG_INFO("New Minimum ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Minimum ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerMaximumNew(const char* name, 
        uint classId, uint limit, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_MAXIMUM_NEW(name, classId, limit, maximum);
            
            LOG_INFO("New Maximum ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Maximum ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerRangeNew(const char* name, 
        uint classId, uint limit, uint lower, uint upper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure event name uniqueness 
            if (m_odeTriggers.find(name) != m_odeTriggers.end())
            {   
                LOG_ERROR("ODE Trigger name '" << name << "' is not unique");
                return DSL_RESULT_ODE_TRIGGER_NAME_NOT_UNIQUE;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_RANGE_NEW(name, classId, limit, lower, upper);
            
            LOG_INFO("New Range ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Range ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerReset(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->Reset();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *enabled = pOdeTrigger->GetEnabled();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Enabled setting");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetEnabled(enabled);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting Enabled");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdGet(const char* name, uint* classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *classId = pOdeTrigger->GetClassId();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdSet(const char* name, uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetClassId(classId);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerSourceIdGet(const char* name, uint* sourceId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *sourceId = pOdeTrigger->GetSourceId();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting source id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerSourceIdSet(const char* name, uint sourceId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetSourceId(sourceId);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerConfidenceMinGet(const char* name, double* minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *minConfidence = pOdeTrigger->GetMinConfidence();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting minimum confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerConfidenceMinSet(const char* name, double minConfidence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetMinConfidence(minConfidence);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting minimum confidence");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMinGet(const char* name, uint* min_width, uint* min_height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetMinDimensions(min_width, min_height);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting minimum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMinSet(const char* name, uint min_width, uint min_height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            // TODO: validate the min values for in-range
            pOdeTrigger->SetMinDimensions(min_width, min_height);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting minimum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMaxGet(const char* name, uint* max_width, uint* max_height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetMaxDimensions(max_width, max_height);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting maximum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDimensionsMaxSet(const char* name, uint max_width, uint max_height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            // TODO: validate the max values for in-range
            pOdeTrigger->SetMaxDimensions(max_width, max_height);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting maximum dimensions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerFrameCountMinGet(const char* name, uint* min_count_n, uint* min_count_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetMinFrameCount(min_count_n, min_count_d);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting minimum frame count");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services:: OdeTriggerFrameCountMinSet(const char* name, uint min_count_n, uint min_count_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            // TODO: validate the min values for in-range
            pOdeTrigger->SetMinFrameCount(min_count_n, min_count_d);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting minimum frame count");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferDoneOnlyGet(const char* name, boolean* inferDoneOnly)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *inferDoneOnly = pOdeTrigger->GetInferDoneOnlySetting();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting source id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferDoneOnlySet(const char* name, boolean inferDoneOnly)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetInferDoneOnlySetting(inferDoneOnly);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerActionAdd(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

            // Note: Actions can be added when in use, i.e. shared between
            // multiple ODE Triggers

            if (!m_odeTriggers[name]->AddAction(m_odeActions[action]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to add ODE Action '" << action << "'");
                return DSL_RESULT_ODE_TRIGGER_ACTION_ADD_FAILED;
            }
            LOG_INFO("ODE Action '" << action
                << "' was added to ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception adding ODE Action '" << action << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerActionRemove(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

            if (!m_odeActions[action]->IsParent(m_odeTriggers[name]))
            {
                LOG_ERROR("ODE Action'" << action << 
                    "' is not in use by ODE Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_ACTION_NOT_IN_USE;
            }

            if (!m_odeTriggers[name]->RemoveAction(m_odeActions[action]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to remove ODE Action '" << action << "'");
                return DSL_RESULT_ODE_TRIGGER_ACTION_REMOVE_FAILED;
            }
            LOG_INFO("ODE Action '" << action
                << "' was removed from ODE Trigger '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception remove ODE Action '" << action << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerActionRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

            m_odeTriggers[name]->RemoveAllActions();

            LOG_INFO("All Events Actions removed from ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw an exception removing All Events Actions");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAreaAdd(const char* name, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, area);

            // Note: Areas can be added when in use, i.e. shared between
            // multiple ODE Triggers

            if (!m_odeTriggers[name]->AddArea(m_odeAreas[area]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to add ODE Area '" << area << "'");
                return DSL_RESULT_ODE_TRIGGER_AREA_ADD_FAILED;
            }
            LOG_INFO("ODE Area '" << area
                << "' was added to ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception adding ODE Area '" << area << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerAreaRemove(const char* name, const char* area)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, area);

            if (!m_odeAreas[area]->IsParent(m_odeTriggers[name]))
            {
                LOG_ERROR("ODE Area'" << area << 
                    "' is not in use by ODE Trigger '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_AREA_NOT_IN_USE;
            }

            if (!m_odeTriggers[name]->RemoveArea(m_odeAreas[area]))
            {
                LOG_ERROR("ODE Trigger '" << name
                    << "' failed to remove ODE Area '" << area << "'");
                return DSL_RESULT_ODE_TRIGGER_AREA_REMOVE_FAILED;
            }
            LOG_INFO("ODE Area '" << area
                << "' was removed from ODE Trigger '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name
                << "' threw exception remove ODE Area '" << area << "'");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerAreaRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

            m_odeTriggers[name]->RemoveAllAreas();

            LOG_INFO("All Events Areas removed from ODE Trigger '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw an exception removing All ODE Areas");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
        
        if (m_odeTriggers[name]->IsInUse())
        {
            LOG_INFO("ODE Trigger '" << name << "' is in use");
            return DSL_RESULT_ODE_TRIGGER_IN_USE;
        }
        m_odeTriggers.erase(name);

        LOG_INFO("ODE Trigger '" << name << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeTriggerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (auto const& imap: m_odeTriggers)
        {
            // In the case of Delete all
            if (imap.second->IsInUse())
            {
                LOG_ERROR("ODE Trigger '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_ODE_TRIGGER_IN_USE;
            }
        }
        m_odeTriggers.clear();

        LOG_INFO("All ODE Triggers deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::OdeTriggerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeTriggers.size();
    }
    
    DslReturnType Services::SourceCsiNew(const char* name,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
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
        LOG_INFO("New CSI Source '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SourceUsbNew(const char* name,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Source name '" << name << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_USB_SOURCE_NEW(name, width, height, fps_n, fps_d);
        }
        catch(...)
        {
            LOG_ERROR("New USB Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("New USB Source '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceUriNew(const char* name, const char* uri, 
        boolean isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Source name '" << name << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        std::string stringUri(uri);
        if (stringUri.find("http") == std::string::npos)
        {
            if (isLive)
            {
                LOG_ERROR("Invalid URI '" << uri << "' for Live source '" << name << "'");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
            std::ifstream streamUriFile(uri);
            if (!streamUriFile.good())
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                return DSL_RESULT_SOURCE_FILE_NOT_FOUND;
            }
        }
        try
        {
            m_components[name] = DSL_URI_SOURCE_NEW(
                name, uri, isLive, cudadecMemType, intraDecode, dropFrameInterval);
        }
        catch(...)
        {
            LOG_ERROR("New URI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("New URI Source '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SourceRtspNew(const char* name, const char* uri, 
        uint protocol, uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Source name '" << name << "' is not unique");
            return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_RTSP_SOURCE_NEW(
                name, uri, protocol, cudadecMemType, intraDecode, dropFrameInterval);
        }
        catch(...)
        {
            LOG_ERROR("New RTSP Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        LOG_INFO("New RTSP Source '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SourceDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
         
            pSourceBintr->GetDimensions(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                
    
    DslReturnType Services::SourceFrameRateGet(const char* name, uint* fps_n, uint* fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);
            
            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
         
            pSourceBintr->GetFrameRate(fps_n, fps_d);
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceDecodeUriGet(const char* name, const char** uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);

            *uri = pSourceBintr->GetUri();
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
            

    DslReturnType Services::SourceDecodeUriSet(const char* name, const char* uri)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);

            if (!pSourceBintr->SetUri(uri));
            {
                LOG_ERROR("Failed to Set URI '" << uri << "' for Decode Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED;
            }
            
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SourceDecodeDewarperAdd(const char* name, const char* dewarper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, dewarper);
            RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, dewarper, DewarperBintr);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);
         
            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[dewarper]);
         
            if (!pSourceBintr->AddDewarperBintr(pDewarperBintr))
            {
                LOG_ERROR("Failed to add Dewarper '" << dewarper << "' to Decode Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceDecodeDewarperRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, name);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[name]);
         
            if (!pSourceBintr->RemoveDewarperBintr())
            {
                LOG_ERROR("Failed to remove Dewarper from Decode Source '" << name << "'");
                return DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception removing Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourcePause(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
                
            if (!pSourceBintr->IsInUse())
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in use");
                return DSL_RESULT_SOURCE_NOT_IN_USE;
            }
            if (pSourceBintr->GetState() != GST_STATE_PLAYING)
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in play");
                return DSL_RESULT_SOURCE_NOT_IN_PLAY;
            }
            if (!pSourceBintr->SetState(GST_STATE_PAUSED))
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
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[name]);
                
            if (!pSourceBintr->IsInUse())
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not in use");
                return DSL_RESULT_SOURCE_NOT_IN_USE;
            }
            if (pSourceBintr->GetState() != GST_STATE_PAUSED)
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not in pause");
                return DSL_RESULT_SOURCE_NOT_IN_PAUSE;
            }

            if (!pSourceBintr->SetState(GST_STATE_PLAYING))
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
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, name);

            return std::dynamic_pointer_cast<SourceBintr>(m_components[name])->IsLive();
        }
        catch(...)
        {
            LOG_ERROR("Component '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    uint Services::SourceNumInUseGet()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        uint numInUse(0);
        
        for (auto const& imap: m_pipelines)
        {
            numInUse += imap.second->GetNumSourcesInUse();
        }
        return numInUse;
    }
    
    uint Services::SourceNumInUseMaxGet()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_sourceNumInUseMax;
    }
    
    boolean Services::SourceNumInUseMaxSet(uint max)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        uint numInUse(0);
        
        if (max < GetNumSourcesInUse())
        {
            LOG_ERROR("max setting = " << max << 
                " is less than the current number of Sources in use = " << numInUse);
            return false;
        }
        m_sourceNumInUseMax = max;
        return true;
    }

    DslReturnType Services::DewarperNew(const char* name, const char* configFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Dewarper name '" << name << "' is not unique");
            return DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE;
        }
        
        LOG_INFO("Dewarper config file: " << configFile);
        
        std::ifstream ifsConfigFile(configFile);
        if (!ifsConfigFile.good())
        {
            LOG_ERROR("Dewarper Config File not found");
            return DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND;
        }

        try
        {
            m_components[name] = DSL_DEWARPER_NEW(name, configFile);
        }
        catch(...)
        {
            LOG_ERROR("New Dewarper '" << name << "' threw exception on create");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
        LOG_INFO("New Dewarper '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PrimaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
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
        
        std::string testPath(modelEngineFile);
        if (testPath.size())
        {
            LOG_INFO("Model engine file: " << modelEngineFile);
            
            std::ifstream modelFile(modelEngineFile);
            if (!modelFile.good())
            {
                LOG_ERROR("Model Engine File not found");
                return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
            }
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
        LOG_INFO("New Primary GIE '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }


    DslReturnType Services::PrimaryGieBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for Primary GIE '" << name << "'");
            return DSL_RESULT_GIE_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, PrimaryGieBintr);
            
            DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
                std::dynamic_pointer_cast<PrimaryGieBintr>(m_components[name]);

            if (!pPrimaryGieBintr->AddBatchMetaHandler(pad, handler, userData))
            {
                LOG_ERROR("Primary GIE '" << name << "' failed to add a Batch Meta Handler");
                return DSL_RESULT_GIE_HANDLER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PrimaryGieBatchMetaHandlerRemove(const char* name, 
        uint pad, dsl_batch_meta_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for Primary GIE '" << name << "'");
            return DSL_RESULT_GIE_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, PrimaryGieBintr);
            
            DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
                std::dynamic_pointer_cast<PrimaryGieBintr>(m_components[name]);

            if (!pPrimaryGieBintr->RemoveBatchMetaHandler(pad, handler))
            {
                LOG_ERROR("Primary GIE '" << name << "' has no matching Batch Meta Handler");
                return DSL_RESULT_GIE_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType  Services::PrimaryGieKittiOutputEnabledSet(const char* name, boolean enabled,
        const char* file)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, PrimaryGieBintr);
            
            DSL_PRIMARY_GIE_PTR pPrimaryGieBintr = 
                std::dynamic_pointer_cast<PrimaryGieBintr>(m_components[name]);

            if (!pPrimaryGieBintr->SetKittiOutputEnabled(enabled, file))
            {
                LOG_ERROR("Invalid Kitti file path " << file << "for Primary GIE '" << name << "'");
                return DSL_RESULT_GIE_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception setting Kitti output enabled");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::SecondaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, const char* inferOnGieName, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
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
                inferConfigFile, modelEngineFile, inferOnGieName, interval);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        LOG_INFO("New Secondary GIE '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieRawOutputEnabledSet(const char* name, boolean enabled,
        const char* path)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);
                
            if (!pGieBintr->SetRawOutputEnabled(enabled, path))
            {
                LOG_ERROR("GIE '" << name << "' failed to enable raw output");
                return DSL_RESULT_GIE_OUTPUT_DIR_DOES_NOT_EXIST;
            }
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on raw output enabled set");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieInferConfigFileGet(const char* name, const char** inferConfigFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);

            *inferConfigFile = pGieBintr->GetInferConfigFile();
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieInferConfigFileSet(const char* name, const char* inferConfigFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);

            if (!pGieBintr->SetInferConfigFile(inferConfigFile))
            {
                LOG_ERROR("GIE '" << name << "' failed to set the Infer Config file");
                return DSL_RESULT_GIE_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieModelEngineFileGet(const char* name, const char** inferConfigFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);

            *inferConfigFile = pGieBintr->GetModelEngineFile();
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieModelEngineFileSet(const char* name, const char* inferConfigFile)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);

            if (!pGieBintr->SetModelEngineFile(inferConfigFile))
            {
                LOG_ERROR("GIE '" << name << "' failed to set the Infer Config file");
                return DSL_RESULT_GIE_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);

            *interval = pGieBintr->GetInterval();
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::GieIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_GIE(m_components, name);
            
            DSL_GIE_PTR pGieBintr = 
                std::dynamic_pointer_cast<GieBintr>(m_components[name]);

            if (!pGieBintr->SetInterval(interval))
            {
                LOG_ERROR("GIE '" << name << "' failed to set new Interval");
                return DSL_RESULT_GIE_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw an exception setting Interval");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TrackerKtlNew(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("KTL Tracker name '" << name << "' is not unique");
            return DSL_RESULT_TRACKER_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new KtlTrackerBintr(
                name, width, height));
        }
        catch(...)
        {
            LOG_ERROR("KTL Tracker '" << name << "' threw exception on create");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
        LOG_INFO("New KTL Tracker '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::TrackerIouNew(const char* name, const char* configFile, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("IOU Tracker name '" << name << "' is not unique");
            return DSL_RESULT_TRACKER_NAME_NOT_UNIQUE;
        }
        LOG_INFO("Infer config file: " << configFile);
        
        std::ifstream streamConfigFile(configFile);
        if (!streamConfigFile.good())
        {
            LOG_ERROR("Infer Config File not found");
            return DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new IouTrackerBintr(
                name, configFile, width, height));
        }
        catch(...)
        {
            LOG_ERROR("IOU Tracker '" << name << "' threw exception on create");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
        LOG_INFO("New IOU Tracker '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
   
       DslReturnType Services::TrackerMaxDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            // TODO verify args before calling
            trackerBintr->GetMaxDimensions(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TrackerMaxDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set Max Dimensions for Tracker '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_TILER_IS_IN_USE;
            }

            DSL_TRACKER_PTR trackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            // TODO verify args before calling
            if (!trackerBintr->SetMaxDimensions(width, height))
            {
                LOG_ERROR("Tracker '" << name << "' failed to set dimensions");
                return DSL_RESULT_TRACKER_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TrackerBatchMetaHandlerAdd(const char* name, uint pad, 
        dsl_batch_meta_handler_cb handler, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for Tracker '" << name << "'");
            return DSL_RESULT_TRACKER_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);

            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!pTrackerBintr->AddBatchMetaHandler(pad, handler, userData))
            {
                LOG_ERROR("Tracker '" << name << "' failed to add Batch Meta Handler");
                return DSL_RESULT_TRACKER_HANDLER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TrackerBatchMetaHandlerRemove(const char* name, 
        uint pad, dsl_batch_meta_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for Tracker '" << name << "'");
            return DSL_RESULT_TRACKER_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);

            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!pTrackerBintr->RemoveBatchMetaHandler(pad, handler))
            {
                LOG_ERROR("Tracker '" << name << "' has no matching Batch Meta Handler");
                return DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
   
    DslReturnType  Services::TrackerKittiOutputEnabledSet(const char* name, boolean enabled,
        const char* file)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);
            
            DSL_TRACKER_PTR pTrackerBintr = 
                std::dynamic_pointer_cast<TrackerBintr>(m_components[name]);

            if (!pTrackerBintr->SetKittiOutputEnabled(enabled, file))
            {
                LOG_ERROR("Invalid Kitti file path " << file << "for Tracker '" << name << "'");
                return DSL_RESULT_TRACKER_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception setting Kitti output enabled");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::TeeDemuxerNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Demuxer Tee name '" << name << "' is not unique");
            return DSL_RESULT_TEE_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new DemuxerBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New Demuxer Tee '" << name << "' threw exception on create");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        LOG_INFO("New Demuxer Tee '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TeeSplitterNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Splitter Tee name '" << name << "' is not unique");
            return DSL_RESULT_TILER_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new SplitterBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New Splitter Tee '" << name << "' threw exception on create");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        LOG_INFO("New Splitter Tee '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::TeeBranchAdd(const char* tee, 
        const char* branch)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tee);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, branch);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, tee);
            RETURN_IF_COMPONENT_IS_NOT_BRANCH(m_components, branch);
            
            // Can't add components if they're In use by another Branch
            if (m_components[branch]->IsInUse())
            {
                LOG_ERROR("Unable to add branch '" << branch 
                    << "' as it's currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[tee]);

            // Cast the Branch to a Bintr to call the correct AddChile method.
            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);

            if (!pTeeBintr->AddChild(pBranchBintr))
            {
                LOG_ERROR("Tee '" << tee << 
                    "' failed to add branch '" << branch << "'");
                return DSL_RESULT_TEE_BRANCH_ADD_FAILED;
            }

        }
        catch(...)
        {
            LOG_ERROR("Tee '" << tee 
                << "' threw an exception removing branch '" << branch << "'");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        LOG_INFO("Branch '" << branch 
            << "' was added to Tee '" << tee << "' successfully");
        return DSL_RESULT_SUCCESS;
    }    
    
    DslReturnType Services::TeeBranchRemove(const char* tee, 
        const char* branch)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tee);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, tee);
            RETURN_IF_BRANCH_NAME_NOT_FOUND(m_components, branch);

            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[tee]);

            if (!pTeeBintr->IsChild(m_components[branch]))
            {
                LOG_ERROR("Branch '" << branch << 
                    "' is not in use by Tee '" << tee << "'");
                return DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD;
            }

            // Cast the Branch to a Bintr to call the correct AddChile method.
            DSL_BINTR_PTR pBranchBintr = 
                std::dynamic_pointer_cast<Bintr>(m_components[branch]);

            if (!pTeeBintr->RemoveChild(pBranchBintr))
            {
                LOG_ERROR("Tee '" << tee << 
                    "' failed to remove branch '" << branch << "'");
                return DSL_RESULT_TEE_BRANCH_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tee '" << tee 
                << "' threw an exception removing branch '" << branch << "'");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TeeBranchRemoveAll(const char* tee)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tee);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, tee);

            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[tee]);
//            m_components[tee]->RemoveAll();
        }
        catch(...)
        {
            LOG_ERROR("Tee '" <<  tee
                << "' threw an exception removing all branches");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TeeBranchCountGet(const char* tee, uint* count)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tee);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, tee);

            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[tee]);

            *count = pTeeBintr->GetNumChildren();
        }
        catch(...)
        {
            LOG_ERROR("Tee '" <<  tee
                << "' threw an exception getting branch count");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TeeBatchMetaHandlerAdd(const char* name, 
        dsl_batch_meta_handler_cb handler, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, name);

            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[name]);

            if (!pTeeBintr->AddBatchMetaHandler(DSL_PAD_SINK, handler, userData))
            {
                LOG_ERROR("Tee '" << name << "' failed to add Batch Meta Handler");
                return DSL_RESULT_TILER_HANDLER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tee '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TeeBatchMetaHandlerRemove(const char* name, 
        dsl_batch_meta_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, name);

            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[name]);

            if (!pTeeBintr->RemoveBatchMetaHandler(DSL_PAD_SINK, handler))
            {
                LOG_ERROR("Tee '" << name << "' has no matching Batch Meta Handler");
                return DSL_RESULT_TEE_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tee '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::TilerNew(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Tiler name '" << name << "' is not unique");
            return DSL_RESULT_TILER_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = std::shared_ptr<Bintr>(new TilerBintr(
                name, width, height));
        }
        catch(...)
        {
            LOG_ERROR("New Tiler'" << name << "' threw exception on create");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        LOG_INFO("New Tiler '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            tilerBintr->GetDimensions(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);


            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            if (!tilerBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Tiler '" << name << "' failed to settin dimensions");
                return DSL_RESULT_TILER_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerTilesGet(const char* name, uint* cols, uint* rows)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            tilerBintr->GetTiles(cols, rows);
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerTilesSet(const char* name, uint cols, uint rows)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            // TODO verify args before calling
            if (!tilerBintr->SetTiles(cols, rows))
            {
                LOG_ERROR("Tiler '" << name << "' failed to set Tiles");
                return DSL_RESULT_TILER_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerBatchMetaHandlerAdd(const char* name, uint pad, 
        dsl_batch_meta_handler_cb handler, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for Tiler '" << name << "'");
            return DSL_RESULT_TILER_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->AddBatchMetaHandler(pad, handler, userData))
            {
                LOG_ERROR("Tiler '" << name << "' failed to add Batch Meta Handler");
                return DSL_RESULT_TILER_HANDLER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerBatchMetaHandlerRemove(const char* name, 
        uint pad, dsl_batch_meta_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for Tiler '" << name << "'");
            return DSL_RESULT_TILER_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->RemoveBatchMetaHandler(pad, handler))
            {
                LOG_ERROR("Tiler '" << name << "' has no matching Batch Meta Handler");
                return DSL_RESULT_TILER_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
   
    DslReturnType Services::OdeHandlerNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("ODE Handler name '" << name << "' is not unique");
            return DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE;
        }
        try
        {   
            m_components[name] = std::shared_ptr<Bintr>(new OdeHandlerBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New ODE Handler '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION;
        }
        LOG_INFO("New OdeHandler '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
   DslReturnType Services::OdeHandlerEnabledGet(const char* handler, boolean* enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, handler);

        try
        {
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, handler, OdeHandlerBintr);

            DSL_ODE_HANDLER_PTR pOdeHandlerBintr = 
                std::dynamic_pointer_cast<OdeHandlerBintr>(m_components[handler]);

            *enabled = pOdeHandlerBintr->GetEnabled();
        }
        catch(...)
        {
            LOG_ERROR("OdeHandler '" << handler
                << "' threw exception getting the Enabled state");
            return DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

   DslReturnType Services::OdeHandlerEnabledSet(const char* handler, boolean enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, handler);

        try
        {
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, handler, OdeHandlerBintr);

            DSL_ODE_HANDLER_PTR pOdeHandlerBintr = 
                std::dynamic_pointer_cast<OdeHandlerBintr>(m_components[handler]);

            if (!pOdeHandlerBintr->SetEnabled(enabled))
            {
                LOG_ERROR("ODE Handler '" << handler
                    << "' failed to set enabled state");
                return DSL_RESULT_ODE_HANDLER_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Handler '" << handler
                << "' threw exception setting the Enabled state");
            return DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

   DslReturnType Services::OdeHandlerTriggerAdd(const char* handler, const char* trigger)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, handler);
        RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

        try
        {
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, handler, OdeHandlerBintr);

            // Can't add Events if they're In use by another OdeHandler
            if (m_odeTriggers[trigger]->IsInUse())
            {
                LOG_ERROR("Unable to add ODE Trigger '" << trigger 
                    << "' as it is currently in use");
                return DSL_RESULT_ODE_TRIGGER_IN_USE;
            }

            DSL_ODE_HANDLER_PTR pOdeHandlerBintr = 
                std::dynamic_pointer_cast<OdeHandlerBintr>(m_components[handler]);

            if (!pOdeHandlerBintr->AddChild(m_odeTriggers[trigger]))
            {
                LOG_ERROR("ODE Handler '" << handler
                    << "' failed to add ODE Trigger '" << trigger << "'");
                return DSL_RESULT_ODE_HANDLER_TRIGGER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Handler '" << handler
                << "' threw exception adding ODE Trigger '" << trigger << "'");
            return DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION;
        }
        LOG_INFO("ODE Trigger '" << trigger 
            << "' was added to ODE Handler '" << handler << "' successfully");

        return DSL_RESULT_SUCCESS;
    }


    DslReturnType Services::OdeHandlerTriggerRemove(const char* handler, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, handler);
        RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

        try
        {
            if (!m_odeTriggers[trigger]->IsParent(m_components[handler]))
            {
                LOG_ERROR("ODE Trigger '" << trigger << 
                    "' is not in use by ODE Handler '" << handler << "'");
                return DSL_RESULT_ODE_HANDLER_TRIGGER_NOT_IN_USE;
            }
            
            DSL_ODE_HANDLER_PTR pOdeHandlerBintr = 
                std::dynamic_pointer_cast<OdeHandlerBintr>(m_components[handler]);
                
            if (!pOdeHandlerBintr->RemoveChild(m_odeTriggers[trigger]))
            {
                LOG_ERROR("ODE Handler '" << handler
                    << "' failed to remove ODE Trigger '" << trigger << "'");
                return DSL_RESULT_ODE_HANDLER_TRIGGER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Handler '" << handler 
                << "' threw an exception removing ODE Trigger");
            return DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION;
        }
        LOG_INFO("ODE Trigger '" << trigger 
            << "' was removed from OdeHandler '" << handler << "' successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeHandlerTriggerRemoveAll(const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, handler);

        try
        {
            DSL_ODE_HANDLER_PTR pOdeHandlerBintr = 
                std::dynamic_pointer_cast<OdeHandlerBintr>(m_components[handler]);

            pOdeHandlerBintr->RemoveAllChildren();
        }
        catch(...)
        {
            LOG_ERROR("ODE Handler '" << handler 
                << "' threw an exception removing All ODE Triggers");
            return DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION;
        }
        LOG_INFO("All ODE Triggers removed from ODE Handler '" << handler << "' successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OfvNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("OFV name '" << name << "' is not unique");
            return DSL_RESULT_OFV_NAME_NOT_UNIQUE;
        }
        try
        {   
            m_components[name] = std::shared_ptr<Bintr>(new OfvBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New OFV '" << name << "' threw exception on create");
            return DSL_RESULT_OFV_THREW_EXCEPTION;
        }
        LOG_INFO("New OFV '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdNew(const char* name, boolean isClockEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
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
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        LOG_INFO("New OSD '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdClockEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetClockEnabled(enabled);
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set The clock enabled setting for the OSD '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_OSD_IS_IN_USE;
            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockOffsetsGet(const char* name, uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetClockOffsets(offsetX, offsetY);
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockOffsetsSet(const char* name, uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set The clock offsets for the OSD '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_OSD_IS_IN_USE;
            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockOffsets(offsetX, offsetY))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock offsets");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockFontGet(const char* name, const char** font, uint* size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetClockFont(font, size);
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockFontSet(const char* name, const char* font, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set The clock offsets for the OSD '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_OSD_IS_IN_USE;
            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockFont(font, size))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock font");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockColorGet(const char* name, double* red, double* green, double* blue, double* alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetClockColor(red, green, blue, alpha);
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockColorSet(const char* name, double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set The clock RGB colors for the OSD '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_OSD_IS_IN_USE;
            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockColor(red, green, blue, alpha))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock RGB colors");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdCropSettingsGet(const char* name, uint* left, uint* top, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetCropSettings(left, top, width, height);
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting crop settings");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdCropSettingsSet(const char* name, uint left, uint top, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetCropSettings(left, top, width, height))
            {
                LOG_ERROR("OSD '" << name << "' failed to set crop settings");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting crop settings");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdBatchMetaHandlerAdd(const char* name, uint pad, 
        dsl_batch_meta_handler_cb handler, void* userData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for OSD '" << name << "'");
            return DSL_RESULT_OSD_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->AddBatchMetaHandler(pad, handler, userData))
            {
                LOG_ERROR("OSD '" << name << "' already has a Batch Meta Handler");
                return DSL_RESULT_OSD_HANDLER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdBatchMetaHandlerRemove(const char* name, 
        uint pad, dsl_batch_meta_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (pad > DSL_PAD_SRC)
        {
            LOG_ERROR("Invalid Pad type = " << pad << " for OSD '" << name << "'");
            return DSL_RESULT_OSD_PAD_TYPE_INVALID;
        }
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->RemoveBatchMetaHandler(pad, handler))
            {
                LOG_ERROR("OSD '" << name << "' has no Batch Meta Handler");
                return DSL_RESULT_OSD_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
   
    DslReturnType Services::SinkFakeNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_FAKE_SINK_NEW(name);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("New Fake Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkOverlayNew(const char* name, uint overlay_id, uint display_id,
        uint depth, uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_OVERLAY_SINK_NEW(
                name, overlay_id, display_id, depth, offsetX, offsetY, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("New Overlay Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkWindowNew(const char* name, 
        uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        try
        {
            m_components[name] = DSL_WINDOW_SINK_NEW(name, offsetX, offsetY, width, height);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("New Window Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkFileNew(const char* name, const char* filepath, 
            uint codec, uint container, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        if (codec > DSL_CODEC_MPEG4)
        {   
            LOG_ERROR("Invalid Codec value = " << codec << " for File Sink '" << name << "'");
            return DSL_RESULT_SINK_CODEC_VALUE_INVALID;
        }
        if (container > DSL_CONTAINER_MKV)
        {   
            LOG_ERROR("Invalid Container value = " << container << " for File Sink '" << name << "'");
            return DSL_RESULT_SINK_CONTAINER_VALUE_INVALID;
        }
        try
        {
            m_components[name] = DSL_FILE_SINK_NEW(name, filepath, codec, container, bitrate, interval);
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("New File Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkFileVideoFormatsGet(const char* name, uint* codec, uint* container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, FileSinkBintr);

            DSL_FILE_SINK_PTR fileSinkBintr = 
                std::dynamic_pointer_cast<FileSinkBintr>(m_components[name]);

            fileSinkBintr->GetVideoFormats(codec, container);
        }
        catch(...)
        {
            LOG_ERROR("File Sink '" << name << "' threw an exception getting Video formats");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkFileEncoderSettingsGet(const char* name, uint* bitrate, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, FileSinkBintr);

            DSL_FILE_SINK_PTR fileSinkBintr = 
                std::dynamic_pointer_cast<FileSinkBintr>(m_components[name]);

            fileSinkBintr->GetEncoderSettings(bitrate, interval);
        }
        catch(...)
        {
            LOG_ERROR("File Sink '" << name << "' threw an exception getting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkFileEncoderSettingsSet(const char* name, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, FileSinkBintr);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set Encoder settings for File Sink '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_SINK_IS_IN_USE;
            }

            DSL_FILE_SINK_PTR fileSinkBintr = 
                std::dynamic_pointer_cast<FileSinkBintr>(m_components[name]);

            if (!fileSinkBintr->SetEncoderSettings(bitrate, interval))
            {
                LOG_ERROR("File Sink '" << name << "' failed to set Encoder settings");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("File Sink'" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRecordNew(const char* name, const char* outdir, 
            uint container, dsl_sink_record_client_listner_cb clientListener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            struct stat info;

            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Sink name '" << name << "' is not unique");
                return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
            }
            // ensure outdir exists
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir << "' for Record Sink '" << name << "'");
                return DSL_RESULT_SINK_FILE_PATH_NOT_FOUND;
            }
            m_components[name] = DSL_RECORD_SINK_NEW(name, outdir, container,
                (NvDsSRCallbackFunc)clientListener);
            
            LOG_INFO("New Record Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Image Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordSessionStart(const char* name, 
        uint* session, uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            if (!recordSinkBintr->StartSession(session, start, duration, clientData))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to Start Session");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Sink'" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordSessionStop(const char* name, uint session)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            if (!recordSinkBintr->StopSession(session))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to Stop Session");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Sink'" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordCacheSizeGet(const char* name, uint* cacheSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            // TODO verify args before calling
            *cacheSize = recordSinkBintr->GetCacheSize();
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRecordCacheSizeSet(const char* name, uint cacheSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            // TODO verify args before calling
            if (!recordSinkBintr->SetCacheSize(cacheSize))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to get dimensions");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::SinkRecordDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            // TODO verify args before calling
            recordSinkBintr->GetDimensions(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRecordDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);


            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            // TODO verify args before calling
            if (!recordSinkBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to set dimensions");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRtspNew(const char* name, const char* host, 
            uint udpPort, uint rtspPort, uint codec, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        if (codec > DSL_CODEC_H265)
        {   
            LOG_ERROR("Invalid Codec value = " << codec << " for File Sink '" << name << "'");
            return DSL_RESULT_SINK_CODEC_VALUE_INVALID;
        }
        try
        {
            m_components[name] = DSL_RTSP_SINK_NEW(name, host, udpPort, rtspPort, codec, bitrate, interval);
        }
        catch(...)
        {
            LOG_ERROR("New RTSP Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("New RTSP Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkRtspServerSettingsGet(const char* name, uint* udpPort, uint* rtspPort, uint* codec)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);

        try
        {
            DSL_RTSP_SINK_PTR rtspSinkBintr = 
                std::dynamic_pointer_cast<RtspSinkBintr>(m_components[name]);

            rtspSinkBintr->GetServerSettings(udpPort, rtspPort, codec);
        }
        catch(...)
        {
            LOG_ERROR("RTSP Sink '" << name << "' threw an exception getting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRtspEncoderSettingsGet(const char* name, uint* bitrate, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSinkBintr);

            DSL_RTSP_SINK_PTR rtspSinkBintr = 
                std::dynamic_pointer_cast<RtspSinkBintr>(m_components[name]);

            rtspSinkBintr->GetEncoderSettings(bitrate, interval);
        }
        catch(...)
        {
            LOG_ERROR("RTSP Sink '" << name << "' threw an exception getting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRtspEncoderSettingsSet(const char* name, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSinkBintr);

            DSL_RTSP_SINK_PTR rtspSinkBintr = 
                std::dynamic_pointer_cast<RtspSinkBintr>(m_components[name]);

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set Encoder settings for RTSP Sink '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_SINK_IS_IN_USE;
            }

            if (!rtspSinkBintr->SetEncoderSettings(bitrate, interval))
            {
                LOG_ERROR("RTSP Sink '" << name << "' failed to set Encoder settings");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("RTSP Sink '" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageNew(const char* name, const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        struct stat info;

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Sink name '" << name << "' is not unique");
            return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
        }
        // ensure outdir exists
        if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
        {
            LOG_ERROR("Unable to access outdir '" << outdir << "' for Image Sink '" << name << "'");
            return DSL_RESULT_SINK_FILE_PATH_NOT_FOUND;
        }
        try
        {
            m_components[name] = DSL_IMAGE_SINK_NEW(name, outdir);
        }
        catch(...)
        {
            LOG_ERROR("New Image Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        LOG_INFO("New Image Sink '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkImageOutdirGet(const char* name, const char** outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            *outdir = sinkBintr->GetOutdir();
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw exception on Outdir get");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageOutdirSet(const char* name, const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            // ensure outdir exists
            struct stat info;
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir << "' for Image Sink '" << name << "'");
                return DSL_RESULT_SINK_FILE_PATH_NOT_FOUND;
            }
            
            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            if (!sinkBintr->SetOutdir(outdir))
            {
                LOG_ERROR("Failed to set outdir '" << outdir << "' for Image Sink '" << name << "'");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw exception on Outdir set");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }


    DslReturnType Services::SinkImageFrameCaptureIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            *interval = sinkBintr->GetFrameCaptureInterval();
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception getting Frame Capture interval");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageFrameCaptureIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            if (!sinkBintr->SetFrameCaptureInterval(interval))
            {
                LOG_ERROR("Image Sink '" << name << "' failed to set Frame Capture interval");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception setting Frame Capture interval");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageFrameCaptureEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            *enabled = sinkBintr->GetFrameCaptureEnabled();
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception getting Frame Capture enabled");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageFrameCaptureEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            if (!sinkBintr->SetFrameCaptureEnabled(enabled))
            {
                LOG_ERROR("Image Sink '" << name << "' failed to set Frame Capture enabled");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception setting Frame Capture enabled");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageObjectCaptureEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            *enabled = sinkBintr->GetObjectCaptureEnabled();
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception getting Object Capture enabled");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageObjectCaptureEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            if (!sinkBintr->SetObjectCaptureEnabled(enabled))
            {
                LOG_ERROR("Image Sink '" << name << "' failed to set Object Capture enabled");
                return DSL_RESULT_SINK_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception setting Object Capture enabled");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkImageObjectCaptureClassAdd(const char* name, 
        uint classId, boolean fullFrame, uint captureLimit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            if (!sinkBintr->AddObjectCaptureClass(classId, fullFrame, captureLimit))
            {
                LOG_ERROR("Image Sink '" << name << "' failed to add Object Capture Class");
                return DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Image Sink '" << name << "' threw an exception adding Object Capture Class");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SinkImageObjectCaptureClassRemove(const char* name, uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, ImageSinkBintr);

            DSL_IMAGE_SINK_PTR sinkBintr = 
                std::dynamic_pointer_cast<ImageSinkBintr>(m_components[name]);

            if (!sinkBintr->RemoveObjectCaptureClass(classId))
            {
                LOG_ERROR("Image Sink '" << name << "' failed to remove Object Capture Class");
                return DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception removing Redaction Class");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    uint Services::SinkNumInUseGet()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        return GetNumSinksInUse();
    }
    
    uint Services::SinkNumInUseMaxGet()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_sinkNumInUseMax;
    }
    
    boolean Services::SinkNumInUseMaxSet(uint max)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        uint numInUse(0);
        
        if (max < GetNumSinksInUse())
        {
            LOG_ERROR("max setting = " << max << 
                " is less than the current number of Sinks in use = " << numInUse);
            return false;
        }
        m_sinkNumInUseMax = max;
        return true;
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
    
    DslReturnType Services::ComponentDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

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

    DslReturnType Services::ComponentGpuIdGet(const char* component, uint* gpuid)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
        
        if (m_components[component]->IsInUse())
        {
            LOG_INFO("Component '" << component << "' is in use");
            return DSL_RESULT_COMPONENT_IN_USE;
        }
        *gpuid = m_components[component]->GetGpuId();

        LOG_INFO("Current GPU ID = " << *gpuid << " for component '" << component << "'");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ComponentGpuIdSet(const char* component, uint gpuid)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
        
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
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, branch);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

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
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, branch);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

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
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        m_pipelines[pipeline]->RemoveAllChildren();
        m_pipelines.erase(pipeline);

        LOG_INFO("Pipeline '" << pipeline << "' deleted successfully");

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
    
    DslReturnType Services::PipelineComponentAdd(const char* pipeline, 
        const char* component)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
        
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
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline
                << "' threw exception adding component '" << component << "'");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        LOG_INFO("Component '" << component 
            << "' was added to Pipeline '" << pipeline << "' successfully");

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
    
    DslReturnType Services::PipelineStreamMuxBatchPropertiesGet(const char* pipeline,
        uint* batchSize, uint* batchTimeout)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            m_pipelines[pipeline]->GetStreamMuxBatchProperties(batchSize, batchTimeout);
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
                return DSL_RESULT_PIPELINE_STREAMMUX_SET_FAILED;
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
        
    DslReturnType Services::PipelineXWindowClear(const char* pipeline)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->ClearXWindow())
            {
                LOG_ERROR("Pipeline '" << pipeline << "' failed to Clear XWindow");
                return DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline << "' threw an exception clearing XWindow");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineXWindowDimensionsGet(const char* pipeline,
        uint* width, uint* height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            m_pipelines[pipeline]->GetXWindowDimensions(width, height);
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception getting the XWindow Dimensions");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineXWindowDimensionsSet(const char* pipeline,
        uint width, uint height)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->SetXWindowDimensions(width, height))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to Set the XWindow Dimensions");
                return DSL_RESULT_PIPELINE_XWINDOW_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception setting the XWindow dimensions");
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
    
    DslReturnType Services::PipelineStateGet(const char* pipeline, uint* state)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            *state = std::dynamic_pointer_cast<PipelineBintr>(m_pipelines[pipeline])->GetState();
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
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        // TODO check state of debug env var and return NON-success if not set

        m_pipelines[pipeline]->DumpToDot(filename);
        
        return DSL_RESULT_SUCCESS;
    }   
    
    DslReturnType Services::PipelineDumpToDotWithTs(const char* pipeline, char* filename)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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

        try
        {
            if (!m_pipelines[pipeline]->AddStateChangeListener(listener, userdata))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineStateChangeListenerRemove(const char* pipeline, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
    
        try
        {
            if (!m_pipelines[pipeline]->RemoveStateChangeListener(listener))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineEosListenerAdd(const char* pipeline, 
        dsl_eos_listener_cb listener, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        try
        {
            if (!m_pipelines[pipeline]->AddEosListener(listener, userdata))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add a EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding a EOS Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::PipelineEosListenerRemove(const char* pipeline, 
        dsl_eos_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
    
        try
        {
            if (!m_pipelines[pipeline]->RemoveEosListener(listener))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing a State Change Lister");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::PipelineXWindowKeyEventHandlerAdd(const char* pipeline, 
        dsl_xwindow_key_event_handler_cb handler, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        try
        {
            if (!m_pipelines[pipeline]->AddXWindowKeyEventHandler(handler, userdata))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add XWindow Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding XWindow Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineXWindowKeyEventHandlerRemove(const char* pipeline, 
        dsl_xwindow_key_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        try
        {
            if (!m_pipelines[pipeline]->RemoveXWindowKeyEventHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove XWindow Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing XWindow Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineXWindowButtonEventHandlerAdd(const char* pipeline, 
        dsl_xwindow_button_event_handler_cb handler, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        try
        {
            if (!m_pipelines[pipeline]->AddXWindowButtonEventHandler(handler, userdata))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add XWindow Button Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding XWindow Button Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineXWindowButtonEventHandlerRemove(const char* pipeline, 
        dsl_xwindow_button_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        try
        {
            if (!m_pipelines[pipeline]->RemoveXWindowButtonEventHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove XWindow Button Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing XWindow Button Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineXWindowDeleteEventHandlerAdd(const char* pipeline, 
        dsl_xwindow_delete_event_handler_cb handler, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        try
        {
            if (!m_pipelines[pipeline]->AddXWindowDeleteEventHandler(handler, userdata))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add XWindow Delete Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception adding XWindow Delete Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::PipelineXWindowDeleteEventHandlerRemove(const char* pipeline, 
        dsl_xwindow_delete_event_handler_cb handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
        
        try
        {
            if (!m_pipelines[pipeline]->RemoveXWindowDeleteEventHandler(handler))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove XWindow Delete Event Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline 
                << "' threw an exception removing XWindow Delete Event Handler");
            return DSL_RESULT_PIPELINE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

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
        
        if (m_stateValueToString.find(state) == m_returnValueToString.end())
        {
            LOG_ERROR("Invalid state = " << state << " unable to convert to string");
            return m_stateValueToString[DSL_STATE_INVALID_STATE_VALUE].c_str();
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
        m_stateValueToString[DSL_STATE_IN_TRANSITION] = L"DSL_STATE_IN_TRANSITION";
        m_stateValueToString[DSL_STATE_INVALID_STATE_VALUE] = L"Invalid DSL_STATE Value";

        m_returnValueToString[DSL_RESULT_SUCCESS] = L"DSL_RESULT_SUCCESS";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE] = L"DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_NOT_FOUND] = L"DSL_RESULT_COMPONENT_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_BAD_FORMAT] = L"DSL_RESULT_COMPONENT_NAME_BAD_FORMAT";
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
        m_returnValueToString[DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE] = L"DSL_RESULT_SOURCE_COMPONENT_IS_NOT_SOURCE";
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
        m_returnValueToString[DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID] = L"DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID";
        m_returnValueToString[DSL_RESULT_TRACKER_IS_IN_USE] = L"DSL_RESULT_TRACKER_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_TRACKER_SET_FAILED] = L"DSL_RESULT_TRACKER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_TRACKER_HANDLER_ADD_FAILED] = L"DSL_RESULT_TRACKER_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_TRACKER_PAD_TYPE_INVALID] = L"DSL_RESULT_TRACKER_PAD_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER] = L"DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_HANDLER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_HANDLER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_NAME_BAD_FORMAT] = L"DSL_RESULT_ODE_HANDLER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION] = L"DSL_RESULT_ODE_HANDLER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_IS_IN_USE] = L"DSL_RESULT_ODE_HANDLER_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_SET_FAILED] = L"DSL_RESULT_ODE_HANDLER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_TRIGGER_ADD_FAILED] = L"DSL_RESULT_ODE_HANDLER_TRIGGER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_TRIGGER_REMOVE_FAILED] = L"DSL_RESULT_ODE_HANDLER_TRIGGER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_TRIGGER_NOT_IN_USE] = L"DSL_RESULT_ODE_HANDLER_TRIGGER_NOT_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_HANDLER_COMPONENT_IS_NOT_ODE_HANDLER] = L"DSL_RESULT_ODE_HANDLER_COMPONENT_IS_NOT_ODE_HANDLER";
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
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_ACTION_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_ACTION_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_THREW_EXCEPTION] = L"DSL_RESULT_ODE_ACTION_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_IN_USE] = L"DSL_RESULT_ODE_ACTION_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_SET_FAILED] = L"DSL_RESULT_ODE_ACTION_SET_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_IS_NOT_ACTION] = L"DSL_RESULT_ODE_ACTION_IS_NOT_ACTION";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND] = L"DSL_RESULT_ODE_ACTION_FILE_PATH_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID] = L"DSL_RESULT_ODE_ACTION_CAPTURE_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE";
        m_returnValueToString[DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_AREA_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_AREA_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_AREA_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_AREA_THREW_EXCEPTION] = L"DSL_RESULT_ODE_AREA_THREW_EXCEPTION";
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
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED";
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
        m_returnValueToString[DSL_RESULT_GIE_NAME_NOT_UNIQUE] = L"DSL_RESULT_GIE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_GIE_NAME_NOT_FOUND] = L"DSL_RESULT_GIE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_GIE_NAME_BAD_FORMAT] = L"DSL_RESULT_GIE_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_GIE_CONFIG_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND] = L"DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_GIE_THREW_EXCEPTION] = L"DSL_RESULT_GIE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_GIE_IS_IN_USE] = L"DSL_RESULT_GIE_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_GIE_SET_FAILED] = L"DSL_RESULT_GIE_SET_FAILED";
        m_returnValueToString[DSL_RESULT_GIE_HANDLER_ADD_FAILED] = L"DSL_RESULT_GIE_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_GIE_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_GIE_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_GIE_PAD_TYPE_INVALID] = L"DSL_RESULT_GIE_PAD_TYPE_INVALID";
        m_returnValueToString[DSL_RESULT_GIE_COMPONENT_IS_NOT_GIE] = L"DSL_RESULT_GIE_COMPONENT_IS_NOT_GIE";
        m_returnValueToString[DSL_RESULT_GIE_OUTPUT_DIR_DOES_NOT_EXIST] = L"DSL_RESULT_GIE_OUTPUT_DIR_DOES_NOT_EXIST";
        m_returnValueToString[DSL_RESULT_TEE_NAME_NOT_UNIQUE] = L"DSL_RESULT_TEE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_TEE_NAME_NOT_FOUND] = L"DSL_RESULT_TEE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TEE_NAME_BAD_FORMAT] = L"DSL_RESULT_TEE_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_TEE_THREW_EXCEPTION] = L"DSL_RESULT_TEE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD] = L"DSL_RESULT_TEE_BRANCH_IS_NOT_CHILD";
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
        m_returnValueToString[DSL_RESULT_INVALID_RESULT_CODE] = L"Invalid DSL Result CODE";
    }

} // namespace 