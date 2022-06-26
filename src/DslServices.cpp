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

void geosNoticeHandler(const char *fmt, ...)
{
    // TODO
}

void geosErrorHandler(const char *fmt, ...)
{
    // TODO
}

// Single GST debug catagory initialization
GST_DEBUG_CATEGORY(GST_CAT_DSL);

GQuark _dsmeta_quark;

namespace DSL
{
    // Initialize the Services's single instance pointer
    Services* Services::m_pInstance = NULL;

    Services* Services::GetServices()
    {
        // one time initialization of the single instance pointer
        if (!m_pInstance)
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
                
            }
            
            // Safe to start logging
            LOG_INFO("Services Initialization");
            
            _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
            
            // Single instantiation for the lib's lifetime
            m_pInstance = new Services(doGstDeinit);
            
            // initialization of GEOS
            initGEOS(geosNoticeHandler, geosErrorHandler);
            
            // Initialize private containers
            m_pInstance->InitToStringMaps();
            
            // Create the default Display types
            m_pInstance->DisplayTypeCreateIntrinsicTypes();

            std::wstring wVersion(DSL_VERSION);
            std::string cVersion(wVersion.begin(), wVersion.end());
            LOG_INFO("DSL Version: " << cVersion);

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
        return m_pInstance;
    }
        
    Services::Services(bool doGstDeinit)
        : m_doGstDeinit(doGstDeinit)
        , m_debugLogFileHandle(NULL)
        , m_pMainLoop(g_main_loop_new(NULL, FALSE))
        , m_sourceNumInUseMax(DSL_DEFAULT_SOURCE_IN_USE_MAX)
        , m_sinkNumInUseMax(DSL_DEFAULT_SINK_IN_USE_MAX)
    {
        LOG_FUNC();

        if (InfoInitDebugSettings() != DSL_RESULT_SUCCESS)
        {
            LOG_ERROR("DSL threw exception intializing Debug Settings");
            throw;
        }
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
            
            InfoDeinitDebugSettings();
            
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
    
    void Services::DeleteAll()
    {
        LOG_FUNC();
        // DO NOT lock mutex - will be done by each service
        
        PipelineDeleteAll();
        PlayerDeleteAll(false);
        ComponentDeleteAll();
        PphDeleteAll();
        OdeTriggerDeleteAll();
        OdeAccumulatorDeleteAll();
        OdeAreaDeleteAll();
        OdeActionDeleteAll();
        DisplayTypeDeleteAll();
        MailerDeleteAll();
        MessageBrokerDeleteAll();
    }
   
    // ------------------------------------------------------------------------------
    
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
        m_returnValueToString[DSL_RESULT_API_NOT_IMPLEMENTED] = L"DSL_RESULT_API_NOT_IMPLEMENTED";
        m_returnValueToString[DSL_RESULT_API_NOT_SUPPORTED] = L"DSL_RESULT_API_NOT_SUPPORTED";
        m_returnValueToString[DSL_RESULT_API_NOT_ENABLED] = L"DSL_RESULT_API_NOT_ENABLED";
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
        m_returnValueToString[DSL_RESULT_COMPONENT_SET_NVBUF_MEM_TYPE_FAILED] = L"DSL_RESULT_COMPONENT_SET_NVBUF_MEM_TYPE_FAILED";

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
        m_returnValueToString[DSL_RESULT_SOURCE_COMPONENT_IS_NOT_DECODE_SOURCE] = L"DSL_RESULT_SOURCE_COMPONENT_IS_NOT_DECODE_SOURCE";
        m_returnValueToString[DSL_RESULT_SOURCE_COMPONENT_IS_NOT_FILE_SOURCE] = L"DSL_RESULT_SOURCE_COMPONENT_IS_NOT_FILE_SOURCE";
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
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_CALLBACK_ADD_FAILED] = L"DSL_RESULT_ODE_TRIGGER_CALLBACK_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_CALLBACK_REMOVE_FAILED] = L"DSL_RESULT_ODE_TRIGGER_CALLBACK_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID] = L"DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE] = L"DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_IS_NOT_TRACK_TRIGGER] = L"DSL_RESULT_ODE_TRIGGER_IS_NOT_TRACK_TRIGGER";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_ADD_FAILED] = L"DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_REMOVE_FAILED] = L"DSL_RESULT_ODE_TRIGGER_ACCUMULATOR_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_ADD_FAILED] = L"DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_REMOVE_FAILED] = L"DSL_RESULT_ODE_TRIGGER_HEAT_MAPPER_REMOVE_FAILED";

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

        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_ACCUMULATOR_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION] = L"DSL_RESULT_ODE_ACCUMULATOR_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_IN_USE] = L"DSL_RESULT_ODE_ACCUMULATOR_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_SET_FAILED] = L"DSL_RESULT_ODE_ACCUMULATOR_SET_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_IS_NOT_ODE_ACCUMULATOR] = L"DSL_RESULT_ODE_ACCUMULATOR_IS_NOT_ODE_ACCUMULATOR";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_ACTION_ADD_FAILED] = L"DSL_RESULT_ODE_ACCUMULATOR_ACTION_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_ACTION_REMOVE_FAILED] = L"DSL_RESULT_ODE_ACCUMULATOR_ACTION_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_ACCUMULATOR_ACTION_NOT_IN_USE] = L"DSL_RESULT_ODE_ACCUMULATOR_ACTION_NOT_IN_USE";

        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_UNIQUE] = L"DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_FOUND] = L"DSL_RESULT_ODE_HEAT_MAPPER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION] = L"DSL_RESULT_ODE_HEAT_MAPPER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_IN_USE] = L"DSL_RESULT_ODE_HEAT_MAPPER_IN_USE";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_SET_FAILED] = L"DSL_RESULT_ODE_HEAT_MAPPER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_IS_NOT_ODE_HEAT_MAPPER] = L"DSL_RESULT_ODE_HEAT_MAPPER_IS_NOT_ODE_HEAT_MAPPER";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_ACTION_ADD_FAILED] = L"DSL_RESULT_ODE_HEAT_MAPPER_ACTION_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_ACTION_REMOVE_FAILED] = L"DSL_RESULT_ODE_HEAT_MAPPER_ACTION_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_ODE_HEAT_MAPPER_ACTION_NOT_IN_USE] = L"DSL_RESULT_ODE_HEAT_MAPPER_ACTION_NOT_IN_USE";

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
        m_returnValueToString[DSL_RESULT_SINK_COMPONENT_IS_NOT_MESSAGE_SINK] = L"DSL_RESULT_SINK_COMPONENT_IS_NOT_MESSAGE_SINK";
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_HANDLER_ADD_FAILED] = L"DSL_RESULT_SINK_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_SINK_HANDLER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_PLAYER_ADD_FAILED] = L"DSL_RESULT_SINK_PLAYER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_PLAYER_REMOVE_FAILED] = L"DSL_RESULT_SINK_PLAYER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_MAILER_ADD_FAILED] = L"DSL_RESULT_SINK_MAILER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_MAILER_REMOVE_FAILED] = L"DSL_RESULT_SINK_MAILER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_ADD_FAILED] = L"DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_REMOVE_FAILED] = L"DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_OVERLAY_NOT_SUPPORTED] = L"DSL_RESULT_SINK_OVERLAY_NOT_SUPPORTED";
        m_returnValueToString[DSL_RESULT_SINK_WEBRTC_CONNECTION_CLOSED_FAILED] = L"DSL_RESULT_SINK_WEBRTC_CONNECTION_CLOSED_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_SINK_MESSAGE_CONFIG_FILE_NOT_FOUND";

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
        m_returnValueToString[DSL_RESULT_INFER_ID_NOT_FOUND] = L"DSL_RESULT_INFER_ID_NOT_FOUND";

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

        m_returnValueToString[DSL_RESULT_BRANCH_NAME_NOT_UNIQUE] = L"DSL_RESULT_BRANCH_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_BRANCH_NAME_NOT_FOUND] = L"DSL_RESULT_BRANCH_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_BRANCH_NAME_BAD_FORMAT] = L"DSL_RESULT_BRANCH_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_BRANCH_THREW_EXCEPTION] = L"DSL_RESULT_BRANCH_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED] = L"DSL_RESULT_BRANCH_COMPONENT_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED] = L"DSL_RESULT_BRANCH_COMPONENT_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_BRANCH_SOURCE_NOT_ALLOWED] = L"DSL_RESULT_BRANCH_SOURCE_NOT_ALLOWED";
        m_returnValueToString[DSL_RESULT_BRANCH_SINK_MAX_IN_USE_REACHED] = L"DSL_RESULT_BRANCH_SINK_MAX_IN_USE_REACHED";

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
        m_returnValueToString[DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED] = L"DSL_RESULT_PIPELINE_MAIN_LOOP_REQUEST_FAILED";

        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION] = L"DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_IN_USE] = L"DSL_RESULT_DISPLAY_TYPE_IN_USE";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE] = L"DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND] = L"DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE";
        m_returnValueToString[DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE] = L"DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE";
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

        m_returnValueToString[DSL_RESULT_WEBSOCKET_SERVER_THREW_EXCEPTION] = L"DSL_RESULT_WEBSOCKET_SERVER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_WEBSOCKET_SERVER_SET_FAILED] = L"DSL_RESULT_WEBSOCKET_SERVER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_ADD_FAILED] = L"DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_REMOVE_FAILED] = L"DSL_RESULT_WEBSOCKET_SERVER_CLIENT_LISTENER_REMOVE_FAILED";

        m_returnValueToString[DSL_RESULT_BROKER_NAME_NOT_UNIQUE] = L"DSL_RESULT_BROKER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_BROKER_NAME_NOT_FOUND] = L"DSL_RESULT_BROKER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_BROKER_THREW_EXCEPTION] = L"DSL_RESULT_BROKER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_BROKER_IN_USE] = L"DSL_RESULT_BROKER_IN_USE";
        m_returnValueToString[DSL_RESULT_BROKER_SET_FAILED] = L"DSL_RESULT_BROKER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_PARAMETER_INVALID] = L"DSL_RESULT_BROKER_PARAMETER_INVALID";
        m_returnValueToString[DSL_RESULT_BROKER_SUBSCRIBER_ADD_FAILED] = L"DSL_RESULT_BROKER_SUBSCRIBER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_SUBSCRIBER_REMOVE_FAILED] = L"DSL_RESULT_BROKER_SUBSCRIBER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_LISTENER_ADD_FAILED] = L"DSL_RESULT_BROKER_LISTENER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_LISTENER_REMOVE_FAILED] = L"DSL_RESULT_BROKER_LISTENER_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_BROKER_CONFIG_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_BROKER_PROTOCOL_LIB_NOT_FOUND] = L"DSL_RESULT_BROKER_PROTOCOL_LIB_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_BROKER_CONNECT_FAILED] = L"DSL_RESULT_BROKER_CONNECT_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_DISCONNECT_FAILED] = L"DSL_RESULT_BROKER_DISCONNECT_FAILED";
        m_returnValueToString[DSL_RESULT_BROKER_MESSAGE_SEND_FAILED] = L"DSL_RESULT_BROKER_MESSAGE_SEND_FAILED";

        m_returnValueToString[DSL_RESULT_INVALID_RESULT_CODE] = L"Invalid DSL Result CODE";
   }

} // namespace
 