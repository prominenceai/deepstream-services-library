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
#include "Dsl.h"
#include "DslApi.h"
#include "DslOdeTrigger.h"
#include "DslServices.h"
#include "DslSourceBintr.h"
#include "DslGieBintr.h"
#include "DslTrackerBintr.h"
#include "DslPadProbeHandler.h"
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

#define RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(components, name) do \
{ \
    if (!components[name]->IsType(typeid(DistanceOdeTrigger)) and  \
        !components[name]->IsType(typeid(IntersectionOdeTrigger))) \
    { \
        LOG_ERROR("Component '" << name << "' is not an AB ODE Trigger"); \
        return DSL_RESULT_ODE_TRIGGER_IS_NOT_AB_TYPE; \
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

#define RETURN_IF_COMPONENT_IS_NOT_ENCODE_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Decode Source"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_ENCODE_SINK; \
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

// All Bintr's that can be added as a "branch" to a "Tee"
#define RETURN_IF_COMPONENT_IS_NOT_BRANCH(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(OverlaySinkBintr)) and  \
        !components[name]->IsType(typeid(WindowSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtspSinkBintr)) and \
        !components[name]->IsType(typeid(BranchBintr)) and \
        !components[name]->IsType(typeid(DemuxerBintr)) and \
        !components[name]->IsType(typeid(BranchBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Branch type"); \
        return DSL_RESULT_TEE_BRANCH_IS_NOT_BRANCH; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_SINK(components, name) do \
{ \
    if (!components[name]->IsType(typeid(FakeSinkBintr)) and  \
        !components[name]->IsType(typeid(OverlaySinkBintr)) and  \
        !components[name]->IsType(typeid(WindowSinkBintr)) and  \
        !components[name]->IsType(typeid(FileSinkBintr)) and  \
        !components[name]->IsType(typeid(RecordSinkBintr)) and  \
        !components[name]->IsType(typeid(RtspSinkBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Sink"); \
        return DSL_RESULT_SINK_COMPONENT_IS_NOT_SINK; \
    } \
}while(0); 

#define RETURN_IF_COMPONENT_IS_NOT_TAP(components, name) do \
{ \
    if (!components[name]->IsType(typeid(RecordTapBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Tap"); \
        return DSL_RESULT_TAP_COMPONENT_IS_NOT_TAP; \
    } \
}while(0); 


#define RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(types, name) do \
{ \
    if (types.find(name) == types.end()) \
    { \
        LOG_ERROR("Display Type '" << name << "' was not found"); \
        return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_FOUND; \
    } \
}while(0); 

#define RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(types, name, displayType) do \
{ \
    if (!types[name]->IsType(typeid(displayType))) \
    { \
        LOG_ERROR("Display Type '" << name << "' is not the correct type"); \
        return DSL_RESULT_DISPLAY_TYPE_NOT_THE_CORRECT_TYPE; \
    } \
}while(0); 

#define RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(types, name) do \
{ \
    if (types[name]->IsType(typeid(RgbaColor)) or \
        types[name]->IsType(typeid(RgbaFont))) \
    { \
        LOG_ERROR("Display Type '" << name << "' is base type and can not be displayed"); \
        return DSL_RESULT_DISPLAY_TYPE_IS_BASE_TYPE; \
    } \
}while(0); 

#define RETURN_IF_PPH_NAME_NOT_FOUND(handlers, name) do \
{ \
    if (handlers.find(name) == handlers.end()) \
    { \
        LOG_ERROR("Pad Probe Handler name '" << name << "' was not found"); \
        return DSL_RESULT_PPH_NAME_NOT_FOUND; \
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
        , m_pComms(std::unique_ptr<Comms>(new Comms()))
    {
        LOG_FUNC();

        g_mutex_init(&m_servicesMutex);
    }

    Services::~Services()
    {
        LOG_FUNC();
        
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
            
            finishGEOS();
            
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
    
    DslReturnType Services::DisplayTypeRgbaColorNew(const char* name, 
        double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Color name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_COLOR_NAME_NOT_UNIQUE;
            }
            m_displayTypes[name] = DSL_RGBA_COLOR_NEW(name, 
                red, green, blue, alpha);

            LOG_INFO("New RGBA Color '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Color '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaFontNew(const char* name, const char* font,
        uint size, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Font name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_FONT_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_FONT_NEW(name, font, size, pColor);

            LOG_INFO("New RGBA Color '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Color '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaTextNew(const char* name, const char* text, 
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Text name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_TEXT_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_RGBA_TEXT_NEW(name,
                text, xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New RGBA Text '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Text '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaLineNew(const char* name, 
        uint x1, uint y1, uint x2, uint y2, uint width, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Line name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_LINE_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_LINE_NEW(name, x1, y1, x2, y2, width, pColor);

            LOG_INFO("New RGBA Line '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Line '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaArrowNew(const char* name, 
        uint x1, uint y1, uint x2, uint y2, uint width, uint head, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Arrow name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_ARROW_NAME_NOT_UNIQUE;
            }

            if (head > DSL_ARROW_BOTH_HEAD)
            {
                LOG_ERROR("RGBA Head Type Invalid for RGBA Arrow'" << name << "'");
                return DSL_RESULT_DISPLAY_RGBA_ARROW_HEAD_INVALID;
            }
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);
            
            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_ARROW_NEW(name, x1, y1, x2, y2, width, head, pColor);

            LOG_INFO("New RGBA Arrow '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Arrow '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaRectangleNew(const char* name, uint left, uint top, 
        uint width, uint height, uint borderWidth, const char* color, bool hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Rectangle name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_RECTANGLE_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }
            
            m_displayTypes[name] = DSL_RGBA_RECTANGLE_NEW(name, 
                left, top, width, height, borderWidth, pColor, hasBgColor, pBgColor);

            LOG_INFO("New RGBA Rectangle '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Rectangle '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DisplayTypeRgbaPolygonNew(const char* name, 
        const dsl_coordinate* coordinates, uint numCoordinates, 
        uint borderWidth, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Polygon name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_POLYGON_NAME_NOT_UNIQUE;
            }
            if (numCoordinates > DSL_MAX_POLYGON_COORDINATES)
            {
                LOG_ERROR("Max coordinates exceeded created RGBA Polygon name '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_POLYGON_NEW(name, 
                coordinates, numCoordinates, borderWidth, pColor);

            LOG_INFO("New RGBA Rectangle '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Rectangle '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
    

    DslReturnType Services::DisplayTypeRgbaCircleNew(const char* name, uint xCenter, uint yCenter, uint radius,
        const char* color, bool hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Rectangle name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_RGBA_CIRCLE_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }
            
            m_displayTypes[name] = DSL_RGBA_CIRCLE_NEW(name, 
                xCenter, yCenter, radius, pColor, hasBgColor, pBgColor);

            LOG_INFO("New RGBA Circle '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Circle '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DisplayTypeSourceNumberNew(const char* name,
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Number name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_SOURCE_NUMBER_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_SOURCE_NUMBER_NEW(name,
                xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New Source Number '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New New Source Number '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeSourceNameNew(const char* name,
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Name name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_SOURCE_NAME_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_SOURCE_NAME_NEW(name,
                xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New Source Name '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Name '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeSourceDimensionsNew(const char* name, 
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Dimensions name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_SOURCE_DIMENSIONS_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_SOURCE_DIMENSIONS_NEW(name,
                xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New Source Dimensions '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Dimensions '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeSourceFrameRateNew(const char* name, 
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Frame-Rate name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_SOURCE_FRAMERATE_NAME_NOT_UNIQUE;
            }
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_SOURCE_FRAME_RATE_NEW(name,
                xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New Source Frame-Rate '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Frame-Rate '" << name << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeMetaAdd(const char* name, void* pDisplayMeta, void* pFrameMeta)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            
            DSL_DISPLAY_TYPE_PTR pDisplayType = 
                std::dynamic_pointer_cast<DisplayType>(m_displayTypes[name]);

            pDisplayType->AddMeta((NvDsDisplayMeta*)pDisplayMeta, (NvDsFrameMeta*)pFrameMeta);
            
            LOG_INFO("Display Type '" << name << "' deleted successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Display Type '" << name << "' threw exception on delete");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::DisplayTypeDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            
            if (m_displayTypes[name].use_count() > 1)
            {
                LOG_INFO("Display Type '" << name << "' is in use");
                return DSL_RESULT_DISPLAY_TYPE_IN_USE;
            }
            m_displayTypes.erase(name);
            
            LOG_INFO("Display Type '" << name << "' deleted successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Display Type '" << name << "' threw exception on delete");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::DisplayTypeDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // Don't check for in-use on deleting all. 
            m_displayTypes.clear();
            
            LOG_INFO("All Display Types deleted successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Display Types threw exception on delete all");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    uint Services::DisplayTypeListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_displayTypes.size();
    }
            
    DslReturnType Services::OdeActionCustomNew(const char* name,
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
            m_odeActions[name] = DSL_ODE_ACTION_CUSTOM_NEW(name, clientHandler, clientData);

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
        const char* outdir, boolean annotate)
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
            m_odeActions[name] = DSL_ODE_ACTION_CAPTURE_FRAME_NEW(name, outdir, annotate);

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
    
    DslReturnType Services::OdeActionDisplayNew(const char* name, uint offsetX, uint offsetY, 
        boolean offsetYWithClassId, const char* font, boolean hasBgColor, const char* bgColor)
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
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);
            
            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);

            m_odeActions[name] = DSL_ODE_ACTION_DISPLAY_NEW(name, 
                offsetX, offsetY, offsetYWithClassId, pFont, hasBgColor, pBgColor);
            LOG_INFO("New Display ODE Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Display ODE Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionEmailNew(const char* name, const char* subject)
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
            
            m_odeActions[name] = DSL_ODE_ACTION_EMAIL_NEW(name, subject);

            LOG_INFO("New ODE Email Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Email Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillSurroundingsNew(const char* name, const char* color)
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
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
                
            m_odeActions[name] = DSL_ODE_ACTION_FILL_SURROUNDINGS_NEW(name, pColor);

            LOG_INFO("New ODE Fill Surroundings Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Surroundings Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillFrameNew(const char* name, const char* color)
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
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
                
            m_odeActions[name] = DSL_ODE_ACTION_FILL_FRAME_NEW(name, pColor);

            LOG_INFO("New ODE Fill Frame Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Fill Frame Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFillObjectNew(const char* name, const char* color)
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

            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
                
            m_odeActions[name] = DSL_ODE_ACTION_FILL_OBJECT_NEW(name, pColor);

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
    
    DslReturnType Services::OdeActionDisplayMetaAddNew(const char* name, const char* displayType)
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
            
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, displayType);
            RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(m_displayTypes, displayType);
            
            DSL_DISPLAY_TYPE_PTR pDisplayType = std::dynamic_pointer_cast<DisplayType>(m_displayTypes[displayType]);
            
            m_odeActions[name] = DSL_ODE_ACTION_DISPLAY_META_ADD_NEW(name, pDisplayType);

            LOG_INFO("New Add Display Meta Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Add Display Meta Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionDisplayMetaAddDisplayType(const char* name, const char* displayType)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(m_odeActions, name, AddDisplayMetaOdeAction);
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, displayType);
            RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(m_displayTypes, displayType);
            
            DSL_DISPLAY_TYPE_PTR pDisplayType = std::dynamic_pointer_cast<DisplayType>(m_displayTypes[displayType]);
            
            DSL_ODE_ACTION_DISPLAY_META_ADD_PTR pAction = 
                std::dynamic_pointer_cast<AddDisplayMetaOdeAction>(m_odeActions[name]);

            pAction->AddDisplayType(pDisplayType);
            
            LOG_INFO("Display Type '" << displayType << "' added to Action '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Overlay Frame Action '" << name << "' threw exception on create");
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
    
    DslReturnType Services::OdeActionRedactNew(const char* name)
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

    DslReturnType Services::OdeActionSinkRecordStartNew(const char* name,
        const char* recordSink, uint start, uint duration, void* clientData)
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
            m_odeActions[name] = DSL_ODE_ACTION_SINK_RECORD_START_NEW(name,
                recordSink, start, duration, clientData);

            LOG_INFO("New ODE Record Sink Start Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Start Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionSinkRecordStopNew(const char* name,
        const char* recordSink)
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
            m_odeActions[name] = DSL_ODE_ACTION_SINK_RECORD_STOP_NEW(name,
                recordSink);

            LOG_INFO("New ODE Record Sink Stop Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Stop Action '" << name << "' threw exception on create");
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

    DslReturnType Services::OdeActionTapRecordStartNew(const char* name,
        const char* recordTap, uint start, uint duration, void* clientData)
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
            m_odeActions[name] = DSL_ODE_ACTION_TAP_RECORD_START_NEW(name,
                recordTap, start, duration, clientData);

            LOG_INFO("New ODE Record Tap Start Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Tap Start Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionTapRecordStopNew(const char* name,
        const char* recordTap)
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
            m_odeActions[name] = DSL_ODE_ACTION_TAP_RECORD_STOP_NEW(name, recordTap);

            LOG_INFO("New ODE Record Tap Stop Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Record Tap Stop Action '" << name << "' threw exception on create");
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

    DslReturnType Services::OdeActionTilerShowSourceNew(const char* name, 
        const char* tiler, uint timeout, bool hasPrecedence)
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
            m_odeActions[name] = DSL_ODE_ACTION_TILER_SHOW_SOURCE_NEW(name, tiler, timeout, hasPrecedence);

            LOG_INFO("New Tiler Show Source ODE Action'" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Tiler Show Source ODE Action '" << name << "' threw exception on create");
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

        try
        {
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
        catch(...)
        {
            LOG_ERROR("ODE Action '" << name << "' threw exception on deletion");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeActionDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
        catch(...)
        {
            LOG_ERROR("ODE Action threw exception on delete all");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    uint Services::OdeActionListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeActions.size();
    }
    
    DslReturnType Services::OdeAreaInclusionNew(const char* name, 
        const char* polygon, boolean show, uint bboxTestPoint)
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
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, polygon);
            
            // Interim ... only supporting rectangles at this
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, polygon, RgbaPolygon);
            
            if (bboxTestPoint > DSL_BBOX_POINT_ANY)
            {
                LOG_ERROR("Bounding box test point value of '" << bboxTestPoint << 
                    "' is invalid when creating ODE Inclusion Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_PARAMETER_INVALID;
            }
            
            DSL_RGBA_POLYGON_PTR pPolygon = 
                std::dynamic_pointer_cast<RgbaPolygon>(m_displayTypes[polygon]);
            
            m_odeAreas[name] = DSL_ODE_AREA_INCLUSION_NEW(name, 
                pPolygon, show, bboxTestPoint);
         
            LOG_INFO("New ODE Inclusion Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Inclusion Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeAreaExclusionNew(const char* name, 
        const char* polygon, boolean show, uint bboxTestPoint)
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
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, polygon);
            
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, polygon, RgbaPolygon);

            if (bboxTestPoint > DSL_BBOX_POINT_ANY)
            {
                LOG_ERROR("Bounding box test point value of '" << bboxTestPoint << 
                    "' is invalid when creating ODE Exclusion Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_PARAMETER_INVALID;
            }

            DSL_RGBA_POLYGON_PTR pPolygon = 
                std::dynamic_pointer_cast<RgbaPolygon>(m_displayTypes[polygon]);
            
            m_odeAreas[name] = DSL_ODE_AREA_EXCLUSION_NEW(name, 
                pPolygon, show, bboxTestPoint);
         
            LOG_INFO("New ODE Exclusion Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Exclusion Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeAreaLineNew(const char* name, 
        const char* line, boolean show, uint bboxTestEdge)
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
            RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, line);
            RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, line, RgbaLine);
            
            if (bboxTestEdge > DSL_BBOX_EDGE_RIGHT)
            {
                LOG_ERROR("Bounding box test edge value of '" << bboxTestEdge << 
                    "' is invalid when creating ODE Line Area '" << name << "'");
                return DSL_RESULT_ODE_AREA_PARAMETER_INVALID;
            }
            
            DSL_RGBA_LINE_PTR pLine = 
                std::dynamic_pointer_cast<RgbaLine>(m_displayTypes[line]);
            
            m_odeAreas[name] = DSL_ODE_AREA_LINE_NEW(name, pLine, show, bboxTestEdge);
         
            LOG_INFO("New ODE Line Area '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Line Area '" << name << "' threw exception on creation");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeAreaDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
        catch(...)
        {
            LOG_ERROR("ODE Area '" << name << "' threw exception on deletion");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeAreaDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            for (auto const& imap: m_odeAreas)
            {
                // In the case of Delete all
                if (imap.second.use_count() > 1)
                {
                    LOG_ERROR("ODE Area '" << imap.second->GetName() << "' is currently in use");
                    return DSL_RESULT_ODE_AREA_IN_USE;
                }
            }
            m_odeAreas.clear();

            LOG_INFO("All ODE Areas deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Area threw exception on delete all");
            return DSL_RESULT_ODE_AREA_THREW_EXCEPTION;
        }
    }

    uint Services::OdeAreaListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeAreas.size();
    }
        
    DslReturnType Services::OdeTriggerAlwaysNew(const char* name, const char* source, uint when)
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
            if (when > DSL_ODE_POST_OCCURRENCE_CHECK)
            {   
                LOG_ERROR("Invalid 'when' parameter for ODE Trigger name '" << name << "'");
                return DSL_RESULT_ODE_TRIGGER_PARAMETER_INVALID;
            }
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ALWAYS_NEW(name, source, when);
            
            LOG_INFO("New Always ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Always ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerOccurrenceNew(const char* name, const char* source, uint classId, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, source, classId, limit);
            
            LOG_INFO("New Occurrence ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Occurrence ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAbsenceNew(const char* name, const char* source, uint classId, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ABSENCE_NEW(name, source, classId, limit);
            
            LOG_INFO("New Absence ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Absence ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerInstanceNew(const char* name, const char* source, uint classId, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_INSTANCE_NEW(name, source, classId, limit);
            
            LOG_INFO("New Instance ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Instance ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerIntersectionNew(const char* name, 
        const char* source, uint classIdA, uint classIdB, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_INTERSECTION_NEW(name, 
                source, classIdA, classIdB, limit);
            
            LOG_INFO("New Intersection ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Intersection ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerSummationNew(const char* name, const char* source, uint classId, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_SUMMATION_NEW(name, source, classId, limit);
            
            LOG_INFO("New Summation ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Summation ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerCustomNew(const char* name, const char* source, 
        uint classId, uint limit,  dsl_ode_check_for_occurrence_cb client_checker, 
        dsl_ode_post_process_frame_cb client_post_processor, void* client_data)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_CUSTOM_NEW(name, source,
                classId, limit, client_checker, client_post_processor, client_data);
            
            LOG_INFO("New Custom ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Custon ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
            
    DslReturnType Services::OdeTriggerPersistenceNew(const char* name, const char* source, 
        uint classId, uint limit, uint minimum, uint maximum)
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
            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;

            m_odeTriggers[name] = DSL_ODE_TRIGGER_PERSISTENCE_NEW(name, 
                source, classId, limit, minimum, maximum);
            
            LOG_INFO("New Persistence ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Persistence ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerCountNew(const char* name, const char* source, 
        uint classId, uint limit, uint minimum, uint maximum)
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
            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
            
            m_odeTriggers[name] = DSL_ODE_TRIGGER_COUNT_NEW(name, 
                source, classId, limit, minimum, maximum);
            
            LOG_INFO("New Count ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Count ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerDistanceNew(const char* name, const char* source, 
        uint classIdA, uint classIdB, uint limit, uint minimum, uint maximum, 
        uint testPoint, uint testMethod)
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
            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
            
            m_odeTriggers[name] = DSL_ODE_TRIGGER_DISTANCE_NEW(name, 
                source, classIdA, classIdB, limit, minimum, maximum, 
                testPoint, testMethod);
            
            LOG_INFO("New Distance ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Distance ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
            
    
    DslReturnType Services::OdeTriggerSmallestNew(const char* name, 
        const char* source, uint classId, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_SMALLEST_NEW(name, source, classId, limit);
            
            LOG_INFO("New Smallest ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Smallest ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerLargestNew(const char* name, 
        const char* source, uint classId, uint limit)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_LARGEST_NEW(name, source, classId, limit);
            
            LOG_INFO("New Largest ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Largest ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeTriggerNewHighNew(const char* name, 
        const char* source, uint classId, uint limit, uint preset)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_NEW_HIGH_NEW(name, 
                source, classId, limit, preset);
            
            LOG_INFO("New New-High ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New New-High ODE Trigger '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerNewLowNew(const char* name, 
        const char* source, uint classId, uint limit, uint preset)
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_NEW_LOW_NEW(name, 
                source, classId, limit, preset);
            
            LOG_INFO("New New-Low ODE Trigger '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New New-Low ODE Trigger '" << name << "' threw exception on create");
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

    DslReturnType Services::OdeTriggerSourceGet(const char* name, const char** source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *source = pOdeTrigger->GetSource();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting source id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerSourceSet(const char* name, const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetSource(source);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
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

    DslReturnType Services::OdeTriggerClassIdABGet(const char* name, 
        uint* classIdA, uint* classIdB)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_AB_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<ABOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetClassIdAB(classIdA, classIdB);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerClassIdABSet(const char* name, 
        uint classIdA, uint classIdB)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_AB_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<ABOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetClassIdAB(classIdA, classIdB);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting class id");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    DslReturnType Services::OdeTriggerLimitGet(const char* name, uint* limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *limit = pOdeTrigger->GetLimit();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting limit");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerLimitSet(const char* name, uint limit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetLimit(limit);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting limit");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    DslReturnType Services::OdeTriggerConfidenceMinGet(const char* name, float* minConfidence)
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

    DslReturnType Services::OdeTriggerConfidenceMinSet(const char* name, float minConfidence)
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

    DslReturnType Services::OdeTriggerDimensionsMinGet(const char* name, float* min_width, float* min_height)
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

    DslReturnType Services::OdeTriggerDimensionsMinSet(const char* name, float min_width, float min_height)
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

    DslReturnType Services::OdeTriggerDimensionsMaxGet(const char* name, float* max_width, float* max_height)
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

    DslReturnType Services::OdeTriggerDimensionsMaxSet(const char* name, float max_width, float max_height)
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

        try
        {
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
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw an exception on deletion");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
        catch(...)
        {
            LOG_ERROR("ODE Trigger threw an exception on delete all");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }

    uint Services::OdeTriggerListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_odeTriggers.size();
    }

    DslReturnType Services::PphCustomNew(const char* name,
        dsl_pph_custom_client_handler_cb clientHandler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("Custom Pad Probe Handler name '" << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPH_CUSTOM_NEW(name, clientHandler, clientData);

            LOG_INFO("New Custom Pad Probe Handler '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Custom Pad Prove handler '" << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphMeterNew(const char* name, uint interval, 
        dsl_pph_meter_client_handler_cb clientHandler, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("Meter Pad Probe Handler name '" << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            if (!interval)
            {
                LOG_ERROR("Meter Pad Probe Handler '" << name << "' failed to set property, interval must be greater than 0");
                return DSL_RESULT_PPH_METER_INVALID_INTERVAL;
            }
            m_padProbeHandlers[name] = DSL_PPH_METER_NEW(name, 
                interval, clientHandler, clientData);

            LOG_INFO("New Meter Pad Probe Handler '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Meter Pad Prove handler '" << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    

    DslReturnType Services::PphMeterIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, MeterPadProbeHandler);

            DSL_PPH_METER_PTR pMeter = 
                std::dynamic_pointer_cast<MeterPadProbeHandler>(m_padProbeHandlers[name]);

            *interval = pMeter->GetInterval();

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Meter Sink '" << name << "' threw an exception getting reporting interval");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphMeterIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, MeterPadProbeHandler);
            
            if (!interval)
            {
                LOG_ERROR("Meter Pad Probe Handler '" << name << "' failed to set property, interval must be greater than 0");
                return DSL_RESULT_PPH_METER_INVALID_INTERVAL;
            }

            DSL_PPH_METER_PTR pMeter = 
                std::dynamic_pointer_cast<MeterPadProbeHandler>(m_padProbeHandlers[name]);

            if (!pMeter->SetInterval(interval))
            {
                LOG_ERROR("Meter Pad Probe Handler '" << name << "' failed to set reporting interval");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Meter Pad Probe Handler '" << name << "' threw an exception setting reporting interval");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PphOdeNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure handler name uniqueness 
            if (m_padProbeHandlers.find(name) != m_padProbeHandlers.end())
            {   
                LOG_ERROR("ODE Pad Probe Handler name '" << name << "' is not unique");
                return DSL_RESULT_PPH_NAME_NOT_UNIQUE;
            }
            m_padProbeHandlers[name] = DSL_PPH_ODE_NEW(name);
            
            LOG_INFO("New ODE Pad Probe Handler '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Pad Probe Handler '" << name << "' threw exception on create");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphOdeTriggerAdd(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

            // Can't add Events if they're In use by another Handler
            if (m_odeTriggers[trigger]->IsInUse())
            {
                LOG_ERROR("Unable to add ODE Trigger '" << trigger 
                    << "' as it is currently in use");
                return DSL_RESULT_ODE_TRIGGER_IN_USE;
            }

            DSL_PPH_ODE_PTR pOde = 
                std::dynamic_pointer_cast<OdePadProbeHandler>(m_padProbeHandlers[name]);

            if (!pOde->AddChild(m_odeTriggers[trigger]))
            {
                LOG_ERROR("ODE Pad Probe Handler '" << name
                    << "' failed to add ODE Trigger '" << trigger << "'");
                return DSL_RESULT_PPH_ODE_TRIGGER_ADD_FAILED;
            }
            LOG_INFO("ODE Trigger '" << trigger 
                << "' was added to ODE Pad Probe Handler '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name
                << "' threw exception adding ODE Trigger '" << trigger << "'");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphOdeTriggerRemove(const char* name, const char* trigger)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

            if (!m_odeTriggers[trigger]->IsParent(m_padProbeHandlers[name]))
            {
                LOG_ERROR("ODE Trigger '" << trigger << 
                    "' is not in use by ODE Pad Probe Handler '" << name << "'");
                return DSL_RESULT_PPH_ODE_TRIGGER_NOT_IN_USE;
            }
            
            if (!m_padProbeHandlers[name]->RemoveChild(m_odeTriggers[trigger]))
            {
                LOG_ERROR("ODE Pad Probe Handler '" << name
                    << "' failed to remove ODE Trigger '" << trigger << "'");
                return DSL_RESULT_PPH_ODE_TRIGGER_REMOVE_FAILED;
            }
            LOG_INFO("ODE Trigger '" << trigger 
                << "' was removed from ODE Pad Probe Handler '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name 
                << "' threw an exception removing ODE Trigger");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PphOdeTriggerRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            
            m_padProbeHandlers[name]->RemoveAllChildren();

            LOG_INFO("All ODE Triggers removed from ODE Pad Probe Handler '" << name << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Pad Probe Handler '" << name 
                << "' threw an exception removing All ODE Triggers");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

   DslReturnType Services::PphEnabledGet(const char* name, boolean* enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);

            *enabled = m_padProbeHandlers[name]->GetEnabled();

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler '" << name
                << "' threw exception getting the Enabled state");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

   DslReturnType Services::PphEnabledSet(const char* name, boolean enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);

            if (!m_padProbeHandlers[name]->SetEnabled(enabled))
            {
                LOG_ERROR("Pad Probe Handler '" << name
                    << "' failed to set enabled state");
                return DSL_RESULT_PPH_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler '" << name
                << "' threw exception setting the Enabled state");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PphDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            
            if (m_padProbeHandlers[name]->IsInUse())
            {
                LOG_INFO("Pad Probe Handler '" << name << "' is in use");
                return DSL_RESULT_PPH_IS_IN_USE;
            }
            m_padProbeHandlers.erase(name);

            LOG_INFO("Pad Probe Handler '" << name << "' deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler '" << name << "' threw an exception on deletion");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PphDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            for (auto const& imap: m_padProbeHandlers)
            {
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("Pad Probe Handler '" << imap.second->GetName() << "' is currently in use");
                    return DSL_RESULT_PPH_IS_IN_USE;
                }
            }
            m_padProbeHandlers.clear();

            LOG_INFO("All Pad Probe Handlers deleted successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Pad Probe Handler threw an exception on delete all");
            return DSL_RESULT_PPH_THREW_EXCEPTION;
        }
    }

    uint Services::PphListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_padProbeHandlers.size();
    }
    
    DslReturnType Services::SourceCsiNew(const char* name,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_CSI_SOURCE_NEW(name, width, height, fps_n, fps_d);

            LOG_INFO("New CSI Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New CSI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceUsbNew(const char* name,
        uint width, uint height, uint fps_n, uint fps_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_USB_SOURCE_NEW(name, width, height, fps_n, fps_d);

            LOG_INFO("New USB Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New USB Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceUriNew(const char* name, const char* uri, 
        boolean isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
            m_components[name] = DSL_URI_SOURCE_NEW(
                name, uri, isLive, cudadecMemType, intraDecode, dropFrameInterval);

            LOG_INFO("New URI Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New URI Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspNew(const char* name, const char* uri,  uint protocol, 
       uint cudadecMemType, uint intraDecode, uint dropFrameInterval, uint latency, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Source name '" << name << "' is not unique");
                return DSL_RESULT_SOURCE_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_RTSP_SOURCE_NEW(
                name, uri, protocol, cudadecMemType, intraDecode, dropFrameInterval, latency, timeout);

            LOG_INFO("New RTSP Source '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RTSP Source '" << name << "' threw exception on create");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
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
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception getting dimensions");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
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
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception removing Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTimeoutGet(const char* name, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            *timeout = pSourceBintr->GetBufferTimeout();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting buffer timeout");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTimeoutSet(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->SetBufferTimeout(timeout);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception setting buffer timeout");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspReconnectionParamsGet(const char* name, uint* sleep, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->GetReconnectionParams(sleep, timeout);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting reconnection params");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspReconnectionParamsSet(const char* name, uint sleep, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            if (!pSourceBintr->SetReconnectionParams(sleep, timeout))
            {
                LOG_ERROR("RTSP Source '" << name << "' failed to set reconnection params");
                return DSL_RESULT_SOURCE_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception setting reconnection params");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspConnectionDataGet(const char* name, dsl_rtsp_connection_data* data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->GetConnectionData(data);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name << "' threw exception getting Connection Data");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspConnectionStatsClear(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            pSourceBintr->ClearConnectionStats();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception clearing Connection Stats");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SourceRtspStateChangeListenerAdd(const char* name, 
        dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->AddStateChangeListener(listener, clientData))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to add a State Change Listener");
                return DSL_RESULT_SOURCE_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw an exception adding a State Change Lister");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::SourceRtspStateChangeListenerRemove(const char* name, 
        dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
    
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);   

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);

            if (!pSourceBintr->RemoveStateChangeListener(listener))
            {
                LOG_ERROR("RTSP Source '" << name 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_SOURCE_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("RTSP Source '" << name 
                << "' threw an exception removeing a State Change Lister");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceRtspTapAdd(const char* name, const char* tap)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, tap);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);
            RETURN_IF_COMPONENT_IS_NOT_TAP(m_components, tap);

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
                
            if (pSourceBintr->IsLinked())
            {
                LOG_ERROR("Can not add Tap '" << tap << "' to RTSP Source '" << name << 
                    "' as the Source is in a linked state");
                return DSL_RESULT_SOURCE_TAP_ADD_FAILED;
            }
         
            DSL_TAP_PTR pTapBintr = 
                std::dynamic_pointer_cast<TapBintr>(m_components[tap]);
         
            if (!pSourceBintr->AddTapBintr(pTapBintr))
            {
                LOG_ERROR("Failed to add Tap '" << tap << "' to RTSP Source '" << name << "'");
                return DSL_RESULT_SOURCE_TAP_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception adding Tap");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceRtspTapRemove(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSourceBintr);

            DSL_RTSP_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<RtspSourceBintr>(m_components[name]);
         
            if (pSourceBintr->IsLinked())
            {
                LOG_ERROR("Can not remove Tap from RTSP Source '" << name << 
                    "' as the Source is in a linked state");
                return DSL_RESULT_SOURCE_TAP_ADD_FAILED;
            }

            if (!pSourceBintr->RemoveTapBintr())
            {
                LOG_ERROR("Failed to remove Tap from RTSP Source '" << name << "'");
                return DSL_RESULT_SOURCE_TAP_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Source '" << name << "' threw exception removing Tap");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SourceNameGet(int sourceId, const char** name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_sourceNames.find(sourceId) != m_sourceNames.end())
        {
            *name = m_sourceNames[sourceId].c_str();
            return DSL_RESULT_SUCCESS;
        }
        *name = NULL;
        return DSL_RESULT_SOURCE_NOT_FOUND;
    }

    DslReturnType Services::SourceIdGet(const char* name, int* sourceId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        if (m_sourceIds.find(name) != m_sourceIds.end())
        {
            *sourceId = m_sourceIds[name];
            return DSL_RESULT_SUCCESS;
        }
        *sourceId = -1;
        return DSL_RESULT_SOURCE_NOT_FOUND;
    }

    DslReturnType Services::_sourceNameSet(uint sourceId, const char* name)
    {
        LOG_FUNC();
        
        // called internally, do not lock mutex
        
        m_sourceNames[sourceId] = name;
        m_sourceIds[name] = sourceId;
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::_sourceNameErase(uint sourceId)
    {
        LOG_FUNC();

        // called internally, do not lock mutex
        
        if (m_sourceNames.find(sourceId) != m_sourceNames.end())
        {
            m_sourceIds.erase(m_sourceNames[sourceId]);
            m_sourceNames.erase(sourceId);
            return DSL_RESULT_SUCCESS;
        }
        return DSL_RESULT_SOURCE_NOT_FOUND;
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
            GstState state;
            pSourceBintr->GetState(state, 0);
            if (state != GST_STATE_PLAYING)
            {
                LOG_ERROR("Source '" << name << "' can not be paused - is not in play");
                return DSL_RESULT_SOURCE_NOT_IN_PLAY;
            }
            if (!pSourceBintr->SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
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
            GstState state;
            pSourceBintr->GetState(state, 0);
            if (state != GST_STATE_PAUSED)
            {
                LOG_ERROR("Source '" << name << "' can not be resumed - is not in pause");
                return DSL_RESULT_SOURCE_NOT_IN_PAUSE;
            }

            if (!pSourceBintr->SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
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

        try
        {
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

            m_components[name] = DSL_DEWARPER_NEW(name, configFile);

            LOG_INFO("New Dewarper '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Dewarper '" << name << "' threw exception on create");
            return DSL_RESULT_DEWARPER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordNew(const char* name, const char* outdir, uint container, 
        dsl_record_client_listener_cb clientListener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            struct stat info;

            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Tap name '" << name << "' is not unique");
                return DSL_RESULT_TAP_NAME_NOT_UNIQUE;
            }
            // ensure outdir exists
            if ((stat(outdir, &info) != 0) or !(info.st_mode & S_IFDIR))
            {
                LOG_ERROR("Unable to access outdir '" << outdir << "' for Record Tape '" << name << "'");
                return DSL_RESULT_TAP_FILE_PATH_NOT_FOUND;
            }

            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " << container << " for File Tap '" << name << "'");
                return DSL_RESULT_TAP_CONTAINER_VALUE_INVALID;
            }

            m_components[name] = DSL_RECORD_TAP_NEW(name, outdir, 
                container, clientListener);
            
            LOG_INFO("New Record Tap '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Record Tap '" << name << "' threw exception on create");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordSessionStart(const char* name, 
        uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->StartSession(start, duration, clientData))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to Start Session");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Session started successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap'" << name << "' threw an exception Starting Session");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordSessionStop(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->StopSession())
            {
                LOG_ERROR("Record Tap '" << name << "' failed to Stop Session");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Session stopped successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap'" << name << "' threw an exception setting Stoping Session");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordOutdirGet(const char* name, const char** outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            
            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *outdir = pRecordTapBintr->GetOutdir();
            
            LOG_INFO("Outdir = " << *outdir << " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap'" << name << "' threw an exception setting getting outdir");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordOutdirSet(const char* name, const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);
            
            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->SetOutdir(outdir))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set the outdir");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Outdir = " << outdir << " set successfully for Record Tap '" << name << "'");
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception setting getting outdir");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TapRecordContainerGet(const char* name, uint* container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *container = pRecordTapBintr->GetContainer();

            LOG_INFO("Container = " << *container 
                << " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception getting Cache Size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordContainerSet(const char* name, uint container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " 
                    << container << " for Record Tap '" << name << "'");
                return DSL_RESULT_TAP_CONTAINER_VALUE_INVALID;
            }

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            if (!pRecordTapBintr->SetContainer(container))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set container");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Container = " << container 
                << " set successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception setting container type");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::TapRecordCacheSizeGet(const char* name, uint* cacheSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *cacheSize = pRecordTapBintr->GetCacheSize();

            LOG_INFO("Cashe size = " << *cacheSize << 
                " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception getting Cache Size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordCacheSizeSet(const char* name, uint cacheSize)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pRecordTapBintr->SetCacheSize(cacheSize))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set cache size");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Cashe size = " << cacheSize << 
                " set successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception setting cache size");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::TapRecordDimensionsGet(const char* name, uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            pRecordTapBintr->GetDimensions(width, height);

            LOG_INFO("Width = " << *width << " height = " << *height << 
                " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception getting dimensions");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordDimensionsSet(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);


            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pRecordTapBintr->SetDimensions(width, height))
            {
                LOG_ERROR("Record Tap '" << name << "' failed to set dimensions");
                return DSL_RESULT_TAP_SET_FAILED;
            }
            LOG_INFO("Width = " << width << " height = " << height << 
                " returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception setting dimensions");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordIsOnGet(const char* name, boolean* isOn)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *isOn = pRecordTapBintr->IsOn();

            LOG_INFO("Is on = " << *isOn 
                << "returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name 
                << "' threw an exception getting is-recording-on flag");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TapRecordResetDoneGet(const char* name, boolean* resetDone)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[name]);

            *resetDone = pRecordTapBintr->ResetDone();

            LOG_INFO("Reset done = " << *resetDone 
                << "returned successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Tap '" << name << "' threw an exception getting reset done flag");
            return DSL_RESULT_TAP_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::PrimaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
            m_components[name] = DSL_PRIMARY_GIE_NEW(name, 
                inferConfigFile, modelEngineFile, interval);
            LOG_INFO("New Primary GIE '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::PrimaryGiePphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, PrimaryGieBintr);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for PrimaryGie '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("Primary GIE '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_GIE_HANDLER_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::PrimaryGiePphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, PrimaryGieBintr);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Primary GIE '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the PrimaryGie
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler << "' is not a child of Primary GIE '" << name << "'");
                return DSL_RESULT_GIE_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Primary GIE '" << name << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SecondaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, const char* inferOnGieName, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
            
            std::string testPath(modelEngineFile);
            if (testPath.size())
            {
                std::ifstream modelFile(modelEngineFile);
                if (!modelFile.good())
                {
                    LOG_ERROR("Model Engine File not found");
                    return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
                }
            }
            m_components[name] = DSL_SECONDARY_GIE_NEW(name, 
                inferConfigFile, modelEngineFile, inferOnGieName, interval);

            LOG_INFO("New Secondary GIE '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on raw output enabled set");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
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
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw exception on Infer Config file get");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("GIE '" << name << "' threw an exception setting Interval");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerKtlNew(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("KTL Tracker name '" << name << "' is not unique");
                return DSL_RESULT_TRACKER_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new KtlTrackerBintr(
                name, width, height));
            LOG_INFO("New KTL Tracker '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("KTL Tracker '" << name << "' threw exception on create");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::TrackerIouNew(const char* name, const char* configFile, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
            m_components[name] = std::shared_ptr<Bintr>(new IouTrackerBintr(
                name, configFile, width, height));
                
            LOG_INFO("New IOU Tracker '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("IOU Tracker '" << name << "' threw exception on create");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TrackerPphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tracker '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("Tracker '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_TRACKER_HANDLER_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::TrackerPphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TRACKER(m_components, name);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tracker '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the Tracker
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler << "' is not a child of Tracker '" << name << "'");
                return DSL_RESULT_TRACKER_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_TRACKER_THREW_EXCEPTION;
        }
    }
        
    DslReturnType Services::TeeDemuxerNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Demuxer Tee name '" << name << "' is not unique");
                return DSL_RESULT_TEE_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new DemuxerBintr(name));
            
            LOG_INFO("New Demuxer Tee '" << name << "' created successfully");
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Demuxer Tee '" << name << "' threw exception on create");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TeeSplitterNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Splitter Tee name '" << name << "' is not unique");
                return DSL_RESULT_TILER_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new SplitterBintr(name));
            
            LOG_INFO("New Splitter Tee '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Splitter Tee '" << name << "' threw exception on create");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
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
            LOG_INFO("Branch '" << branch 
                << "' was added to Tee '" << tee << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tee '" << tee 
                << "' threw an exception removing branch '" << branch << "'");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tee '" << tee 
                << "' threw an exception removing branch '" << branch << "'");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
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
                
            // TODO WHY?
//            m_components[tee]->RemoveAll();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tee '" <<  tee
                << "' threw an exception removing all branches");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
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
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tee '" <<  tee
                << "' threw an exception getting branch count");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TeePphAdd(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, name);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], DSL_PAD_SINK))
            {
                LOG_ERROR("Tracker '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_TEE_HANDLER_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::TeePphRemove(const char* name, const char* handler) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, name);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to remove itself from the Tee
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], DSL_PAD_SINK))
            {
                LOG_ERROR("Pad Probe Handler '" << handler << "' is not a child of Tracker '" << name << "'");
                return DSL_RESULT_TEE_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tee '" << name << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_TEE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::TilerNew(const char* name, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Tiler name '" << name << "' is not unique");
                return DSL_RESULT_TILER_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new TilerBintr(
                name, width, height));
                
            LOG_INFO("New Tiler '" << name << "' created successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Tiler'" << name << "' threw exception on create");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
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

            tilerBintr->GetDimensions(width, height);
            LOG_INFO("New Tiler '" << name << "' created successfully");
            
            LOG_INFO("Width = " << *width << " height = " << *height << 
                " returned successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
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
            LOG_INFO("Width = " << width << " height = " << height << 
                " set successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerTilesGet(const char* name, uint* columns, uint* rows)
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
            tilerBintr->GetTiles(columns, rows);

            LOG_INFO("Columns = " << *columns << " rows = " << *rows << 
                " returned successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception getting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerTilesSet(const char* name, uint columns, uint rows)
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
            if (!tilerBintr->SetTiles(columns, rows))
            {
                LOG_ERROR("Tiler '" << name << "' failed to set Tiles");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Columns = " << columns << " rows = " << rows << 
                " set successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowGet(const char* name, 
        const char** source, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR tilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            int sourceId(-1);
            tilerBintr->GetShowSource(&sourceId, timeout);
            
            if (sourceId == -1)
            {
                *source = NULL;
                return DSL_RESULT_SUCCESS;
            }
            if (m_sourceNames.find(sourceId) == m_sourceNames.end())
            {
                *source = NULL;
                LOG_ERROR("Tiler '" << name << "' failed to get Source name from Id");
                return DSL_RESULT_SOURCE_NAME_NOT_FOUND;
            }
            *source = m_sourceNames[sourceId].c_str();
            LOG_INFO("Source = " << *source 
                << " returned successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception setting Tiles");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowSet(const char* name, 
        const char* source, uint timeout, bool hasPrecedence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->IsLinked())
            {
                LOG_ERROR("Tiler '" << name 
                    << "' must be in a linked state to show a specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }

            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, source);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[source]);
                    
            if (!pTilerBintr->SetShowSource(pSourceBintr->GetId(), timeout, hasPrecedence))
            {
                LOG_ERROR("Tiler '" << name << "' failed to show specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Source = " << source << " timeout = " << timeout << 
                " has precedence = " << hasPrecedence 
                << " set successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception showing a specific source");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    // Note this instance called internally, i.e. not exposed to client 
    DslReturnType Services::TilerSourceShowSet(const char* name, 
        uint sourceId, uint timeout, bool hasPrecedence)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->IsLinked())
            {
                LOG_ERROR("Tiler '" << name 
                    << "' must be in a linked state to show a specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }

            // called by automation - so set hasPrecedence to false always
            if (!pTilerBintr->SetShowSource(sourceId, timeout, hasPrecedence))
            {
                // Don't log error as this can happen with the ODE actions frequently
                LOG_DEBUG("Tiler '" << name << "' failed to show specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception showing a specific source");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowSelect(const char* name, 
        int xPos, int yPos, uint windowWidth, uint windowHeight, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->IsLinked())
            {
                LOG_ERROR("Tiler '" << name 
                    << "' must be in a linked state to show a specific source");
                return DSL_RESULT_TILER_SET_FAILED;
            }

            int sourceId(0);
            uint currentTimeout(0);
            pTilerBintr->GetShowSource(&sourceId, &currentTimeout);
            
            // if currently showing all sources
            if (sourceId == -1)
            {
                uint cols(0), rows(0);
                pTilerBintr->GetTiles(&cols, &rows);
                if (rows*cols == 1)
                {
                    // single source, noting to do
                    return DSL_RESULT_SUCCESS;
                }
                float xRel((float)xPos/windowWidth), yRel((float)yPos/windowHeight);
                sourceId = (int)(xRel*cols);
                sourceId += ((int)(yRel*rows))*cols;
                
                if (sourceId > pTilerBintr->GetBatchSize())
                {
                    // clicked on empty tile, noting to do
                    return DSL_RESULT_SUCCESS;
                }

                if (!pTilerBintr->SetShowSource(sourceId, timeout, true))
                {
                    LOG_ERROR("Tiler '" << name << "' failed to select specific source");
                    return DSL_RESULT_TILER_SET_FAILED;
                }
                LOG_INFO("xPos = " << xPos << " yPos = " << yPos 
                    << " window width = " << windowWidth 
                    << " window hidth = " << windowHeight
                    << " timeout = " << timeout << "selected successfully for Tiler '" 
                    << name << "'");
            }
            // else, showing a single source so return to all sources. 
            else
            {
                pTilerBintr->ShowAllSources();
                LOG_INFO("Return to show all set successfully for Tiler '" << name << "'");
            }
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception showing a specific source");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            pTilerBintr->ShowAllSources();
            LOG_INFO("Show all sources set successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception showing all sources");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::TilerSourceShowCycle(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);

            DSL_TILER_PTR pTilerBintr = 
                std::dynamic_pointer_cast<TilerBintr>(m_components[name]);

            if (!pTilerBintr->CycleAllSources(timeout))
            {
                    LOG_ERROR("Tiler '" << name << "' failed to select specific source");
                    return DSL_RESULT_TILER_SET_FAILED;
            }
            LOG_INFO("Cycle all sources with timeout " << timeout 
                << " set successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception showing all sources");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }
    DslReturnType Services::TilerPphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tiler '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("Tiler '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_TILER_HANDLER_ADD_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << handler 
                << "' added to Tiler '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::TilerPphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, TilerBintr);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for Tiler '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the Tiler
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Tiler '" << name << "'");
                return DSL_RESULT_TILER_HANDLER_REMOVE_FAILED;
            }
            LOG_INFO("Pad Probe Handler '" << handler 
                << "' removed from Tiler '" << name << "' successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Tiler '" << name << "' threw an exception removing ODE Handle");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
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
    
    DslReturnType Services::OsdNew(const char* name, 
        boolean textEnabled, boolean clockEnabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {   
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("OSD name '" << name << "' is not unique");
                return DSL_RESULT_OSD_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new OsdBintr(
                name, textEnabled, clockEnabled));
                    
            LOG_INFO("New OSD '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New OSD '" << name << "' threw exception on create");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OsdTextEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetTextEnabled(enabled);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting text enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdTextEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

//            if (m_components[name]->IsInUse())
//            {
//                LOG_ERROR("Unable to set The clock enabled setting for the OSD '" << name 
//                    << "' as it's currently in use");
//                return DSL_RESULT_OSD_IS_IN_USE;
//            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetTextEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Text enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

//            if (m_components[name]->IsInUse())
//            {
//                LOG_ERROR("Unable to set The clock enabled setting for the OSD '" << name 
//                    << "' as it's currently in use");
//                return DSL_RESULT_OSD_IS_IN_USE;
//            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
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
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockOffsetsSet(const char* name, uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

//            if (m_components[name]->IsInUse())
//            {
//                LOG_ERROR("Unable to set The clock offsets for the OSD '" << name 
//                    << "' as it's currently in use");
//                return DSL_RESULT_OSD_IS_IN_USE;
//            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockOffsets(offsetX, offsetY))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock offsets");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
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
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockFontSet(const char* name, const char* font, uint size)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

//            if (m_components[name]->IsInUse())
//            {
//                LOG_ERROR("Unable to set The clock offsets for the OSD '" << name 
//                    << "' as it's currently in use");
//                return DSL_RESULT_OSD_IS_IN_USE;
//            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockFont(font, size))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock font");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            LOG_INFO("OSD '" << name << "' set clock font successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OsdClockColorSet(const char* name, double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

//            if (m_components[name]->IsInUse())
//            {
//                LOG_ERROR("Unable to set The clock RGB colors for the OSD '" << name 
//                    << "' as it's currently in use");
//                return DSL_RESULT_OSD_IS_IN_USE;
//            }

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            // TODO verify args before calling
            if (!osdBintr->SetClockColor(red, green, blue, alpha))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Clock RGB colors");
                return DSL_RESULT_OSD_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Clock offsets");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OsdPphAdd(const char* name, const char* handler, uint pad)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for OSD '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to add itself to the Osd as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], pad))
            {
                LOG_ERROR("OSD '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_OSD_HANDLER_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::OsdPphRemove(const char* name, const char* handler, uint pad) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            if (pad > DSL_PAD_SRC)
            {
                LOG_ERROR("Invalid Pad type = " << pad << " for OSD '" << name << "'");
                return DSL_RESULT_PPH_PAD_TYPE_INVALID;
            }

            // call on the Handler to remove itself from the Osd
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], pad))
            {
                LOG_ERROR("Pad Probe Handler '" << handler << "' is not a child of OSD '" << name << "'");
                return DSL_RESULT_OSD_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Osd '" << name << "' threw an exception removing ODE Handle");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::SinkFakeNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Sink name '" << name << "' is not unique");
                return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_FAKE_SINK_NEW(name);

            LOG_INFO("New Fake Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkOverlayNew(const char* name, uint overlay_id, uint display_id,
        uint depth, uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Sink name '" << name << "' is not unique");
                return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_OVERLAY_SINK_NEW(
                name, overlay_id, display_id, depth, offsetX, offsetY, width, height);

            LOG_INFO("New Overlay Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SinkWindowNew(const char* name, 
        uint offsetX, uint offsetY, uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Sink name '" << name << "' is not unique");
                return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
            }
            m_components[name] = DSL_WINDOW_SINK_NEW(name, offsetX, offsetY, width, height);

            LOG_INFO("New Window Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SinkWindowForceAspectRationGet(const char* name, 
        boolean* force)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, WindowSinkBintr);

            DSL_WINDOW_SINK_PTR pWindowSinkBintr = 
                std::dynamic_pointer_cast<WindowSinkBintr>(m_components[name]);

            *force = pWindowSinkBintr->GetForceAspectRatio();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Window Sink'" << name 
                << "' threw an exception getting 'force-aspect-ratio' property");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkWindowForceAspectRationSet(const char* name, 
        boolean force)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, WindowSinkBintr);

            DSL_WINDOW_SINK_PTR pWindowSinkBintr = 
                std::dynamic_pointer_cast<WindowSinkBintr>(m_components[name]);

            if (!pWindowSinkBintr->SetForceAspectRatio(force))
            {
                LOG_ERROR("Window Sink '" << name 
                    << "' failed to Set 'force-aspec-ratio' property");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Window Sink'" << name 
                << "' threw an exception setting 'force-apect-ratio' property");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
        
    
    DslReturnType Services::SinkFileNew(const char* name, const char* filepath, 
            uint codec, uint container, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
            m_components[name] = DSL_FILE_SINK_NEW(name, filepath, codec, container, bitrate, interval);
            LOG_INFO("New File Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SinkRecordNew(const char* name, const char* outdir, uint codec, uint container, 
        uint bitrate, uint interval, dsl_record_client_listener_cb clientListener)
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

            if (codec > DSL_CODEC_MPEG4)
            {   
                LOG_ERROR("Invalid Codec value = " << codec << " for Record Sink '" << name << "'");
                return DSL_RESULT_SINK_CODEC_VALUE_INVALID;
            }
            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " << container << " for Record Sink '" << name << "'");
                return DSL_RESULT_SINK_CONTAINER_VALUE_INVALID;
            }

            m_components[name] = DSL_RECORD_SINK_NEW(name, outdir, 
                codec, container, bitrate, interval, clientListener);
            
            LOG_INFO("New Record Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Record Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordSessionStart(const char* name, 
        uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            if (!recordSinkBintr->StartSession(start, duration, clientData))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to Start Session");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink'" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordSessionStop(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            if (!recordSinkBintr->StopSession())
            {
                LOG_ERROR("Record Sink '" << name << "' failed to Stop Session");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink'" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordOutdirGet(const char* name, const char** outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);
            
            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            *outdir = pRecordSinkBintr->GetOutdir();
            
            LOG_INFO("Outdir = " << *outdir << " returned successfully for Record Sink '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink'" << name << "' threw an exception setting getting outdir");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordOutdirSet(const char* name, const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);
            
            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            if (!pRecordSinkBintr->SetOutdir(outdir))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to set the outdir");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            LOG_INFO("Outdir = " << outdir << " set successfully for Record Sink '" << name << "'");
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception setting getting outdir"); 
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SinkRecordContainerGet(const char* name, uint* container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            *container = pRecordSinkBintr->GetContainer();

            LOG_INFO("Container = " << *container 
                << " returned successfully for Record Sink '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting Cache Size");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordContainerSet(const char* name, uint container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            if (container > DSL_CONTAINER_MKV)
            {   
                LOG_ERROR("Invalid Container value = " 
                    << container << " for Record Sink '" << name << "'");
                return DSL_RESULT_SINK_CONTAINER_VALUE_INVALID;
            }

            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            if (!pRecordSinkBintr->SetContainer(container))
            {
                LOG_ERROR("Record Sink '" << name << "' failed to set container");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            LOG_INFO("Container = " << container 
                << " set successfully for Record Tap '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception setting container type");
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting cache size");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
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
                LOG_ERROR("Record Sink '" << name << "' failed to set cache size");
                return DSL_RESULT_SINK_SET_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception setting s");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception setting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordIsOnGet(const char* name, boolean* isOn)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            *isOn = recordSinkBintr->IsOn();

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting is-recording-on flag");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkRecordResetDoneGet(const char* name, boolean* resetDone)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RecordSinkBintr);

            DSL_RECORD_SINK_PTR recordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[name]);

            *resetDone = recordSinkBintr->ResetDone();

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Record Sink '" << name << "' threw an exception getting reset done flag");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkEncodeVideoFormatsGet(const char* name, uint* codec, uint* container)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_ENCODE_SINK(m_components, name);

            DSL_ENCODE_SINK_PTR encodeSinkBintr = 
                std::dynamic_pointer_cast<EncodeSinkBintr>(m_components[name]);

            encodeSinkBintr->GetVideoFormats(codec, container);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Sink '" << name << "' threw an exception getting Video formats");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkEncodeSettingsGet(const char* name, uint* bitrate, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_ENCODE_SINK(m_components, name);

            DSL_ENCODE_SINK_PTR encodeSinkBintr = 
                std::dynamic_pointer_cast<EncodeSinkBintr>(m_components[name]);

            encodeSinkBintr->GetEncoderSettings(bitrate, interval);
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("File Sink '" << name << "' threw an exception getting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkEncodeSettingsSet(const char* name, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_ENCODE_SINK(m_components, name);

            if (m_components[name]->IsLinked())
            {
                LOG_ERROR("Unable to set Encoder settings for File Sink '" << name 
                    << "' as it's currently linked");
                return DSL_RESULT_SINK_IS_IN_USE;
            }

            DSL_ENCODE_SINK_PTR encodeSinkBintr = 
                std::dynamic_pointer_cast<EncodeSinkBintr>(m_components[name]);

            if (!encodeSinkBintr->SetEncoderSettings(bitrate, interval))
            {
                LOG_ERROR("Encode Sink '" << name << "' failed to set Encoder settings");
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

    DslReturnType Services::SinkRtspNew(const char* name, const char* host, 
            uint udpPort, uint rtspPort, uint codec, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
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
            m_components[name] = DSL_RTSP_SINK_NEW(name, host, udpPort, rtspPort, codec, bitrate, interval);

            LOG_INFO("New RTSP Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RTSP Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SinkRtspServerSettingsGet(const char* name, uint* udpPort, uint* rtspPort, uint* codec)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, RtspSinkBintr);
            
            DSL_RTSP_SINK_PTR rtspSinkBintr = 
                std::dynamic_pointer_cast<RtspSinkBintr>(m_components[name]);

            rtspSinkBintr->GetServerSettings(udpPort, rtspPort, codec);

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Sink '" << name << "' threw an exception getting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
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

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Sink '" << name << "' threw an exception getting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
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
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RTSP Sink '" << name << "' threw an exception setting Encoder settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkPphAdd(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SINK(m_components, name);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], DSL_PAD_SINK))
            {
                LOG_ERROR("SINK '" << name << "' failed to add Pad Probe Handler");
                return DSL_RESULT_SINK_HANDLER_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Sink '" << name << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::SinkPphRemove(const char* name, const char* handler) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SINK(m_components, name);
            RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to remove itself from the Tee
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], DSL_PAD_SINK))
            {
                LOG_ERROR("Pad Probe Handler '" << handler << "' is not a child of Tracker '" << name << "'");
                return DSL_RESULT_SINK_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Sink '" << name << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkSyncSettingsGet(const char* name,  boolean* sync, boolean* async)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SINK(m_components, name);

            DSL_SINK_PTR pSinkBintr = 
                std::dynamic_pointer_cast<SinkBintr>(m_components[name]);

            bool bSync(false), bAsync(false);
            pSinkBintr->GetSyncSettings(&bSync, &bAsync);
            *sync = bSync;
            *async = bAsync;
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Sink '" << name << "' threw an exception getting  Sync/Async settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkSyncSettingsSet(const char* name,  boolean sync, boolean async)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_SINK(m_components, name);

            DSL_SINK_PTR pSinkBintr = 
                std::dynamic_pointer_cast<SinkBintr>(m_components[name]);

            if (!pSinkBintr->SetSyncSettings(sync, async))
            {
                LOG_ERROR("Sink '" << name << "' failed to set sync/async attributes");
                return DSL_RESULT_SINK_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Sink '" << name << "' threw an exception setting sync/async settings");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
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

        m_components.clear();
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
        
        try
        {
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, component);

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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
        dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddStateChangeListener(listener, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->RemoveStateChangeListener(listener))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove a State Change Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddEosListener(listener, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add an EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->RemoveEosListener(listener))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to remove an EOS Listener");
                return DSL_RESULT_PIPELINE_CALLBACK_REMOVE_FAILED;
            }
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
            if (!m_pipelines[pipeline]->AddErrorMessageHandler(handler, clientData))
            {
                LOG_ERROR("Pipeline '" << pipeline 
                    << "' failed to add an Error Message Handler");
                return DSL_RESULT_PIPELINE_CALLBACK_ADD_FAILED;
            }
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);
            
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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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
            RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

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

    DslReturnType Services::SmtpMailEnabledGet(boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            *enabled = m_pComms->GetSmtpMailEnabled();
            LOG_INFO("Returning SMTP Mail Enabled = " << *enabled);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpMailEnabledSet(boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            m_pComms->SetSmtpMailEnabled(enabled);
            LOG_INFO("Setting SMTP Mail Enabled = " << enabled);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
     
    DslReturnType Services::SmtpCredentialsSet(const char* username, const char* password)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            LOG_INFO("New SMTP Username and Password set");
            
            m_pComms->SetSmtpCredentials(username, password);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpServerUrlGet(const char** serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->GetSmtpServerUrl(serverUrl);

            LOG_INFO("Returning SMTP Server URL = '" << *serverUrl << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpServerUrlSet(const char* serverUrl)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->SetSmtpServerUrl(serverUrl);

            LOG_INFO("New SMTP Server URL = '" << serverUrl << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpFromAddressGet(const char** name, const char** address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->GetSmtpFromAddress(name, address);

            LOG_INFO("Returning SMTP From Address with Name = '" << *name 
                << "', and Address = '" << *address << "'" );
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpFromAddressSet(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->SetSmtpFromAddress(name, address);

            LOG_INFO("New SMTP From Address with Name = '" << name 
                << "', and Address = '" << address << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpSslEnabledGet(boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            *enabled = m_pComms->GetSmtpSslEnabled();
            
            LOG_INFO("Returning SMTP SSL Enabled = '" << *enabled  << "'" );
            
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpSslEnabledSet(boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->SetSmtpSslEnabled(enabled);
            LOG_INFO("Set SMTP SSL Enabled = '" << enabled  << "'" );
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpToAddressAdd(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->AddSmtpToAddress(name, address);

            LOG_INFO("New SMTP To Address with Name = '" << name 
                << "', and Address = '" << address << "' added");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpToAddressesRemoveAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            m_pComms->RemoveAllSmtpToAddresses();

            LOG_INFO("All SMTP To Addresses removed");
        
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SmtpCcAddressAdd(const char* name, const char* address)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            m_pComms->AddSmtpCcAddress(name, address);

            LOG_INFO("New SMTP Cc Address with Name = '" << name 
                << "', and Address = '" << address << "' set");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SmtpCcAddressesRemoveAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            m_pComms->RemoveAllSmtpCcAddresses();

            LOG_INFO("All SMTP Cc Addresses removed");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SendSmtpTestMessage()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            std::string subject("Test message");
            std::string bline1("Test message.\r\n");
            
            std::vector<std::string> body{bline1};

            if (!m_pComms->QueueSmtpMessage(subject, body))
            {
                LOG_ERROR("Failed to queue SMTP Test Message");
                return DSL_RESULT_FAILURE;
            }
            LOG_INFO("Test message Queued successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("DSL threw an exception enabling SMTP Mail");
            return DSL_RESULT_THREW_EXCEPTION;
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
        m_returnValueToString[DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID] = L"DSL_RESULT_TRACKER_MAX_DIMENSIONS_INVALID";
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
        m_returnValueToString[DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_ODE_ACTION_NOT_THE_CORRECT_TYPE";
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
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED] = L"DSL_RESULT_SINK_OBJECT_CAPTURE_CLASS_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_HANDLER_ADD_FAILED] = L"DSL_RESULT_SINK_HANDLER_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_SINK_HANDLER_REMOVE_FAILED] = L"DSL_RESULT_SINK_HANDLER_REMOVE_FAILED";
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
        
        m_returnValueToString[DSL_RESULT_INVALID_RESULT_CODE] = L"Invalid DSL Result CODE";
    }
    
    std::shared_ptr<Comms> Services::GetComms()
    {
        return m_pComms;
    }

} // namespace
 