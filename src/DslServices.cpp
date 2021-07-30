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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);
            
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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            
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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            
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
            if (m_displayTypes.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
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

    DslReturnType Services::OdeActionCaptureCompleteListenerAdd(const char* name, 
        dsl_capture_complete_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);   

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->AddCaptureCompleteListener(listener, clientData))
            {
                LOG_ERROR("ODE Capture Action '" << name 
                    << "' failed to add a Capture Complete Listener");
                return DSL_RESULT_ODE_ACTION_CALLBACK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding a Capture Complete Lister");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
        
    DslReturnType Services::OdeActionCaptureCompleteListenerRemove(const char* name, 
        dsl_capture_complete_listener_cb listener)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);   

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->RemoveCaptureCompleteListener(listener))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to add a Capture Complete Listener");
                return DSL_RESULT_ODE_ACTION_CALLBACK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding a Capture Complete Lister");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeActionCaptureImagePlayerAdd(const char* name, 
        const char* player)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, player);
            DSL_RETURN_IF_PLAYER_IS_NOT_IMAGE_PLAYER(m_players, player)

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->AddImagePlayer(m_players[player]))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to add Player '" << player << "'");
                return DSL_RESULT_ODE_ACTION_PLAYER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding Player '" << player << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OdeActionCaptureImagePlayerRemove(const char* name, 
        const char* player)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_PLAYER_NAME_NOT_FOUND(m_players, player);
            DSL_RETURN_IF_PLAYER_IS_NOT_IMAGE_PLAYER(m_players, player)

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->RemoveImagePlayer(m_players[player]))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to remove Player '" << player << "'");
                return DSL_RESULT_ODE_ACTION_PLAYER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception removeing Player '" << player << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OdeActionCaptureMailerAdd(const char* name, 
        const char* mailer, const char* subject, boolean attach)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer);

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->AddMailer(m_mailers[mailer], subject, attach))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to add Mailer '" << mailer << "'");
                return DSL_RESULT_ODE_ACTION_MAILER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception adding Mailer '" << mailer << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OdeActionCaptureMailerRemove(const char* name, 
        const char* mailer)
    {
        LOG_FUNC();
    
        try
        {
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CAPTURE_TYPE(m_odeActions, name);
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer);

            DSL_ODE_ACTION_CATPURE_PTR pOdeAction = 
                std::dynamic_pointer_cast<CaptureOdeAction>(m_odeActions[name]);

            if (!pOdeAction->RemoveMailer(m_mailers[mailer]))
            {
                LOG_ERROR("Capture Action '" << name 
                    << "' failed to remove Mailer '" << mailer << "'");
                return DSL_RESULT_ODE_ACTION_MAILER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("ODE Capture Action '" << name 
                << "' threw an exception removeing Player '" << mailer << "'");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, font, RgbaFont);
            
            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, bgColor, RgbaColor);

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

    DslReturnType Services::OdeActionEmailNew(const char* name, 
        const char* mailer, const char* subject)
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
            DSL_RETURN_IF_MAILER_NAME_NOT_FOUND(m_mailers, mailer)
            
            m_odeActions[name] = DSL_ODE_ACTION_EMAIL_NEW(name, 
                m_mailers[mailer], subject);

            LOG_INFO("New ODE Email Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE Email Action '" << name << "' threw exception on create");
            return DSL_RESULT_ODE_ACTION_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::OdeActionFileNew(const char* name, 
        const char* filePath, uint mode, uint format, boolean forceFlush)
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
            if (mode > DSL_EVENT_FILE_MODE_TRUNCATE)
            {
                LOG_ERROR("File open mode " << mode 
                    << " is invalid for ODE Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            if (format > DSL_EVENT_FILE_FORMAT_CSV)
            {
                LOG_ERROR("File format " << format 
                    << " is invalid for ODE Action '" << name << "'");
                return DSL_RESULT_ODE_ACTION_PARAMETER_INVALID;
            }
            m_odeActions[name] = DSL_ODE_ACTION_FILE_NEW(name, 
                filePath, mode, format, forceFlush);

            LOG_INFO("New ODE File Action '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New ODE File Action '" << name << "' threw exception on create");
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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

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

            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, color, RgbaColor);

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
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, displayType);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(m_displayTypes, displayType);
            
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
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            DSL_RETURN_IF_ODE_ACTION_IS_NOT_CORRECT_TYPE(m_odeActions, name, AddDisplayMetaOdeAction);
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, displayType);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_BASE_TYPE(m_displayTypes, displayType);
            
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
    
    DslReturnType Services::OdeActionPrintNew(const char* name,
        boolean forceFlush)
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
            m_odeActions[name] = DSL_ODE_ACTION_PRINT_NEW(name, forceFlush);

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

            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordSink);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordSink, RecordSinkBintr);

            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[recordSink]);
            
            m_odeActions[name] = DSL_ODE_ACTION_SINK_RECORD_START_NEW(name,
                pRecordSinkBintr, start, duration, clientData);

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

            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordSink);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordSink, RecordSinkBintr);

            DSL_RECORD_SINK_PTR pRecordSinkBintr = 
                std::dynamic_pointer_cast<RecordSinkBintr>(m_components[recordSink]);
            
            m_odeActions[name] = DSL_ODE_ACTION_SINK_RECORD_STOP_NEW(name,
                pRecordSinkBintr);

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
            
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordTap);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordTap, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[recordTap]);

            m_odeActions[name] = DSL_ODE_ACTION_TAP_RECORD_START_NEW(name,
                pRecordTapBintr, start, duration, clientData);

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
            
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, recordTap);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, recordTap, RecordTapBintr);

            DSL_RECORD_TAP_PTR pRecordTapBintr = 
                std::dynamic_pointer_cast<RecordTapBintr>(m_components[recordTap]);
            
            m_odeActions[name] = DSL_ODE_ACTION_TAP_RECORD_STOP_NEW(name, pRecordTapBintr);

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
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
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
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
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
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, name);
            
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
            if (m_odeActions.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, polygon);
            
            // Interim ... only supporting rectangles at this
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, polygon, RgbaPolygon);
            
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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, polygon);
            
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, polygon, RgbaPolygon);

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
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, line);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, line, RgbaLine);
            
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
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeAreas, name);
            
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
            if (m_odeAreas.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
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
            LOG_ERROR("New Always ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerOccurrenceNew(const char* name, 
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_OCCURRENCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Occurrence ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Occurrence ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerAbsenceNew(const char* name, 
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ABSENCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Absence ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Absence ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    
    DslReturnType Services::OdeTriggerAccumulationNew(const char* name, 
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_ACCUMULATION_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Accumulation ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Accumulation ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerInstanceNew(const char* name, 
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
            m_odeTriggers[name] = DSL_ODE_TRIGGER_INSTANCE_NEW(name, 
                source, classId, limit);
            
            LOG_INFO("New Instance ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Instance ODE Trigger '" << name 
                << "' threw exception on create");
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
            
            LOG_INFO("New Intersection ODE Trigger '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Intersection ODE Trigger '" << name 
                << "' threw exception on create");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::OdeTriggerSummationNew(const char* name, 
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

    DslReturnType Services::OdeTriggerPersistenceRangeGet(const char* name, 
        uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, PersistenceOdeTrigger);
            
            DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<PersistenceOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetRange(minimum, maximum);
            
            // check for no maximum
            *maximum = (*maximum == UINT32_MAX) ? 0 : *maximum;

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Persistence Trigger '" << name 
                << "' threw exception getting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerPersistenceRangeSet(const char* name, 
        uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, PersistenceOdeTrigger);
            
            DSL_ODE_TRIGGER_PERSISTENCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<PersistenceOdeTrigger>(m_odeTriggers[name]);

            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
         
            pOdeTrigger->SetRange(minimum, maximum);
            
            LOG_INFO("ODE Persistence Trigger '" << name << "' set new range from mimimum " 
                << minimum << " to maximum " << maximum << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Persistence Trigger '" << name 
                << "' threw exception setting range");
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

    DslReturnType Services::OdeTriggerCountRangeGet(const char* name, 
        uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, CountOdeTrigger);
            
            DSL_ODE_TRIGGER_COUNT_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CountOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetRange(minimum, maximum);
            
            // check for no maximum
            *maximum = (*maximum == UINT32_MAX) ? 0 : *maximum;

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Count Trigger '" << name 
                << "' threw exception getting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerCountRangeSet(const char* name, 
        uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, CountOdeTrigger);
            
            DSL_ODE_TRIGGER_COUNT_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<CountOdeTrigger>(m_odeTriggers[name]);

            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
         
            pOdeTrigger->SetRange(minimum, maximum);
            
            LOG_INFO("ODE Count Trigger '" << name << "' set new range from mimimum " 
                << minimum << " to maximum " << maximum << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Count Trigger '" << name 
                << "' threw exception setting range");
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
            
    DslReturnType Services::OdeTriggerDistanceRangeGet(const char* name, 
        uint* minimum, uint* maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->GetRange(minimum, maximum);
            
            // check for no maximum
            *maximum = (*maximum == UINT32_MAX) ? 0 : *maximum;

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception getting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerDistanceRangeSet(const char* name, 
        uint minimum, uint maximum)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);

            // check for no maximum
            maximum = (maximum == 0) ? UINT32_MAX : maximum;
         
            pOdeTrigger->SetRange(minimum, maximum);
            
            LOG_INFO("ODE Distance Trigger '" << name << "' set new range from mimimum " 
                << minimum << " to maximum " << maximum << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception setting range");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerDistanceTestParamsGet(const char* name, 
        uint* testPoint, uint* testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->GetTestParams(testPoint, testMethod);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception getting test parameters");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerDistanceTestParamsSet(const char* name, 
        uint testPoint, uint testMethod)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_odeTriggers, name, DistanceOdeTrigger);
            
            DSL_ODE_TRIGGER_DISTANCE_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<DistanceOdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetTestParams(testPoint, testMethod);

            LOG_INFO("ODE Distance Trigger '" << name << "' set new test parameters test_point " 
                << testPoint << " and test_method " << testMethod << " successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Distance Trigger '" << name 
                << "' threw exception setting test parameters");
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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

    DslReturnType Services::OdeTriggerResetTimeoutGet(const char* name, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *timeout = pOdeTrigger->GetResetTimeout();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Reset Timer");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerResetTimeoutSet(const char* name, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetResetTimeout(timeout);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting Reset Timer");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_TRIGGER_IS_NOT_AB_TYPE(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *inferDoneOnly = pOdeTrigger->GetInferDoneOnlySetting();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Inference Done Only");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerInferDoneOnlySet(const char* name, boolean inferDoneOnly)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);

            pOdeTrigger->SetInferDoneOnlySetting(inferDoneOnly);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name 
                << "' threw exception getting Inference Done Only");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                

    DslReturnType Services::OdeTriggerIntervalGet(const char* name, uint* interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            *interval = pOdeTrigger->GetInterval();
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception getting Interval");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerIntervalSet(const char* name, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
            DSL_ODE_TRIGGER_PTR pOdeTrigger = 
                std::dynamic_pointer_cast<OdeTrigger>(m_odeTriggers[name]);
         
            pOdeTrigger->SetInterval(interval);
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("ODE Trigger '" << name << "' threw exception setting Interval");
            return DSL_RESULT_ODE_TRIGGER_THREW_EXCEPTION;
        }
    }                
    
    DslReturnType Services::OdeTriggerActionAdd(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_ACTION_NAME_NOT_FOUND(m_odeActions, action);

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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, area);

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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            DSL_RETURN_IF_ODE_AREA_NAME_NOT_FOUND(m_odeAreas, area);

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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);

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
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, name);
            
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
            if (m_odeTriggers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, MeterPadProbeHandler);

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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, MeterPadProbeHandler);
            
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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            DSL_RETURN_IF_ODE_TRIGGER_NAME_NOT_FOUND(m_odeTriggers, trigger);

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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_padProbeHandlers, name, OdePadProbeHandler);
            
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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);

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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);

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
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, name);
            
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
            if (m_padProbeHandlers.empty())
            {
                return DSL_RESULT_SUCCESS;
            }
            for (auto const& imap: m_padProbeHandlers)
            {
                if (imap.second->IsInUse())
                {
                    LOG_ERROR("Pad Probe Handler '" << imap.second->GetName() 
                        << "' is currently in use");
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
    
    DslReturnType Services::SegVisualNew(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure element name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Segmentation Visualizer name '" << name 
                    << "' is not unique");
                return DSL_RESULT_SEGVISUAL_NAME_NOT_UNIQUE;
            }
            m_components[name] = std::shared_ptr<Bintr>(new SegVisualBintr(
                name, width, height));
                
            LOG_INFO("New Segmentation Visualizer '" << name 
                << "' created successfully");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Segmentation Visualizer'" << name 
                << "' threw exception on create");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SegVisualDimensionsGet(const char* name, 
        uint* width, uint* height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);

            DSL_SEGVISUAL_PTR pSegVisual = 
                std::dynamic_pointer_cast<SegVisualBintr>(m_components[name]);

            pSegVisual->GetDimensions(width, height);
            
            LOG_INFO("Width = " << *width << " height = " << *height << 
                " returned successfully for Segmentation Visualizer '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception getting dimensions");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SegVisualDimensionsSet(const char* name, 
        uint width, uint height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);

            DSL_SEGVISUAL_PTR pSegVisual = 
                std::dynamic_pointer_cast<SegVisualBintr>(m_components[name]);

            // TODO verify args before calling
            if (!pSegVisual->SetDimensions(width, height))
            {
                LOG_ERROR("Segmentation Visualizer '" << name 
                    << "' failed to set dimensions");
                return DSL_RESULT_SEGVISUAL_SET_FAILED;
            }
            LOG_INFO("Width = " << width << " height = " << height << 
                " set successfully for Tiler '" << name << "'");
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception setting dimensions");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SegVisualPphAdd(const char* name, const char* handler)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to add itself to the Tiler as a PadProbeHandler
            if (!m_padProbeHandlers[handler]->AddToParent(m_components[name], DSL_PAD_SRC))
            {
                LOG_ERROR("Segmentation Visualizer '" << name 
                    << "' failed to add Pad Probe Handler");
                return DSL_RESULT_SEGVISUAL_HANDLER_ADD_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception adding Pad Probe Handler");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
        }
    }
   
    DslReturnType Services::SegVisualPphRemove(const char* name, const char* handler) 
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, SegVisualBintr);
            DSL_RETURN_IF_PPH_NAME_NOT_FOUND(m_padProbeHandlers, handler);

            // call on the Handler to remove itself from the Tee
            if (!m_padProbeHandlers[handler]->RemoveFromParent(m_components[name], DSL_PAD_SRC))
            {
                LOG_ERROR("Pad Probe Handler '" << handler 
                    << "' is not a child of Segmentation Visualizer '" << name << "'");
                return DSL_RESULT_SEGVISUAL_HANDLER_REMOVE_FAILED;
            }
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("Segmentation Visualizer '" << name 
                << "' threw an exception removing Pad Probe Handler");
            return DSL_RESULT_SEGVISUAL_THREW_EXCEPTION;
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
 