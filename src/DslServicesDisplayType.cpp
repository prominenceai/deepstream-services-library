/*
The MIT License

Copyright (c)   2021, Prominence AI, Inc.

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
#include "DslServicesValidate.h"
#include "DslTrackerBintr.h"

namespace DSL
{
    void Services::DisplayTypeCreateIntrinsicTypes()
    {
        LOG_FUNC();

        DSL_RGBA_COLOR_PTR pNoColor = DSL_RGBA_COLOR_NEW(
            DISPLAY_TYPE_NO_COLOR.c_str(), 0.0, 0.0, 0.0, 0.0);
        m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_COLOR.c_str()] = pNoColor;
        m_intrinsicDisplayTypes[DISPLAY_TYPE_NO_FONT.c_str()] = DSL_RGBA_FONT_NEW(
            DISPLAY_TYPE_NO_FONT.c_str(), "arial", 0, pNoColor);
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

}