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
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if ( red > 1.0)
            {
                LOG_ERROR("Invalid red parameter = " << red 
                    << " greater than 1.0 for RGBA Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( green > 1.0)
            {
                LOG_ERROR("Invalid green parameter = " << green 
                    << " greater than 1.0 for RGBA Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( blue > 1.0)
            {
                LOG_ERROR("Invalid blue parameter = " << blue
                    << " greater than 1.0 for RGBA Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( alpha > 1.0)
            {
                LOG_ERROR("Invalid alpha parameter = " << alpha 
                    << " greater than 1.0 for RGBA Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
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

    DslReturnType Services::DisplayTypeRgbaColorPredefinedNew(const char* name, 
        uint colorId, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Predefined Color name '" << name 
                    << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if (colorId > DSL_COLOR_PREDEFINED_LAVENDER)
            {
                LOG_ERROR("Invalid color_id value of " << colorId 
                    << " for New RGBA Predefined Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( alpha > 1.0)
            {
                LOG_ERROR("Invalid alpha parameter = " << alpha 
                    << " greater than 1.0 for RGBA Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            m_displayTypes[name] = DSL_RGBA_PREDEFINED_COLOR_NEW(name, 
                colorId, alpha);

            LOG_INFO("New RGBA Predefined Color '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Predefined Color '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorRandomNew(const char* name, 
        uint hue, uint luminosity, double alpha, uint seed)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Random Color name '" << name 
                    << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if (hue > DSL_COLOR_HUE_BROWN)
            {
                LOG_ERROR("Invalid hue value of " << hue 
                    << " for New RGBA Random Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if (luminosity > DSL_COLOR_LUMINOSITY_RANDOM)
            {
                LOG_ERROR("Invalid luminosity value of " << luminosity 
                    << " for New RGBA Random Color '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( alpha > 1.0)
            {
                LOG_ERROR("Invalid alpha parameter = " << alpha 
                    << " greater than 1.0 for RGBA RandomColor '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            m_displayTypes[name] = DSL_RGBA_RANDOM_COLOR_NEW(name, 
                hue, luminosity, alpha, seed);

            LOG_INFO("New RGBA Random Color '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Random Color '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorOnDemandNew(const char* name, 
        dsl_display_type_rgba_color_provider_cb provider, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Color name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            m_displayTypes[name] = DSL_RGBA_ON_DEMAND_COLOR_NEW(name, 
                provider, clientData);

            LOG_INFO("New RGBA Color On-Demand '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Color On-Demand '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorPaletteNew(const char* name, 
        const char** colors, uint num_colors)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Color name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
                std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>(
                    new std::vector<DSL_RGBA_COLOR_PTR>);
                    
            for (uint i = 0; i < num_colors; i++)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, colors[i]);

                DSL_RGBA_COLOR_PTR pColor = 
                    std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[colors[i]]);
                    
                pColorPalette->push_back(pColor);
            }
            m_displayTypes[name] = DSL_RGBA_COLOR_PALETTE_NEW(name, 
                pColorPalette);

            LOG_INFO("New RGBA Color Palette '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Color Palette '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorPalettePredefinedNew(const char* name,
        uint paletteId, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Predefined Color Palette name '" << name 
                    << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if (paletteId > DSL_COLOR_PREDEFINED_PALETTE_GREY)
            {
                LOG_ERROR("Invalid palette_id value of " << paletteId 
                    << " for New RGBA Predefined Color Palette '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( alpha > 1.0)
            {
                LOG_ERROR("Invalid alpha parameter = " << alpha 
                    << " greater than 1.0 for RGBA Color Palette'" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            
            // new shared pointer to an emptry vector of RGBA Color pointers to create
            // a palette of Predefined colors to provide to the Color Palette constructor
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
               std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                    new std::vector<DSL_RGBA_COLOR_PTR>};
            
            for (auto const& ivec: 
                RgbaPredefinedColor::s_predefinedColorPalettes[paletteId])
            {
                DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                    (new RgbaColor("", ivec));
                pColor->alpha = alpha;
                
                pColorPalette->push_back(pColor);
            }
            
            m_displayTypes[name] = DSL_RGBA_COLOR_PALETTE_NEW(name, 
                pColorPalette);

            LOG_INFO("New RGBA Predefined Color Palette '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Predefined Color Palette '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorPaletteRandomNew(const char* name, 
            uint size, uint hue, uint luminosity, double alpha, uint seed)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Random Color Palette name '" << name 
                    << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if (size < 2)
            {
                LOG_ERROR("Invalid size value of " << size 
                    << " for New RGBA Random Color Palette '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if (hue > DSL_COLOR_HUE_BROWN)
            {
                LOG_ERROR("Invalid hue value of " << hue 
                    << " for New RGBA Random Color Palette '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if (luminosity > DSL_COLOR_LUMINOSITY_RANDOM)
            {
                LOG_ERROR("Invalid luminosity value of " << luminosity 
                    << " for New RGBA Random Color Palette '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            if ( alpha > 1.0)
            {
                LOG_ERROR("Invalid alpha parameter = " << alpha 
                    << " greater than 1.0 for RGBA Color Palette '" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            
            // new shared pointer to an emptry vector of RGBA Color pointers to create
            // a palette of Random colors to provide to the Color Palette constructor
            std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>> pColorPalette = 
               std::shared_ptr<std::vector<DSL_RGBA_COLOR_PTR>>{
                    new std::vector<DSL_RGBA_COLOR_PTR>};

            // create a temporary Random RGBA color to generate a set of
            // random colors for our Color Palette.
            DSL_RGBA_COLOR_PTR pRandomColor= DSL_RGBA_RANDOM_COLOR_NEW("", 
                hue, luminosity, alpha, seed);
                
            for (uint i=0; i<size; i++)
            {
                // create a new RGBA color from the Random color and add
                // it to the vector or colors 
                DSL_RGBA_COLOR_PTR pColor = std::shared_ptr<RgbaColor>
                    (new RgbaColor("", *pRandomColor));
                
                pColorPalette->push_back(pColor);
                
                // Get the next random color
                pRandomColor->SetNext();
            }
            
            m_displayTypes[name] = DSL_RGBA_COLOR_PALETTE_NEW(name, 
                pColorPalette);

            LOG_INFO("New RGBA Random Color Palette '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Random Color Palette '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorPaletteIndexGet(const char* name,
        uint* index)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                name, RgbaColorPalette);
            
            DSL_RGBA_COLOR_PALETTE_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColorPalette>(m_displayTypes[name]);
            
            *index = pColor->GetIndex();
            
            LOG_INFO("RGBA Color Palette '" << name 
                << "' returned index = " << *index << "correctly");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RGBA Color Palette '" << name 
                << "' threw exception on Get Index");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorPaletteIndexSet(const char* name,
        uint index)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                name, RgbaColorPalette);
            
            DSL_RGBA_COLOR_PALETTE_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColorPalette>(m_displayTypes[name]);
            
            if (!pColor->SetIndex(index))
            {
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            
            LOG_INFO("RGBA Color Palette '" << name 
                << "' set index = " << index << "correctly");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Color Palette '" << name 
                << "' threw exception on Set Index");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaColorNextSet(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, name);
            
            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[name]);
            
            pColor->SetNext();
            
            LOG_INFO("Dynamic RGBA Color '" << name << "' set next successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Color On-Demand '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeRgbaFontNew(const char* name, 
        const char* font, uint size, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Font name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

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

    DslReturnType Services::DisplayTypeRgbaTextNew(const char* name, 
        const char* text,         uint xOffset, uint yOffset, const char* font, 
        boolean hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Text name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
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
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_LINE_NEW(name, 
                x1, y1, x2, y2, width, pColor);

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
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }

            if (head > DSL_ARROW_BOTH_HEAD)
            {
                LOG_ERROR("RGBA Head Type Invalid for RGBA Arrow'" << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);
            
            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_ARROW_NEW(name, 
                x1, y1, x2, y2, width, head, pColor);

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
        uint width, uint height, uint borderWidth, const char* color, bool hasBgColor, 
        const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Rectangle name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }
            
            m_displayTypes[name] = DSL_RGBA_RECTANGLE_NEW(name, 
                left, top, width, height, borderWidth, pColor, hasBgColor, pBgColor);

            LOG_INFO("New RGBA Rectangle '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Rectangle '" << name 
                << "' threw exception on create");
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
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if (numCoordinates > DSL_MAX_POLYGON_COORDINATES)
            {
                LOG_ERROR("Max coordinates exceeded created RGBA Polygon name '" 
                    << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_POLYGON_NEW(name, 
                coordinates, numCoordinates, borderWidth, pColor);

            LOG_INFO("New RGBA Polygon '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Polygon '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DisplayTypeRgbaLineMultiNew(const char* name, 
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
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            if (numCoordinates > DSL_MAX_MULTI_LINE_COORDINATES)
            {
                LOG_ERROR("Max coordinates exceeded creating RGBA Multi-Line name '" 
                    << name << "'");
                return DSL_RESULT_DISPLAY_PARAMETER_INVALID;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);
            
            m_displayTypes[name] = DSL_RGBA_MULTI_LINE_NEW(name, 
                coordinates, numCoordinates, borderWidth, pColor);

            LOG_INFO("New RGBA Multi Line '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New RGBA Multi Line '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DisplayTypeRgbaCircleNew(const char* name, 
        uint xCenter, uint yCenter, uint radius,const char* color, 
        bool hasBgColor, const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("RGBA Rectangle name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
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
            LOG_ERROR("New RGBA Circle '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::DisplayTypeSourceNumberNew(const char* name,
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, 
        const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Number name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_SOURCE_NUMBER_NEW(name,
                xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New Source Number '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New New Source Number '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeSourceNameNew(const char* name,
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, 
        const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Name name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
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
            LOG_ERROR("New Source Name '" << name
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeSourceDimensionsNew(const char* name, 
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, 
        const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Dimensions name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
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
            LOG_ERROR("New Source Dimensions '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayTypeSourceFrameRateNew(const char* name, 
        uint xOffset, uint yOffset, const char* font, boolean hasBgColor, 
        const char* bgColor)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure type name uniqueness 
            if (m_displayTypes.find(name) != m_displayTypes.end())
            {   
                LOG_ERROR("Source Frame-Rate name '" << name << "' is not unique");
                return DSL_RESULT_DISPLAY_TYPE_NAME_NOT_UNIQUE;
            }
            
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, font);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_displayTypes, 
                font, RgbaFont);

            DSL_RGBA_COLOR_PTR pBgColor(nullptr);
            if (hasBgColor)
            {
                DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, bgColor);
                DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, bgColor);

                pBgColor = std::dynamic_pointer_cast<RgbaColor>(
                    m_displayTypes[bgColor]);
            }
            else
            {
                pBgColor = DSL_RGBA_COLOR_NEW("_no_color_", 0.0, 0.0, 0.0, 0.0);
            }

            DSL_RGBA_FONT_PTR pFont = 
                std::dynamic_pointer_cast<RgbaFont>(m_displayTypes[font]);
            
            m_displayTypes[name] = DSL_SOURCE_FRAME_RATE_NEW(name,
                xOffset, yOffset, pFont, hasBgColor, pBgColor);

            LOG_INFO("New Source Frame-Rate '" << name 
                << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New Source Frame-Rate '" << name 
                << "' threw exception on create");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::DisplayRgbaTextShadowAdd(const char* name, 
        uint xOffset, uint yOffset, const char* color)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_TEXT(m_displayTypes, name);

            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, color);
            DSL_RETURN_IF_DISPLAY_TYPE_IS_NOT_COLOR(m_displayTypes, color);

            DSL_RGBA_COLOR_PTR pColor = 
                std::dynamic_pointer_cast<RgbaColor>(m_displayTypes[color]);

            DSL_RGBA_TEXT_PTR pText = 
                std::dynamic_pointer_cast<RgbaText>(m_displayTypes[name]);
            
            pText->AddShadow(xOffset, yOffset, pColor);
            
            LOG_INFO("Shadow added to RGBA Text '" << name 
                << "' successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("RGBA Text '" << name 
                << "' threw exception adding shadow");
            return DSL_RESULT_DISPLAY_TYPE_THREW_EXCEPTION;
        }
    }
            
            
    DslReturnType Services::DisplayTypeMetaAdd(const char* name, 
        void* pDisplayMeta, void* pFrameMeta)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_DISPLAY_TYPE_NAME_NOT_FOUND(m_displayTypes, name);
            
            DSL_DISPLAY_TYPE_PTR pDisplayType = 
                std::dynamic_pointer_cast<DisplayType>(m_displayTypes[name]);

//            pDisplayType->AddMeta((NvDsDisplayMeta*)pDisplayMeta, (NvDsFrameMeta*)pFrameMeta);
            
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